"""Force activation checkpointing on the ViT blocks and the feature pyramid.

Why this file exists
--------------------
The EVA02 config sets `use_act_checkpoint=True`, but in this environment the
flag does nothing -- the blocks arrive with a plain `Block.forward`, not a
wrapped one (most likely a `from fairscale.nn.checkpoint import
checkpoint_wrapper` sitting in a silent `try/except ImportError`).  Measured on
one A6000, EVA02-L at 1600x640, 17 frames, stop_prev_grad=4, one sample:

    retained by the gradient-carrying extract_img_feat call
      nothing checkpointed            ~37-43 GiB   (OOMs inside the backbone)
      ViT blocks only                  31.8 GiB    (OOMs at the cat)
      ViT blocks + pyramid stages       9.4 GiB    (iteration completes)

The pyramid is the larger term by far: ViTDet's SimpleFeaturePyramid builds its
stride-4 level by upsampling 4x, and the hand-written LayerNorm in
`batch_norm.py` materialises ~6 full-size intermediates, every one saved for
backward, in fp32 -- 1.46 GiB per tensor for a 24-image batch.

Accuracy
--------
None.  Checkpointing recomputes the forward instead of storing it; loss and
gradients are bit-identical.  Two details make that true here:

  * `use_reentrant=False` -- `frozen_blocks=N` means the tensor entering block N
    does not require grad, and the old reentrant checkpoint silently produces
    no gradient in that case.
  * RNG state is preserved (the default), so `drop_path_rate=0.3` replays the
    same stochastic-depth mask on recompute.  Without that the gradients WOULD
    be wrong.

The pyramid is conv / LayerNorm / GELU with no dropout, so its recompute is
deterministic outright.

Cost: one extra forward through the checkpointed modules during backward.
Measured at ~12% on the iteration (backward 6.4 s -> 7.2 s).

Usage
-----
    from models.act_ckpt import enable_act_checkpoint

    model = build_model(cfgs.model)
    model.init_weights()
    model = model.cuda()
    wrap_fp16_model(model)          # if the config uses Fp16OptimizerHook
    model.train()                   # BEFORE the call: `frozen_blocks` is
                                    # applied in train(), and frozen modules
                                    # are skipped (they save nothing)
    enable_act_checkpoint(model)
    # ... then wrap in DDP / MMDataParallel

DDP note: with `use_reentrant=False` this generally works, but if you hit
"Expected to mark a variable ready only once", enable a static graph on the
DDP wrapper (`ddp_model._set_static_graph()`) or pass
`find_unused_parameters=False`.
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as tcp

__all__ = ['enable_act_checkpoint', 'report_act_checkpoint',
           'find_blocks', 'find_fpn_stages']


def find_blocks(model):
    """The transformer block list, wherever the backbone keeps it."""
    for m in model.modules():
        b = getattr(m, 'blocks', None)
        if isinstance(b, nn.ModuleList) and len(b) >= 8:
            return b
    return None


def find_fpn_stages(model):
    """The feature-pyramid stage list.

    detectron2's SimpleFeaturePyramid keeps `self.stages` -- sometimes a plain
    Python list of add_module'd Sequentials, sometimes a ModuleList.  Either
    way the objects are the same, so patching their `forward` works.
    """
    found = []
    for m in model.modules():
        st = m.__dict__.get('stages', None)
        if st is None:
            st = getattr(m, 'stages', None)
        if st is None:
            continue
        try:
            items = list(st)
        except TypeError:
            continue
        if len(items) >= 2 and all(isinstance(s, nn.Module) for s in items) \
                and any(any(True for _ in s.parameters()) for s in items):
            found.append((m, items))
    return found


def _ckpt_forward(fn):
    def inner(*args, **kwargs):
        # under no_grad there is nothing to save, so don't pay the recompute
        if torch.is_grad_enabled():
            return tcp.checkpoint(fn, *args, use_reentrant=False, **kwargs)
        return fn(*args, **kwargs)
    inner.__qualname__ = 'forced_checkpoint(%s)' % getattr(
        fn, '__qualname__', 'forward')
    return inner


def _wrap(module):
    """True if this module was wrapped now (False: frozen or already wrapped)."""
    if not any(p.requires_grad for p in module.parameters()):
        return False                       # frozen: nothing would be saved
    if 'forward' in module.__dict__:
        return False                       # already wrapped
    module.forward = _ckpt_forward(module.forward)
    return True


def enable_act_checkpoint(model, verbose=True):
    """Wrap every trainable transformer block and pyramid stage.

    Safe to call more than once; already-wrapped modules are skipped.
    Returns the number of modules wrapped.
    """
    n = 0

    blocks = find_blocks(model)
    if blocks is None:
        if verbose:
            print('act_ckpt: no transformer block list found')
    else:
        k = sum(1 for b in blocks if _wrap(b))
        if verbose:
            print('act_ckpt: wrapped %d of %d transformer blocks'
                  % (k, len(blocks)))
        n += k

    for owner, stages in find_fpn_stages(model):
        k = sum(1 for s in stages if _wrap(s))
        if verbose and k:
            print('act_ckpt: wrapped %d of %d pyramid stages on %s'
                  % (k, len(stages), type(owner).__name__))
        n += k

    if verbose and n == 0:
        print('act_ckpt: nothing wrapped -- already active, or the module '
              'layout is not recognised')
    return n


def report_act_checkpoint(model):
    """Print whether the blocks are currently checkpointed."""
    b = find_blocks(model)
    if b is None:
        print('act_ckpt: no transformer block list found')
        return
    k = len(b) // 2
    blk = b[k]
    ntr = sum(1 for x in b if any(p.requires_grad for p in x.parameters()))
    print('act_ckpt: %d blocks (%d trainable), block[%d] is %s, forward = %s'
          '\n          -> %s'
          % (len(b), ntr, k, type(blk).__name__,
             getattr(blk.forward, '__qualname__', '?'),
             'WRAPPED' if 'forward' in blk.__dict__ else
             'NOT WRAPPED (activations stored, not recomputed)'))
