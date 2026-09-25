"""
FocalHead: 2D auxiliary supervision for PQR3D.

Ported from StreamPETR (https://github.com/exiawsh/StreamPETR),
`projects/mmdet3d_plugin/models/dense_heads/focal_head.py`.

Differences vs. the original StreamPETR implementation
------------------------------------------------------
1. The focal *token sampling* path (`sample_weight` / `topk_indexes`) is removed.
   In StreamPETR the head does double duty: auxiliary 2D loss AND selecting
   which image tokens the 3D decoder attends to. PQR3D does not use
   token-based cross attention -- it samples features by projecting 3D points
   through `lidar2img` -- so only the auxiliary-loss half is meaningful here.
2. `stride` is used everywhere instead of the hard-coded `16` in
   `_get_heatmap_single`.
2b. `stride` may be a *list* of strides. The head then runs on several FPN
   levels at once with shared weights, and all levels are pooled into a single
   Hungarian assignment -- see `forward()` for why that is sound and
   `_get_heatmap_single()` for the one place it is not free.
3. `forward()` takes an explicit `img_feats` tensor instead of `**data`.
4. `HungarianAssigner2D` and the small geometry helpers are inlined so that
   integration is a single new file.
5. `build_2d_gt_from_3d()` is added: it synthesises the 2D ground truth by
   projecting the 3D boxes through `lidar2img`, so PQR3D's data pipeline
   does not have to be touched at all.

Everything is registered in the standard mmdet registries, so it is usable
straight from a config.
"""

import numpy as np
import torch
import torch.nn as nn

from mmcv.cnn import bias_init_with_prob
from mmcv.runner import force_fp32
from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_overlaps, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply, reduce_mean)
from mmdet.core.bbox.assigners import AssignResult, BaseAssigner
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.match_costs import build_match_cost
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead

from .utils import inverse_sigmoid

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


# ---------------------------------------------------------------------------
# geometry / heatmap helpers (from StreamPETR's models/utils/misc.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def locations(features, stride, pad_h, pad_w):
    """Pixel centre of every feature-map cell, normalised to [0, 1].

    Args:
        features (Tensor): [N, C, H, W], only the spatial size is used.
        stride (int): stride of this feature level w.r.t. the input image.
        pad_h, pad_w (int): padded input image size.

    Returns:
        Tensor: [H, W, 2] in (x, y) order, normalised by (pad_w, pad_h).
    """
    h, w = features.size()[-2:]
    device = features.device

    shifts_x = (torch.arange(0, stride * w, step=stride,
                             dtype=torch.float32, device=device) + stride // 2) / pad_w
    shifts_y = (torch.arange(0, stride * h, step=stride,
                             dtype=torch.float32, device=device) + stride // 2) / pad_h

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    loc = torch.stack((shift_x.reshape(-1), shift_y.reshape(-1)), dim=1)
    return loc.reshape(h, w, 2)


def apply_ltrb(locations, pred_ltrb):
    """Turn per-cell (l, t, r, b) distances into normalised cxcywh boxes.

    Args:
        locations (Tensor): [N, H, W, 2] normalised cell centres.
        pred_ltrb (Tensor): [N, H, W, 4] in [0, 1] (already sigmoid'ed).
    """
    pred_boxes = torch.zeros_like(pred_ltrb)
    pred_boxes[..., 0] = locations[..., 0] - pred_ltrb[..., 0]  # x1
    pred_boxes[..., 1] = locations[..., 1] - pred_ltrb[..., 1]  # y1
    pred_boxes[..., 2] = locations[..., 0] + pred_ltrb[..., 2]  # x2
    pred_boxes[..., 3] = locations[..., 1] + pred_ltrb[..., 3]  # y2

    min_xy = pred_boxes.new_tensor(0.0)
    max_xy = pred_boxes.new_tensor(1.0)
    pred_boxes = torch.where(pred_boxes < min_xy, min_xy, pred_boxes)
    pred_boxes = torch.where(pred_boxes > max_xy, max_xy, pred_boxes)

    return bbox_xyxy_to_cxcywh(pred_boxes)


def apply_center_offset(locations, center_offset):
    """Turn per-cell offsets into normalised 2D centres in [0, 1]."""
    centers_2d = torch.zeros_like(center_offset)
    locations = inverse_sigmoid(locations)
    centers_2d[..., 0] = locations[..., 0] + center_offset[..., 0]
    centers_2d[..., 1] = locations[..., 1] + center_offset[..., 1]
    return centers_2d.sigmoid()


def gaussian_2d(shape, sigma=1.0):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_heatmap_gaussian(heatmap, center, radius, k=1):
    """Splat a 2D gaussian into `heatmap` (modified in place)."""
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    # centres are guaranteed inside the image by build_2d_gt_from_3d, but be safe
    if x < 0 or y < 0 or x >= width or y >= height:
        return heatmap

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = torch.from_numpy(
        gaussian[radius - top:radius + bottom,
                 radius - left:radius + right]).to(heatmap.device, torch.float32)

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap


# ---------------------------------------------------------------------------
# 2D ground truth, synthesised from the 3D boxes
# ---------------------------------------------------------------------------

# The 12 edges of the 3D box, in mmdet3d's `LiDARInstance3DBoxes.corners`
# ordering (unravel_index over [2,2,2] reindexed by [0,1,3,2,4,5,7,6]).
# Verified: exactly 12 pairs differing in one axis, 4 per axis.
BOX_EDGES = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3),
             (2, 6), (3, 7), (4, 5), (4, 7), (5, 6), (6, 7))


def project_box_clipped(corners_cam, img_h, img_w, min_depth=0.5):
    """Exact 2D extent of a 3D box, near-plane clipped and image clipped.

    Improves on StreamPETR's `get_2d_boxes` in two ways.

    1. Near plane. StreamPETR keeps only the corners with z > 0 and projects
       those (`in_front = argwhere(corners_3d[2,:] > 0)`), which *under*-
       estimates the extent: the box continues toward the camera past the last
       in-front corner, and that part projects wider. We instead clip each of
       the 12 box edges at z = min_depth and project the clip point, which is
       the true silhouette of the visible part.

    2. Image boundary. StreamPETR intersects the convex hull with the image
       canvas (shapely) and takes the bbox of the intersection -- correct. Our
       previous version took the bbox of all corners and clamped it to bounds,
       which *over*-estimates for rotated hulls: hull vertices (-10,0),
       (-10,100), (100,50) clamp to (0,0,100,100) but truly intersect the
       canvas at (0,4.5,100,95.5). We reproduce the exact intersection by
       intersecting the 12 projected edges with the four canvas sides -- same
       answer as shapely, no dependency, and vectorised.

    Silhouette edges of a convex polyhedron are always projections of real
    edges, so the 12 box edges are a sufficient (super)set.

    Known limitation: when a box straddles the near plane, the cross-section
    polygon it cuts on that plane has its own edges, which we do not test
    against the canvas sides. Only reachable for an object partly behind the
    camera AND truncated by the image border; the error is a slight
    under-estimate, and StreamPETR under-estimates those more.

    Args:
        corners_cam (Tensor): [M, 8, 4] corners after `lidar2img`, i.e.
            homogeneous image coordinates (u*z, v*z, z, w). The map is linear,
            so clipping by interpolating in this space is exact.

    Returns:
        boxes (Tensor): [M, 4] xyxy in pixels.
        valid (Tensor): [M] bool, False when nothing of the box is visible.
    """
    M = corners_cam.shape[0]
    dev, eps = corners_cam.device, 1e-6
    e_i = corners_cam.new_tensor([e[0] for e in BOX_EDGES], dtype=torch.long)
    e_j = corners_cam.new_tensor([e[1] for e in BOX_EDGES], dtype=torch.long)

    a = corners_cam[:, e_i, :3]                       # [M, 12, 3]
    b = corners_cam[:, e_j, :3]
    za, zb = a[..., 2], b[..., 2]
    in_a, in_b = za > min_depth, zb > min_depth
    edge_ok = in_a | in_b                             # [M, 12]

    # clip the out-of-front endpoint onto the near plane
    denom = torch.where((zb - za).abs() > eps, zb - za, torch.full_like(za, eps))
    t = ((min_depth - za) / denom).clamp(0.0, 1.0)[..., None]
    cut = a + t * (b - a)
    a = torch.where(in_a[..., None], a, cut)
    b = torch.where(in_b[..., None], b, cut)

    p0 = a[..., :2] / a[..., 2:3].clamp(min=min_depth)   # [M, 12, 2]
    p1 = b[..., :2] / b[..., 2:3].clamp(min=min_depth)

    X1, Y1 = float(img_w - 1), float(img_h - 1)
    cand, mask = [], []

    def inside(p):
        return (p[..., 0] >= 0) & (p[..., 0] <= X1) & (p[..., 1] >= 0) & (p[..., 1] <= Y1)

    for p in (p0, p1):                                  # endpoints inside canvas
        cand.append(p)
        mask.append(edge_ok & inside(p))

    d = p1 - p0
    for axis, val in ((0, 0.0), (0, X1), (1, 0.0), (1, Y1)):   # 4 canvas sides
        den = torch.where(d[..., axis].abs() > eps, d[..., axis],
                          torch.full_like(d[..., axis], eps))
        s = (val - p0[..., axis]) / den
        hit = p0 + s[..., None] * d
        other = 1 - axis
        ok = (edge_ok & (d[..., axis].abs() > eps) & (s >= 0) & (s <= 1)
              & (hit[..., other] >= 0) & (hit[..., other] <= (Y1 if other else X1)))
        cand.append(hit)
        mask.append(ok)

    pts = torch.cat(cand, dim=1)                        # [M, 72, 2]
    m = torch.cat(mask, dim=1)                          # [M, 72]
    valid = m.any(dim=1)

    big = torch.full_like(pts, float('inf'))
    lo = torch.where(m[..., None], pts, big).amin(dim=1)
    hi = torch.where(m[..., None], pts, -big).amax(dim=1)

    boxes = torch.cat([lo, hi], dim=-1)
    boxes = torch.where(valid[:, None], boxes, torch.zeros_like(boxes))
    return boxes, valid


def filter_invisible(boxes, depths, img_h, img_w):
    """Painter's-algorithm occlusion test -- StreamPETR's `_filter_invisible`.

    nuScenes annotates in 3D from LiDAR, so a box can be fully hidden behind a
    nearer object and still be a valid annotation. Projecting it anyway hands
    FocalHead an unlearnable *positive*: because the matching is one-to-one, a
    fully occluded object consumes a matched query and contributes cls + L1 +
    GIoU + centre losses plus a gaussian peak in the centerness heatmap, all
    demanding that e.g. van pixels predict a hidden pedestrian.

    Boxes are painted far -> near into an index map; whatever index still
    survives somewhere is visible. Partially visible objects are kept
    automatically -- a single surviving pixel is enough.

    Caveat: the test uses axis-aligned rectangles, so it can over-delete. A
    pedestrian standing in the gap between two parked cars may have its
    rectangle fully covered by the union of the two car rectangles despite
    being plainly visible. tools/check_2d_gt.py draws dropped boxes as grey
    dashes and prints per-class drop rates; `--no-filter` disables the test.

    Differs from StreamPETR in one detail: upstream initialises the index map
    with `np.zeros`, so index 0 -- the *farthest* box -- is always present in
    `np.unique` and therefore never dropped, however occluded it is. We
    initialise to -1 and drop it properly.

    Args:
        boxes (Tensor): [K, 4] xyxy in pixels.
        depths (Tensor): [K] camera-frame depth of each object's centre.

    Returns:
        Tensor: [K] bool keep-mask, in the original box order.
    """
    K = boxes.shape[0]
    if K <= 1:
        return torch.ones(K, dtype=torch.bool, device=boxes.device)

    b = boxes.detach().cpu().numpy()
    d = depths.detach().cpu().numpy()

    order = np.argsort(-d, kind='stable')  # far -> near
    x1 = np.ceil(b[order, 0]).astype(np.int64)
    y1 = np.ceil(b[order, 1]).astype(np.int64)
    x2 = np.floor(b[order, 2]).astype(np.int64)
    y2 = np.floor(b[order, 3]).astype(np.int64)

    imap = np.full((img_h, img_w), -1, dtype=np.int64)
    for i in range(K):
        imap[y1[i]:y2[i], x1[i]:x2[i]] = i  # nearer boxes overwrite farther ones

    visible = np.unique(imap)
    visible = visible[visible >= 0]

    keep = np.zeros(K, dtype=bool)
    keep[order[visible]] = True
    return torch.from_numpy(keep).to(boxes.device)


@torch.no_grad()
def build_2d_gt_from_3d(gt_bboxes_3d,
                        gt_labels_3d,
                        img_metas,
                        num_cams=6,
                        min_box_size=4.0,
                        min_depth=0.5,
                        filter_occluded=True,
                        occlusion_depth='center',
                        return_near_depth=False,
                        projection='clip'):
    """Project 3D GT boxes into the *current-frame* cameras.

    This works without touching the data pipeline because PQR3D bakes every
    image augmentation into `lidar2img`:
      * `RandomTransformImage`   left-multiplies the IDA matrix,
      * `GlobalRotScaleTransImage` rotates/scales `gt_bboxes_3d` and
        right-multiplies the inverse into `lidar2img`.
    So `lidar2img @ gt_bboxes_3d.corners` lands in the augmented image frame.

    Only cameras [0:6] are used. Sweep frames are unannotated (nuScenes labels
    key-frames at 2 Hz) and moving objects have shifted between timestamps, so
    projecting current GT into them would be pure label noise.

    `projection='clip'` (default) computes the exact visible extent: every box
    edge is clipped at the near plane and against the image canvas -- see
    `project_box_clipped`. This keeps partially-behind-camera and
    border-truncated objects that StreamPETR's converter also keeps, and is
    tighter than StreamPETR on truncated boxes. `projection='simple'` restores
    the earlier conservative rule (all 8 corners in front, centre inside the
    frame, bbox clamped to bounds) so the two can be compared directly.
    `projection='tight'` is 'simple' plus only the two changes where 'simple' is
    strictly worse than StreamPETR -- exact canvas intersection instead of a
    clamp, and no centre-inside-frame rejection. It creates no frame-spanning
    boxes, because near-plane clipping cannot trigger when all 8 corners are
    required to be in front.

    With `filter_occluded=True` (default) a painter's-algorithm visibility test
    then removes objects with no visible pixel -- see `filter_invisible`. This
    is what makes `depths` load-bearing: it is unused inside FocalHead's loss,
    but it is the sort key for the occlusion test.

    Args:
        gt_bboxes_3d (list[LiDARInstance3DBoxes]): length B.
        gt_labels_3d (list[Tensor]): length B, each [M].
        img_metas (list[dict]): length B, must contain 'lidar2img' and 'pad_shape'.

    Returns:
        tuple of 4 nested lists, each [B][num_cams]:
            gt_bboxes2d: [K, 4] xyxy, absolute pixels
            gt_labels2d: [K]
            centers2d:   [K, 2] projected 3D gravity centre, absolute pixels
            depths:      [K] camera-frame depth of the centre
    """
    device = gt_labels_3d[0].device
    gt_bboxes2d, gt_labels2d, centers2d, depths = [], [], [], []
    near_depths = []

    for b in range(len(gt_bboxes_3d)):
        boxes3d = gt_bboxes_3d[b]
        labels = gt_labels_3d[b].to(device)
        pad_h, pad_w = img_metas[b]['pad_shape'][0][:2]

        l2i = np.asarray(img_metas[b]['lidar2img'][:num_cams], dtype=np.float32)
        l2i = torch.from_numpy(l2i).to(device)  # [N, 4, 4]

        # 8 box corners + the gravity centre -> [M, 9, 4] homogeneous
        corners = boxes3d.corners.to(device)                       # [M, 8, 3]
        centers = boxes3d.gravity_center.to(device)                # [M, 3]
        pts = torch.cat([corners, centers[:, None, :]], dim=1)     # [M, 9, 3]
        pts = torch.cat([pts, torch.ones_like(pts[..., :1])], -1)  # [M, 9, 4]
        M = pts.shape[0]

        cam_boxes, cam_labels, cam_centers, cam_depths = [], [], [], []
        cam_near = []
        for c in range(num_cams):
            if M == 0:
                cam_boxes.append(torch.zeros((0, 4), device=device))
                cam_labels.append(torch.zeros((0,), dtype=torch.long, device=device))
                cam_centers.append(torch.zeros((0, 2), device=device))
                cam_depths.append(torch.zeros((0,), device=device))
                cam_near.append(torch.zeros((0,), device=device))
                continue

            cam = pts @ l2i[c].t()                                  # [M, 9, 4]
            z = cam[..., 2]
            uv = cam[..., :2] / z.clamp(min=1e-5)[..., None]        # [M, 9, 2]

            corner_uv, corner_z = uv[:, :8], z[:, :8]
            cu, cv, cz = uv[:, 8, 0], uv[:, 8, 1], z[:, 8]
            near_z = corner_z.clamp(min=min_depth).min(dim=1).values

            if projection == 'tight':
                # 'simple' geometry (all 8 corners in front) but with the exact
                # canvas intersection instead of a clamp. Boxes can only get
                # SMALLER than 'simple', never larger, and no near-plane
                # clipping runs -- so no frame-spanning boxes are created.
                # The centre may fall outside the frame; that is a valid amodal
                # target, matching StreamPETR, so it is not a rejection rule.
                cbox, vis = project_box_clipped(cam[:, :8], int(pad_h), int(pad_w),
                                                min_depth)
                x1, y1, x2, y2 = cbox.unbind(-1)
                keep = vis & (corner_z > min_depth).all(dim=1) & (cz > min_depth)
            elif projection == 'clip':
                # exact near-plane + canvas clipping (see project_box_clipped)
                cbox, vis = project_box_clipped(cam[:, :8], int(pad_h), int(pad_w),
                                                min_depth)
                x1, y1, x2, y2 = cbox.unbind(-1)
                # the 2D centre may fall outside the frame for truncated
                # objects -- that is a valid amodal target, so it is NOT a
                # rejection criterion. It only has to be in front of the camera
                # for the projection to mean anything.
                keep = vis & (cz > min_depth)
            else:
                # 'simple': previous behaviour, kept so the change is A/B-able
                x1 = corner_uv[..., 0].min(dim=1).values.clamp(0, pad_w - 1)
                y1 = corner_uv[..., 1].min(dim=1).values.clamp(0, pad_h - 1)
                x2 = corner_uv[..., 0].max(dim=1).values.clamp(0, pad_w - 1)
                y2 = corner_uv[..., 1].max(dim=1).values.clamp(0, pad_h - 1)
                keep = ((corner_z > min_depth).all(dim=1)
                        & (cz > min_depth)
                        & (cu >= 0) & (cu <= pad_w - 1)
                        & (cv >= 0) & (cv <= pad_h - 1))

            keep = keep & (x2 - x1 >= min_box_size) & (y2 - y1 >= min_box_size)

            c_boxes = torch.stack([x1, y1, x2, y2], dim=-1)[keep]
            c_labels, c_centers, c_depths = labels[keep], torch.stack([cu, cv], -1)[keep], cz[keep]
            c_near = near_z[keep]

            if filter_occluded and c_boxes.shape[0] > 1:
                # 'center' matches StreamPETR. 'near' sorts by the closest
                # corner instead, which stops long objects (trucks, buses) from
                # being ordered behind shorter ones they actually occlude --
                # their centre is far even when their nose is close.
                sort_z = c_depths if occlusion_depth == 'center' else c_near
                vis = filter_invisible(c_boxes, sort_z, int(pad_h), int(pad_w))
                c_boxes, c_labels = c_boxes[vis], c_labels[vis]
                c_centers, c_depths, c_near = c_centers[vis], c_depths[vis], c_near[vis]

            cam_boxes.append(c_boxes)
            cam_labels.append(c_labels)
            cam_centers.append(c_centers)
            cam_depths.append(c_depths)
            cam_near.append(c_near)

        gt_bboxes2d.append(cam_boxes)
        gt_labels2d.append(cam_labels)
        centers2d.append(cam_centers)
        depths.append(cam_depths)
        near_depths.append(cam_near)

    if return_near_depth:
        return gt_bboxes2d, gt_labels2d, centers2d, depths, near_depths
    return gt_bboxes2d, gt_labels2d, centers2d, depths


# ---------------------------------------------------------------------------
# assigner
# ---------------------------------------------------------------------------

@BBOX_ASSIGNERS.register_module()
class HungarianAssigner2D(BaseAssigner):
    """DETR-style one-to-one matching with an extra 2D-centre cost."""

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 iou_cost=dict(type='IoUCost', iou_mode='giou', weight=1.0),
                 centers2d_cost=dict(type='BBox3DL1Cost', weight=1.0)):
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_cost = build_match_cost(iou_cost)
        self.centers2d_cost = build_match_cost(centers2d_cost)

    def assign(self, bbox_pred, cls_pred, pred_centers2d, gt_bboxes, gt_labels,
               centers2d, img_meta, gt_bboxes_ignore=None, eps=1e-7):
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        assigned_gt_inds = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            if num_gts == 0:
                assigned_gt_inds[:] = 0
            return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)

        img_h, img_w = img_meta['pad_shape'][:2]
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)

        cls_cost = self.cls_cost(cls_pred, gt_labels)
        reg_cost = self.reg_cost(bbox_pred, gt_bboxes / factor)
        iou_cost = self.iou_cost(bbox_cxcywh_to_xyxy(bbox_pred) * factor, gt_bboxes)
        centers2d_cost = self.centers2d_cost(pred_centers2d, centers2d / factor[:, 0:2])

        cost = cls_cost + reg_cost + iou_cost + centers2d_cost
        cost = torch.nan_to_num(cost, nan=100.0, posinf=100.0, neginf=-100.0)

        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost.detach().cpu())
        matched_row_inds = torch.from_numpy(matched_row_inds).to(bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(bbox_pred.device)

        assigned_gt_inds[:] = 0
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)


# ---------------------------------------------------------------------------
# the head
# ---------------------------------------------------------------------------

@HEADS.register_module()
class FocalHead(AnchorFreeHead):
    """Dense 2D head applied to one or more FPN levels of the current cameras.

    Predicts, per feature cell: class logits, centerness, an (l, t, r, b) box
    and a 2D object centre. Trained with Hungarian matching against 2D boxes
    obtained by projecting the 3D GT.

    Multi-level
    -----------
    Pass `stride=(8, 16, 32)` (any length, any order matching `aux_2d_level`)
    to run on several levels. One set of weights is shared across levels and
    the cells of all levels are *concatenated* before matching, so there is a
    single `linear_sum_assignment` over the union. Two consequences worth
    knowing:

    * One-to-one semantics survive. Each object is claimed by exactly one cell
      at whichever level fits it best -- learnt, not imposed by an FCOS-style
      size->level rule. `num_total_pos` stays equal to the number of GT boxes,
      so loss_cls / loss_bbox / loss_iou / loss_centers2d keep exactly the
      scale they had in the single-level runs.
    * The centerness heatmap does NOT: every object is drawn on every level's
      map, so the positive count is num_gt * num_levels. `loss_single`
      divides by that product, which keeps the reported number comparable to
      the single-level runs at the cost of 1/num_levels of the per-level
      gradient. Set `loss_centerness.loss_weight = num_levels` to get the
      un-normalised behaviour back.
    """

    def __init__(self,
                 num_classes,
                 in_channels=256,
                 embed_dims=256,
                 stride=16,             # int, or a sequence for multi-level
                 heatmap_radius_max=None,  # cap the gaussian radius (cells)
                 sync_cls_avg_factor=False,
                 loss_cls2d=dict(type='QualityFocalLoss', use_sigmoid=True,
                                 beta=2.0, loss_weight=2.0),
                 loss_centerness=dict(type='GaussianFocalLoss', reduction='mean',
                                      loss_weight=1.0),
                 loss_bbox2d=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou2d=dict(type='GIoULoss', loss_weight=2.0),
                 loss_centers2d=dict(type='L1Loss', loss_weight=10.0),
                 train_cfg=dict(
                     assigner2d=dict(
                         type='HungarianAssigner2D',
                         cls_cost=dict(type='FocalLossCost', weight=2.0),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                         iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                         centers2d_cost=dict(type='BBox3DL1Cost', weight=10.0))),
                 test_cfg=dict(max_per_img=100),
                 init_cfg=None,
                 **kwargs):
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor

        if train_cfg:
            assert 'assigner2d' in train_cfg, \
                'assigner2d should be provided when train_cfg is set.'
            self.assigner2d = build_assigner(train_cfg['assigner2d'])
            # DETR-style: sampling is disabled, use a pseudo sampler
            self.sampler = build_sampler(dict(type='PseudoSampler'), context=self)

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.heatmap_radius_max = heatmap_radius_max

        # accept an int (single level, the original behaviour) or any sequence
        _strides = ((int(stride),) if isinstance(stride, (int, float))
                    else tuple(int(s) for s in stride))
        assert len(_strides) > 0 and all(s > 0 for s in _strides), \
            'stride must be a positive int or a non-empty sequence of them'

        super(FocalHead, self).__init__(num_classes, in_channels, init_cfg=init_cfg)

        # NOTE: AnchorFreeHead.__init__ assigns self.strides = (4, 8, 16, 32),
        # so ours has to be written AFTER the super() call, not before.
        self.strides = _strides
        self.num_levels = len(_strides)
        # backward compatibility: an int when single-level, so old callers that
        # do `pad_h // head.stride` keep working. Multi-level leaves the tuple
        # in place, which makes such a caller fail loudly instead of silently.
        self.stride = _strides[0] if self.num_levels == 1 else _strides

        self.loss_cls2d = build_loss(loss_cls2d)
        self.loss_bbox2d = build_loss(loss_bbox2d)
        self.loss_iou2d = build_loss(loss_iou2d)
        self.loss_centers2d = build_loss(loss_centers2d)
        self.loss_centerness = build_loss(loss_centerness)

        self._init_layers()

    def _init_layers(self):
        self.shared_cls = nn.Sequential(
            nn.Conv2d(self.in_channels, self.embed_dims, kernel_size=3, padding=1),
            nn.GroupNorm(32, num_channels=self.embed_dims),
            nn.ReLU())
        self.shared_reg = nn.Sequential(
            nn.Conv2d(self.in_channels, self.embed_dims, kernel_size=3, padding=1),
            nn.GroupNorm(32, num_channels=self.embed_dims),
            nn.ReLU())

        self.cls = nn.Conv2d(self.embed_dims, self.num_classes, kernel_size=1)
        self.centerness = nn.Conv2d(self.embed_dims, 1, kernel_size=1)
        self.ltrb = nn.Conv2d(self.embed_dims, 4, kernel_size=1)
        self.center2d = nn.Conv2d(self.embed_dims, 2, kernel_size=1)

        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.cls.bias, bias_init)
        nn.init.constant_(self.centerness.bias, bias_init)

    def forward(self, location, img_feats):
        """
        Args:
            location: [B*N, h, w, 2] normalised cell centres, or a list of such
                tensors, one per level (in the same order as `self.strides`).
            img_feats: [B, N, C, h, w] current-frame camera features, or a list
                of such tensors, one per level.

        Returns:
            dict with the cells of every level concatenated along dim 1:
                enc_cls_scores  [B*N, sum_l(h_l*w_l), num_classes]
                enc_bbox_preds  [B*N, sum_l(h_l*w_l), 4]
                pred_centers2d  [B*N, sum_l(h_l*w_l), 2]
                centerness      [B*N, sum_l(h_l*w_l), 1]

        Concatenating across strides is only legitimate because `apply_ltrb`
        and `apply_center_offset` emit coordinates normalised by (pad_w, pad_h),
        not by the feature grid: a box predicted at stride 8 and one predicted
        at stride 32 are already in the same units, so the pooled Hungarian
        cost matrix compares like with like.
        """
        if torch.is_tensor(img_feats):          # single-level call, unchanged
            location, img_feats = [location], [img_feats]
        assert len(img_feats) == self.num_levels == len(location), (
            'FocalHead was built with strides=%s but got %d feature level(s)'
            % (self.strides, len(img_feats)))

        cls_logits, centerness, pred_bboxes, pred_centers2d = [], [], [], []

        for loc_l, feat_l in zip(location, img_feats):
            bs, n = feat_l.shape[:2]
            x = feat_l.flatten(0, 1)  # [B*N, C, h, w]

            cls_feat = self.shared_cls(x)
            cls_logits.append(self.cls(cls_feat).permute(0, 2, 3, 1)
                              .reshape(bs * n, -1, self.num_classes))
            centerness.append(self.centerness(cls_feat).permute(0, 2, 3, 1)
                              .reshape(bs * n, -1, 1))

            reg_feat = self.shared_reg(x)
            ltrb = self.ltrb(reg_feat).permute(0, 2, 3, 1).contiguous().sigmoid()
            center_offset = self.center2d(reg_feat).permute(0, 2, 3, 1).contiguous()

            pred_bboxes.append(apply_ltrb(loc_l, ltrb).view(bs * n, -1, 4))
            pred_centers2d.append(
                apply_center_offset(loc_l, center_offset).view(bs * n, -1, 2))

        # level-major, then row-major inside each level. `_get_heatmap_single`
        # flattens the heatmaps in exactly this order.
        return dict(
            enc_cls_scores=torch.cat(cls_logits, dim=1),
            enc_bbox_preds=torch.cat(pred_bboxes, dim=1),
            pred_centers2d=torch.cat(pred_centers2d, dim=1),
            centerness=torch.cat(centerness, dim=1),
        )

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes2d_list, gt_labels2d_list, centers2d, depths,
             preds_dicts, img_metas, gt_bboxes_ignore=None):
        """All GT lists are nested [B][N] to match `img_feats.flatten(0, 1)`,
        i.e. batch-major / camera-minor."""
        assert gt_bboxes_ignore is None

        # flatten [B][N] -> [B*N] in the exact order the features were flattened
        all_gt_bboxes2d = [b for i in gt_bboxes2d_list for b in i]
        all_gt_labels2d = [l for i in gt_labels2d_list for l in i]
        all_centers2d = [c for i in centers2d for c in i]
        all_depths = [d for i in depths for d in i]

        loss_cls, loss_bbox, loss_iou, loss_centers2d, loss_centerness = self.loss_single(
            preds_dicts['enc_cls_scores'], preds_dicts['enc_bbox_preds'],
            preds_dicts['pred_centers2d'], preds_dicts['centerness'],
            all_gt_bboxes2d, all_gt_labels2d, all_centers2d, all_depths,
            img_metas, gt_bboxes_ignore)

        return dict(
            aux2d_loss_cls=loss_cls,
            aux2d_loss_bbox=loss_bbox,
            aux2d_loss_iou=loss_iou,
            aux2d_loss_centers2d=loss_centers2d,
            aux2d_loss_centerness=loss_centerness,
        )

    def loss_single(self, cls_scores, bbox_preds, pred_centers2d, centerness,
                    gt_bboxes_list, gt_labels_list, all_centers2d_list,
                    all_depths_list, img_metas, gt_bboxes_ignore_list=None):
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        centers2d_preds_list = [pred_centers2d[i] for i in range(num_imgs)]

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         centers2d_targets_list, num_total_pos, num_total_neg) = self.get_targets(
            cls_scores_list, bbox_preds_list, centers2d_preds_list,
            gt_bboxes_list, gt_labels_list, all_centers2d_list,
            all_depths_list, img_metas, gt_bboxes_ignore_list)

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        centers2d_targets = torch.cat(centers2d_targets_list, 0)

        img_h, img_w = img_metas[0]['pad_shape'][0][:2]

        factors = []
        for bbox_pred in bbox_preds:
            factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h])
            factors.append(factor.unsqueeze(0).repeat(bbox_pred.size(0), 1))
        factors = torch.cat(factors, 0)

        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # GIoU loss.
        # NOTE: this uses the *local* (per-GPU) positive count, while the box /
        # centre / centerness losses below use the DDP-averaged one, because
        # reduce_mean is only applied after loss_cls. That asymmetry is copied
        # verbatim from StreamPETR. Under DDP the two differ only by the
        # per-GPU variation in object count, so the effect is small -- but if
        # you want them consistent, move the reduce_mean block above this line.
        loss_iou = self.loss_iou2d(bboxes, bboxes_gt, bbox_weights,
                                   avg_factor=max(num_total_pos, 1))
        iou_score = bbox_overlaps(bboxes_gt, bboxes, is_aligned=True).reshape(-1)

        # quality focal classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls2d(cls_scores, (labels, iou_score.detach()),
                                   label_weights, avg_factor=cls_avg_factor)

        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # centerness heatmap loss.
        # Every object is drawn on every level, so the number of target-1 cells
        # is num_gt * num_levels; dividing by that keeps this loss on the same
        # scale as the single-level runs. Set loss_centerness.loss_weight =
        # num_levels in the config to recover the un-normalised behaviour.
        img_shape = [img_metas[0]['pad_shape'][0]] * num_imgs
        (heatmaps,) = multi_apply(self._get_heatmap_single, all_centers2d_list,
                                  gt_bboxes_list, img_shape)
        heatmaps = torch.stack(heatmaps, dim=0)
        centerness = torch.clamp(centerness.sigmoid(), min=1e-4, max=1 - 1e-4)
        loss_centerness = self.loss_centerness(
            centerness, heatmaps.view(num_imgs, -1, 1),
            avg_factor=max(num_total_pos * self.num_levels, 1))

        # L1 box loss
        loss_bbox = self.loss_bbox2d(bbox_preds, bbox_targets, bbox_weights,
                                     avg_factor=num_total_pos)

        # L1 centre loss
        pred_centers2d = pred_centers2d.view(-1, 2)
        loss_centers2d = self.loss_centers2d(pred_centers2d, centers2d_targets,
                                             bbox_weights[:, 0:2], avg_factor=num_total_pos)

        return (torch.nan_to_num(loss_cls), torch.nan_to_num(loss_bbox),
                torch.nan_to_num(loss_iou), torch.nan_to_num(loss_centers2d),
                torch.nan_to_num(loss_centerness))

    def _get_heatmap_single(self, obj_centers2d, obj_bboxes, img_shape):
        """One flattened heatmap per level, concatenated level-major.

        The result lines up cell-for-cell with `centerness` from `forward()`.

        The gaussian radius is `min(l, t, r, b) / stride`, i.e. it scales with
        the level: the same bus gets radius 2 at stride 32 and radius 8 at
        stride 8. That is intentional -- the blob covers the same image area
        either way -- but on the finest level a large object turns a good
        fraction of the map into (1 - target)^4 ~ 0 ignore-region. Cap it with
        `heatmap_radius_max` if that becomes a problem.
        """
        img_h, img_w = img_shape[:2]
        dev = obj_centers2d.device

        if len(obj_centers2d) != 0:
            l = obj_centers2d[..., 0:1] - obj_bboxes[..., 0:1]
            t = obj_centers2d[..., 1:2] - obj_bboxes[..., 1:2]
            r = obj_bboxes[..., 2:3] - obj_centers2d[..., 0:1]
            b = obj_bboxes[..., 3:4] - obj_centers2d[..., 1:2]
            bound = torch.cat([l, t, r, b], dim=-1)
            min_bound = torch.min(bound, dim=-1)[0]
        else:
            min_bound = None

        heatmaps = []
        for s in self.strides:
            heatmap = torch.zeros(img_h // s, img_w // s, device=dev)
            if min_bound is not None:
                radius = torch.clamp(torch.ceil(min_bound / s), 1.0)
                if self.heatmap_radius_max is not None:
                    radius = torch.clamp(radius, max=float(self.heatmap_radius_max))
                radius = radius.cpu().numpy().tolist()
                for center, rad in zip(obj_centers2d, radius):
                    heatmap = draw_heatmap_gaussian(heatmap, center / s,
                                                    radius=int(rad), k=1)
            heatmaps.append(heatmap.reshape(-1))

        return (torch.cat(heatmaps, dim=0),)

    def get_targets(self, cls_scores_list, bbox_preds_list, centers2d_preds_list,
                    gt_bboxes_list, gt_labels_list, all_centers2d_list,
                    all_depths_list, img_metas, gt_bboxes_ignore_list=None):
        assert gt_bboxes_ignore_list is None
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        img_meta = {'pad_shape': img_metas[0]['pad_shape'][0]}
        img_meta_list = [img_meta for _ in range(num_imgs)]

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         centers2d_targets_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            centers2d_preds_list, gt_bboxes_list, gt_labels_list,
            all_centers2d_list, all_depths_list, img_meta_list, gt_bboxes_ignore_list)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
                centers2d_targets_list, num_total_pos, num_total_neg)

    def _get_target_single(self, cls_score, bbox_pred, pred_centers2d, gt_bboxes,
                           gt_labels, centers2d, depths, img_meta, gt_bboxes_ignore=None):
        # NOTE: `depths` is unused *here* -- kept for signature parity with
        # StreamPETR. It is load-bearing upstream of this call: it is the sort
        # key for the occlusion test in build_2d_gt_from_3d.
        num_bboxes = bbox_pred.size(0)

        assign_result = self.assigner2d.assign(bbox_pred, cls_score, pred_centers2d,
                                               gt_bboxes, gt_labels, centers2d,
                                               img_meta, gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        labels = gt_bboxes.new_full((num_bboxes,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds].long()
        label_weights = gt_bboxes.new_ones(num_bboxes)

        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        img_h, img_w = img_meta['pad_shape'][:2]
        factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
        bbox_targets[pos_inds] = bbox_xyxy_to_cxcywh(sampling_result.pos_gt_bboxes / factor)

        centers2d_targets = bbox_pred.new_full((num_bboxes, 2), 0.0, dtype=torch.float32)
        if gt_bboxes.numel() == 0:
            assert sampling_result.pos_assigned_gt_inds.numel() == 0
            centers2d_labels = torch.empty_like(gt_bboxes).view(-1, 2)
        else:
            centers2d_labels = centers2d[sampling_result.pos_assigned_gt_inds.long(), :]
        centers2d_targets[pos_inds] = centers2d_labels / factor[:, 0:2]

        return (labels, label_weights, bbox_targets, bbox_weights,
                centers2d_targets, pos_inds, neg_inds)
