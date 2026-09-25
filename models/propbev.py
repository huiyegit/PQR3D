import queue
import torch
import numpy as np
from mmcv.runner import force_fp32, auto_fp16
from mmcv.runner import get_dist_info
from mmcv.runner.fp16_utils import cast_tensor_type
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from .utils import GridMask, pad_multiple, GpuPhotoMetricDistortion

NUM_CAMS = 6  # nuScenes: 6 cameras per frame


@DETECTORS.register_module()
class PropBEV(MVXTwoStageDetector):
    def __init__(self,
                 data_aug=None,
                 stop_prev_grad=0,
                 num_propgated=256,  # Number of queries propagated sequentially: t-2 → t-1 → t
                 t1_slot=1,   # Slot index of t-1 in the frame sequence (8f past-only: 1, 15f interleave: 1)
                 t2_slot=2,   # Slot index of t-2 in the frame sequence (8f past-only: 2, 15f interleave: 3)
                 use_t2=True, # Whether to use t-2 keyframe (False saves GPU memory for large backbones)
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(PropBEV, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.data_aug = data_aug
        self.stop_prev_grad = stop_prev_grad
        self.num_propgated = num_propgated  # NEW
        self.t1_slot = t1_slot  # slot index of t-1 frame
        self.t2_slot = t2_slot  # slot index of t-2 frame
        self.use_t2 = use_t2    # whether to use t-2 keyframe
        self.color_aug = GpuPhotoMetricDistortion()
        self.grid_mask = GridMask(ratio=0.5, prob=0.7)
        self.use_grid_mask = True

        self.memory = {}
        self.queue = queue.Queue()

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_img_feat(self, img):
        if self.use_grid_mask:
            img = self.grid_mask(img)

        img_feats = self.img_backbone(img)

        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        return img_feats

    def extract_feat(self, img, img_metas):
        if isinstance(img, list):
            img = torch.stack(img, dim=0)

        assert img.dim() == 5

        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)
        img = img.float()

        # move some augmentations to GPU
        if self.data_aug is not None:
            if 'img_color_aug' in self.data_aug and self.data_aug['img_color_aug'] and self.training:
                img = self.color_aug(img)

            if 'img_norm_cfg' in self.data_aug:
                img_norm_cfg = self.data_aug['img_norm_cfg']

                norm_mean = torch.tensor(img_norm_cfg['mean'], device=img.device)
                norm_std = torch.tensor(img_norm_cfg['std'], device=img.device)

                if img_norm_cfg['to_rgb']:
                    img = img[:, [2, 1, 0], :, :]  # BGR to RGB

                img = img - norm_mean.reshape(1, 3, 1, 1)
                img = img / norm_std.reshape(1, 3, 1, 1)

            for b in range(B):
                img_shape = (img.shape[2], img.shape[3], img.shape[1])
                img_metas[b]['img_shape'] = [img_shape for _ in range(N)]
                img_metas[b]['ori_shape'] = [img_shape for _ in range(N)]

            if 'img_pad_cfg' in self.data_aug:
                img_pad_cfg = self.data_aug['img_pad_cfg']
                img = pad_multiple(img, img_metas, size_divisor=img_pad_cfg['size_divisor'])

        input_shape = img.shape[-2:]
        # update real input shape of each single img
        for img_meta in img_metas:
            img_meta.update(input_shape=input_shape)

        if self.training and self.stop_prev_grad > 0:
            H, W = input_shape
            img = img.reshape(B, -1, NUM_CAMS, C, H, W)

            img_grad = img[:, :self.stop_prev_grad]
            img_nograd = img[:, self.stop_prev_grad:]

            all_img_feats = [self.extract_img_feat(img_grad.reshape(-1, C, H, W))]

            with torch.no_grad():
                self.eval()
                for k in range(img_nograd.shape[1]):
                    all_img_feats.append(self.extract_img_feat(img_nograd[:, k].reshape(-1, C, H, W)))
                self.train()

            img_feats = []
            for lvl in range(len(all_img_feats[0])):
                C, H, W = all_img_feats[0][lvl].shape[1:]
                img_feat = torch.cat([feat[lvl].reshape(B, -1, NUM_CAMS, C, H, W) for feat in all_img_feats], dim=1)
                img_feat = img_feat.reshape(-1, C, H, W)
                img_feats.append(img_feat)
        else:
            img_feats = self.extract_img_feat(img)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        return img_feats_reshaped

    # =========================================================================
    # NEW: Dual keyframe helper methods
    # =========================================================================

    def _reorder_hist_feats(self, feats_past, feats_future):
        """Reorder features: past frames first, future frames at back.
        Args:
            feats_past: list of [B, N_past, C, H, W]
            feats_future: list of [B, N_future, C, H, W]
        Returns: list of [B, N_total, C, H, W]
        """
        return [torch.cat([p, f], dim=1) for p, f in zip(feats_past, feats_future)]

    def _transform_bbox_to_current_frame(self, bbox_denorm, lidar_to_target, pc_range, img_metas, from_frame='t1', to_frame='t'):
        """Transform bbox from lidar_{from_frame} to lidar_{to_frame} frame with velocity compensation.
        Args:
            from_frame: 't1' or 't2'
            to_frame: 't' or 't1'
        """
        B, N, _ = bbox_denorm.shape
        device, dtype = bbox_denorm.device, bbox_denorm.dtype
        pc_range = torch.tensor(pc_range, device=device, dtype=dtype)

        # Get time interval: to_frame timestamp - from_frame timestamp
        frame_to_idx = {'t': 0, 't1': self.t1_slot * NUM_CAMS, 't2': self.t2_slot * NUM_CAMS}
        dt = img_metas[0]['img_timestamp'][frame_to_idx[to_frame]] - img_metas[0]['img_timestamp'][frame_to_idx[from_frame]]
        dt = torch.tensor(dt, device=device, dtype=dtype)

        # Transform position (cx, cy, cz)
        pos = torch.stack([bbox_denorm[..., 0], bbox_denorm[..., 1], bbox_denorm[..., 4]], dim=-1)
        pos_homo = torch.cat([pos, torch.ones(B, N, 1, device=device, dtype=dtype)], dim=-1)
        pos_t = torch.bmm(pos_homo, lidar_to_target.transpose(-1, -2).to(dtype))[..., :3]

        # Transform rotation
        rot_2x2 = lidar_to_target[:, :2, :2].to(dtype)
        delta_cos, delta_sin = rot_2x2[:, 0, 0].view(B, 1), rot_2x2[:, 1, 0].view(B, 1)
        sin_t = bbox_denorm[..., 6] * delta_cos + bbox_denorm[..., 7] * delta_sin
        cos_t = bbox_denorm[..., 7] * delta_cos - bbox_denorm[..., 6] * delta_sin

        # Transform velocity to current frame
        vel_t = torch.bmm(bbox_denorm[..., 8:10], rot_2x2.transpose(-1, -2))

        # Velocity compensation for position
        pos_t[..., :2] = pos_t[..., :2] + vel_t * dt

        # Normalize position
        pos_norm = torch.stack([
            (pos_t[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0]),
            (pos_t[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1]),
            (pos_t[..., 2] - pc_range[2]) / (pc_range[5] - pc_range[2]),
        ], dim=-1).clamp(0, 1)

        return torch.cat([pos_norm, bbox_denorm[..., 2:4], bbox_denorm[..., 5:6],
                          sin_t.unsqueeze(-1), cos_t.unsqueeze(-1), vel_t], dim=-1)

    def _select_topk_bbox(self, bbox, cls_scores, k):
        """Select top-K bbox based on classification scores."""
        scores = cls_scores.sigmoid().max(dim=-1).values
        k = min(k, scores.shape[1])
        _, idx = torch.topk(scores, k, dim=1)
        return torch.gather(bbox, 1, idx.unsqueeze(-1).expand(-1, -1, bbox.shape[-1]))

    def _select_topk_bbox_and_feat(self, bbox, cls_scores, query_feat, k):
        """Select top-K bbox and corresponding query features based on classification scores."""
        scores = cls_scores.sigmoid().max(dim=-1).values
        k = min(k, scores.shape[1])
        _, idx = torch.topk(scores, k, dim=1)
        bbox_topk = torch.gather(bbox, 1, idx.unsqueeze(-1).expand(-1, -1, bbox.shape[-1]))
        feat_topk = torch.gather(query_feat, 1, idx.unsqueeze(-1).expand(-1, -1, query_feat.shape[-1]))
        return bbox_topk, feat_topk

    def _prepare_metas(self, img_metas, keyframe='t'):
        """Prepare img_metas for keyframe t, t-1, or t-2.
        All keyframes use the same set of frames, rotated so that the
        target keyframe sits at the front (slot 0).
            t:  original order
            t1: rotate by 1 frame  (NUM_CAMS images)
            t2: rotate by 2 frames (2*NUM_CAMS images)
        """
        total_imgs = len(img_metas[0]['filename'])  # num_frames * NUM_CAMS
        rot = {'t': 0, 't1': self.t1_slot * NUM_CAMS, 't2': self.t2_slot * NUM_CAMS}[keyframe]

        metas = []
        for m in img_metas:
            new_m = {k: v for k, v in m.items()}
            lidar_key = {'t': 'lidar2img', 't1': 'lidar2img_t1', 't2': 'lidar2img_t2'}[keyframe]
            if rot > 0:
                new_m['lidar2img'] = m[lidar_key][rot:total_imgs] + m[lidar_key][0:rot]
                new_m['img_timestamp'] = m['img_timestamp'][rot:total_imgs] + m['img_timestamp'][0:rot]
                new_m['filename'] = m['filename'][rot:total_imgs] + m['filename'][0:rot]
            else:
                new_m['lidar2img'] = m[lidar_key][:total_imgs]
                new_m['img_timestamp'] = m['img_timestamp'][:total_imgs]
                new_m['filename'] = m['filename'][:total_imgs]
            metas.append(new_m)
        return metas

    def _get_lidar2lidar_transform(self, img_metas, from_frame='t1', to_frame='t'):
        """Compute transformation from lidar_{from_frame} to lidar_{to_frame}. Returns [B, 4, 4].
        Args:
            from_frame: 't', 't1', or 't2'
            to_frame: 't', 't1', or 't2'
        """
        def get_ego_pose(meta, frame):
            if frame == 't':
                return meta['ego_pose']
            elif frame == 't1':
                return meta['ego_pose_t1']
            else:  # 't2'
                return meta['ego_pose_t2']
        
        transforms = []
        for meta in img_metas:
            ego_pose_from = get_ego_pose(meta, from_frame)
            ego_pose_to = get_ego_pose(meta, to_frame)
            # lidar_{from_frame} → lidar_{to_frame}
            transforms.append(np.linalg.inv(ego_pose_to) @ ego_pose_from)

        device = img_metas[0]['lidar2img'][0].device if torch.is_tensor(img_metas[0]['lidar2img'][0]) else 'cuda'
        return torch.tensor(np.stack(transforms), dtype=torch.float32, device=device)

    def _process_keyframe(self, img_feats, img_metas, keyframe='t1', gt_bboxes_3d=None, gt_labels_3d=None, prev_bbox=None, prev_feat=None, to_frame='t'):
        """Process a single keyframe and return transformed bbox predictions and query features.
        Args:
            keyframe: 't1' or 't2'
            prev_bbox: propagated queries from previous keyframe (e.g., t-2 for t-1)
            prev_feat: propagated query features from previous keyframe
            to_frame: target frame for coordinate transformation ('t' or 't1')
        """
        metas = self._prepare_metas(img_metas, keyframe=keyframe)

        if gt_bboxes_3d is not None:
            for i, m in enumerate(metas):
                m['gt_bboxes_3d'] = gt_bboxes_3d[i]
                m['gt_labels_3d'] = gt_labels_3d[i]

        with torch.no_grad():
            outs = self.pts_bbox_head(img_feats, metas, prev_bbox=prev_bbox, prev_feat=prev_feat, use_dn=False)  # CHANGED: pass prev_feat
            bbox = outs['all_bbox_preds'][-1]
            cls = outs['all_cls_scores'][-1]
            query_feat = outs['all_query_feat']  # NEW: get query features

        # Transform to target frame
        lidar_to_target = self._get_lidar2lidar_transform(img_metas, from_frame=keyframe, to_frame=to_frame)
        bbox_norm = self._transform_bbox_to_current_frame(bbox, lidar_to_target, self.pts_bbox_head.pc_range, img_metas, from_frame=keyframe, to_frame=to_frame)

        # Select top-k and return both bbox and feat
        bbox_topk, feat_topk = self._select_topk_bbox_and_feat(bbox_norm, cls, query_feat, self.num_propgated)
        return bbox_topk, feat_topk

    def _process_dual_keyframes(self, img_feats_t1, img_feats_t2, img_metas, gt_bboxes_3d=None, gt_labels_3d=None):
        """Process keyframes sequentially: (t-2 →) t-1 → t.
        When use_t2=False, skips t-2 and runs t-1 without propagated queries.
        Returns: (bbox, feat) predictions from t-1 only (to be used by keyframe t).
        """
        if self.use_t2:
            # Step 1: Process t-2 without prev_bbox/prev_feat, transform to t-1 frame
            bbox_t2, feat_t2 = self._process_keyframe(img_feats_t2, img_metas, 't2', gt_bboxes_3d, gt_labels_3d, prev_bbox=None, prev_feat=None, to_frame='t1')
        else:
            bbox_t2, feat_t2 = None, None
        
        # Step 2: Process t-1 with (optional) t-2's predictions, transform to t frame
        bbox_t1, feat_t1 = self._process_keyframe(img_feats_t1, img_metas, 't1', gt_bboxes_3d, gt_labels_3d, prev_bbox=bbox_t2, prev_feat=feat_t2, to_frame='t')
        
        # Step 3: Return t-1's predictions in t frame (t will use these as prev_bbox/prev_feat)
        return bbox_t1, feat_t1
    

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bbox=None,
                          prev_feat=None):  # NEW: added prev_feat
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats, img_metas, prev_bbox=prev_bbox, prev_feat=prev_feat)  # NEW: pass prev_feat
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)

        return losses

    @force_fp32(apply_to=('img', 'points'))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        B, N, C, H, W = img.shape

        # Dual keyframe path: rotate features so target keyframe is at front
        if self.num_propgated > 0:
            img_feats_all = self.extract_feat(img, img_metas)
            img_feats_t = img_feats_all  # t: original order

            # t-1: rotate to put t-1 at front
            r1 = self.t1_slot * NUM_CAMS
            img_feats_t1 = self._reorder_hist_feats(
                [feat[:, r1:N] for feat in img_feats_all],
                [feat[:, 0:r1] for feat in img_feats_all])

            # t-2: rotate to put t-2 at front (skip if use_t2=False to save memory)
            if self.use_t2:
                r2 = self.t2_slot * NUM_CAMS
                img_feats_t2 = self._reorder_hist_feats(
                    [feat[:, r2:N] for feat in img_feats_all],
                    [feat[:, 0:r2] for feat in img_feats_all])
            else:
                img_feats_t2 = None
            
            # Get prev_bbox from t-1 (and optionally t-2)
            # Temporal dropout: use prev_bbox 90% of the time
            use_temporal = torch.rand(1).item() > 0.1
            if use_temporal:
                prev_bbox, prev_feat = self._process_dual_keyframes(img_feats_t1, img_feats_t2, img_metas, gt_bboxes_3d, gt_labels_3d)
            else:
                prev_bbox = None
                prev_feat = None
            
            metas_t = self._prepare_metas(img_metas, keyframe='t')
            for i in range(len(metas_t)):
                metas_t[i]['gt_bboxes_3d'] = gt_bboxes_3d[i]
                metas_t[i]['gt_labels_3d'] = gt_labels_3d[i]
            losses = self.forward_pts_train(img_feats_t, gt_bboxes_3d, gt_labels_3d, metas_t,
                                            gt_bboxes_ignore, prev_bbox=prev_bbox, prev_feat=prev_feat)
        else:
            # Original single keyframe path
            img_feats = self.extract_feat(img, img_metas)

            for i in range(len(img_metas)):
                img_metas[i]['gt_bboxes_3d'] = gt_bboxes_3d[i]
                img_metas[i]['gt_labels_3d'] = gt_labels_3d[i]

            losses = self.forward_pts_train(img_feats, gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore)

        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img
        return self.simple_test(img_metas[0], img[0], **kwargs)

    def simple_test_pts(self, x, img_metas, rescale=False, prev_bbox=None, prev_feat=None):  # NEW: added prev_feat
        outs = self.pts_bbox_head(x, img_metas, prev_bbox=prev_bbox, prev_feat=prev_feat)  # NEW: pass prev_feat
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas[0], rescale=rescale)

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        return bbox_results
    
    def simple_test(self, img_metas, img=None, rescale=False):
        world_size = get_dist_info()[1]
        if world_size == 1:  # online
            return self.simple_test_online(img_metas, img, rescale)
        else:  # offline
            return self.simple_test_offline(img_metas, img, rescale)

    def simple_test_offline(self, img_metas, img=None, rescale=False):
        B, N, C, H, W = img.shape

        # Dual keyframe path: rotate features so target keyframe is at front
        if self.num_propgated > 0:
            img_feats_all = self.extract_feat(img, img_metas)
            img_feats_t = img_feats_all  # t: original order

            # t-1: rotate to put t-1 at front
            r1 = self.t1_slot * NUM_CAMS
            img_feats_t1 = self._reorder_hist_feats(
                [feat[:, r1:N] for feat in img_feats_all],
                [feat[:, 0:r1] for feat in img_feats_all])

            # t-2: rotate to put t-2 at front (skip if use_t2=False to save memory)
            if self.use_t2:
                r2 = self.t2_slot * NUM_CAMS
                img_feats_t2 = self._reorder_hist_feats(
                    [feat[:, r2:N] for feat in img_feats_all],
                    [feat[:, 0:r2] for feat in img_feats_all])
            else:
                img_feats_t2 = None

            prev_bbox, prev_feat = self._process_dual_keyframes(img_feats_t1, img_feats_t2, img_metas)

            metas_t = self._prepare_metas(img_metas, keyframe='t')
            bbox_pts = self.simple_test_pts(img_feats_t, metas_t, rescale=rescale, prev_bbox=prev_bbox, prev_feat=prev_feat)
            
        else:
            # Original path
            img_feats = self.extract_feat(img=img, img_metas=img_metas)
            bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)

        bbox_list = [dict() for _ in range(len(img_metas))]
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox

        return bbox_list

    def simple_test_online(self, img_metas, img=None, rescale=False):
        self.fp16_enabled = False
        assert len(img_metas) == 1  # batch_size = 1

        B, N, C, H, W = img.shape

        img = img.reshape(B, N//NUM_CAMS, NUM_CAMS, C, H, W)

        img_filenames = img_metas[0]['filename']
        num_frames = len(img_filenames) // NUM_CAMS
        # assert num_frames == img.shape[1]

        img_shape = (H, W, C)
        img_metas[0]['img_shape'] = [img_shape for _ in range(len(img_filenames))]
        img_metas[0]['ori_shape'] = [img_shape for _ in range(len(img_filenames))]
        img_metas[0]['pad_shape'] = [img_shape for _ in range(len(img_filenames))]

        img_feats_list, img_metas_list = [], []

        # extract feature frame by frame
        for i in range(num_frames):
            img_indices = list(np.arange(i * NUM_CAMS, (i + 1) * NUM_CAMS))

            img_metas_curr = [{}]
            for k in img_metas[0].keys():
                if isinstance(img_metas[0][k], list):
                    img_metas_curr[0][k] = [img_metas[0][k][i] for i in img_indices]

            if img_filenames[img_indices[0]] in self.memory:
                # found in memory
                img_feats_curr = self.memory[img_filenames[img_indices[0]]]
            else:
                # extract feature and put into memory
                img_feats_curr = self.extract_feat(img[:, i], img_metas_curr)
                self.memory[img_filenames[img_indices[0]]] = img_feats_curr
                self.queue.put(img_filenames[img_indices[0]])
                while self.queue.qsize() >= 16:  # avoid OOM
                    pop_key = self.queue.get()
                    self.memory.pop(pop_key)

            img_feats_list.append(img_feats_curr)
            img_metas_list.append(img_metas_curr)

        # reorganize
        feat_levels = len(img_feats_list[0])
        img_feats_reorganized = []
        for j in range(feat_levels):
            feat_l = torch.cat([img_feats_list[i][j] for i in range(len(img_feats_list))], dim=0)
            feat_l = feat_l.flatten(0, 1)[None, ...]
            img_feats_reorganized.append(feat_l)

        img_metas_reorganized = img_metas_list[0]
        for i in range(1, len(img_metas_list)):
            for k, v in img_metas_list[i][0].items():
                if isinstance(v, list):
                    img_metas_reorganized[0][k].extend(v)

        img_feats = img_feats_reorganized
        img_metas = img_metas_reorganized
        img_feats = cast_tensor_type(img_feats, torch.half, torch.float32)

        # run detector
        bbox_list = [dict() for _ in range(1)]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox

        return bbox_list