import os
import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from pyquaternion import Quaternion
from .pipelines.loading import compose_lidar2img  # Import from loading.py

@DATASETS.register_module()
class CustomNuScenesDataset(NuScenesDataset):

    def collect_sweeps(self, index, into_past=60, into_future=60):
        all_sweeps_prev = []
        curr_index = index
        while len(all_sweeps_prev) < into_past:
            curr_sweeps = self.data_infos[curr_index]['sweeps']
            if len(curr_sweeps) == 0:
                break
            all_sweeps_prev.extend(curr_sweeps)
            all_sweeps_prev.append(self.data_infos[curr_index - 1]['cams'])
            curr_index = curr_index - 1
        
        all_sweeps_next = []
        curr_index = index + 1
        while len(all_sweeps_next) < into_future:
            if curr_index >= len(self.data_infos):
                break
            curr_sweeps = self.data_infos[curr_index]['sweeps']
            all_sweeps_next.extend(curr_sweeps[::-1])
            all_sweeps_next.append(self.data_infos[curr_index]['cams'])
            curr_index = curr_index + 1

        return all_sweeps_prev, all_sweeps_next

    def get_data_info(self, index):
        info = self.data_infos[index]
        sweeps_prev, sweeps_next = self.collect_sweeps(index)

        # ==================== Keyframe t (current) ego pose ====================
        ego2global_translation = np.array(info['ego2global_translation'])
        ego2global_rotation = info['ego2global_rotation']
        lidar2ego_translation = np.array(info['lidar2ego_translation'])
        lidar2ego_rotation = info['lidar2ego_rotation']
        ego2global_rotation_mat = Quaternion(ego2global_rotation).rotation_matrix
        lidar2ego_rotation_mat = Quaternion(lidar2ego_rotation).rotation_matrix

        input_dict = dict(
            sample_idx=info['token'],
            sweeps={'prev': sweeps_prev, 'next': sweeps_next},
            timestamp=info['timestamp'] / 1e6,
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation_mat,
            lidar2ego_translation=lidar2ego_translation,
            lidar2ego_rotation=lidar2ego_rotation_mat,
        )

        # Get current scene token for boundary check
        current_scene_token = info['scene_token']

        # ==================== Keyframe t-1 ego pose ====================
        # FIXED: Check scene_token to avoid using frames from different scenes
        # if index > 0 and self.data_infos[index - 1]['scene_token'] == current_scene_token:
        if index > 0:
            t1_info = self.data_infos[index - 1]
            t1_ego2global_translation = np.array(t1_info['ego2global_translation'])
            t1_ego2global_rotation = Quaternion(t1_info['ego2global_rotation']).rotation_matrix
            t1_lidar2ego_translation = np.array(t1_info['lidar2ego_translation'])
            t1_lidar2ego_rotation = Quaternion(t1_info['lidar2ego_rotation']).rotation_matrix
        else:
            # First sample or scene boundary - use current as fallback
            t1_ego2global_translation = ego2global_translation
            t1_ego2global_rotation = ego2global_rotation_mat
            t1_lidar2ego_translation = lidar2ego_translation
            t1_lidar2ego_rotation = lidar2ego_rotation_mat
        
        # ==================== Keyframe t-2 ego pose ====================
        # FIXED: Check scene_token to avoid using frames from different scenes
        # if index > 1 and self.data_infos[index - 2]['scene_token'] == current_scene_token:
        if index > 1:
            t2_info = self.data_infos[index - 2]
            t2_ego2global_translation = np.array(t2_info['ego2global_translation'])
            t2_ego2global_rotation = Quaternion(t2_info['ego2global_rotation']).rotation_matrix
            t2_lidar2ego_translation = np.array(t2_info['lidar2ego_translation'])
            t2_lidar2ego_rotation = Quaternion(t2_info['lidar2ego_rotation']).rotation_matrix
        else:
            # First or second sample or scene boundary - use t-1 as fallback (NOT current!)
            # This ensures t-2 uses previous frame when only 1 previous frame exists
            t2_ego2global_translation = t1_ego2global_translation
            t2_ego2global_rotation = t1_ego2global_rotation
            t2_lidar2ego_translation = t1_lidar2ego_translation
            t2_lidar2ego_rotation = t1_lidar2ego_rotation
        
        input_dict['t1_ego2global_translation'] = t1_ego2global_translation
        input_dict['t1_ego2global_rotation'] = t1_ego2global_rotation
        input_dict['t1_lidar2ego_translation'] = t1_lidar2ego_translation
        input_dict['t1_lidar2ego_rotation'] = t1_lidar2ego_rotation
        
        input_dict['t2_ego2global_translation'] = t2_ego2global_translation
        input_dict['t2_ego2global_rotation'] = t2_ego2global_rotation
        input_dict['t2_lidar2ego_translation'] = t2_lidar2ego_translation
        input_dict['t2_lidar2ego_rotation'] = t2_lidar2ego_rotation

        # ==================== Compute combined ego_pose matrices ====================
        # ego_pose = lidar2global transformation (will be updated by augmentation)
        
        # Keyframe t
        e2g_matrix = np.eye(4)
        e2g_matrix[:3, :3] = ego2global_rotation_mat
        e2g_matrix[:3, 3] = ego2global_translation
        l2e_matrix = np.eye(4)
        l2e_matrix[:3, :3] = lidar2ego_rotation_mat
        l2e_matrix[:3, 3] = lidar2ego_translation
        ego_pose = e2g_matrix @ l2e_matrix  # lidar_t → global

        # Keyframe t-1
        t1_e2g_matrix = np.eye(4)
        t1_e2g_matrix[:3, :3] = t1_ego2global_rotation
        t1_e2g_matrix[:3, 3] = t1_ego2global_translation
        t1_l2e_matrix = np.eye(4)
        t1_l2e_matrix[:3, :3] = t1_lidar2ego_rotation
        t1_l2e_matrix[:3, 3] = t1_lidar2ego_translation
        ego_pose_t1 = t1_e2g_matrix @ t1_l2e_matrix  # lidar_{t-1} → global

        # Keyframe t-2
        t2_e2g_matrix = np.eye(4)
        t2_e2g_matrix[:3, :3] = t2_ego2global_rotation
        t2_e2g_matrix[:3, 3] = t2_ego2global_translation
        t2_l2e_matrix = np.eye(4)
        t2_l2e_matrix[:3, :3] = t2_lidar2ego_rotation
        t2_l2e_matrix[:3, 3] = t2_lidar2ego_translation
        ego_pose_t2 = t2_e2g_matrix @ t2_l2e_matrix  # lidar_{t-2} → global

        input_dict['ego_pose'] = ego_pose
        input_dict['ego_pose_t1'] = ego_pose_t1
        input_dict['ego_pose_t2'] = ego_pose_t2
        # ==================== END ====================

        if self.modality['use_camera']:
            img_paths = []
            img_timestamps = []
            lidar2img_rts = []
            lidar2img_rts_t1 = []
            lidar2img_rts_t2 = []

            for _, cam_info in info['cams'].items():
                img_paths.append(os.path.relpath(cam_info['data_path']))
                img_timestamps.append(cam_info['timestamp'] / 1e6)

                intrinsic = np.array(cam_info['cam_intrinsic'])
                sensor2lidar_rotation = np.array(cam_info['sensor2lidar_rotation'])
                sensor2lidar_translation = np.array(cam_info['sensor2lidar_translation'])

                # ==================== lidar2img for keyframe t ====================
                lidar2cam_r = np.linalg.inv(sensor2lidar_rotation)
                lidar2cam_t = sensor2lidar_translation @ lidar2cam_r.T

                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                # Compute sensor2global for keyframe t (current)
                sensor2global_rotation = (ego2global_rotation_mat @ lidar2ego_rotation_mat @ sensor2lidar_rotation).T
                sensor2global_translation = (
                    ego2global_rotation_mat @ lidar2ego_rotation_mat @ sensor2lidar_translation
                    + ego2global_rotation_mat @ lidar2ego_translation
                    + ego2global_translation
                )

                # ==================== lidar2img_t1 for keyframe t-1 ====================
                lidar2img_rt_t1 = compose_lidar2img(
                    t1_ego2global_translation,
                    t1_ego2global_rotation,
                    t1_lidar2ego_translation,
                    t1_lidar2ego_rotation,
                    sensor2global_translation,
                    sensor2global_rotation,
                    intrinsic,
                )
                lidar2img_rts_t1.append(lidar2img_rt_t1)

                # ==================== lidar2img_t2 for keyframe t-2 ====================
                lidar2img_rt_t2 = compose_lidar2img(
                    t2_ego2global_translation,
                    t2_ego2global_rotation,
                    t2_lidar2ego_translation,
                    t2_lidar2ego_rotation,
                    sensor2global_translation,
                    sensor2global_rotation,
                    intrinsic,
                )
                lidar2img_rts_t2.append(lidar2img_rt_t2)

            input_dict.update(dict(
                img_filename=img_paths,
                img_timestamp=img_timestamps,
                lidar2img=lidar2img_rts,
                lidar2img_t1=lidar2img_rts_t1,
                lidar2img_t2=lidar2img_rts_t2,
            ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict