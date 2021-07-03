import json
import os
import sys

import cv2
import ray
import glob
import yaml
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from common.augmenter import augment
from utils import filter_sem

# CHANNELS = [
#     4,  # Pedestrians    
#     6,  # Road lines
#     7,  # Road masks
#     8,  # Side walks
#     10, # Vehicles
#     # 12, # Traffic light poles
#     # 18, # Traffic boxes
# ]

MAX_STOP = 500

class VisualizationDataset(Dataset):
    
    def __init__(self, data_dir, config_path):
        super().__init__()

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        self.T = config['num_plan']
        self.seg_channels = config['seg_channels']
        
        self.num_speeds = config['num_speeds']
        self.num_steers = config['num_steers']
        self.num_throts = config['num_throts']

        self.num_frames = 0
        self.txn_map = dict()
        self.idx_map = dict()
        self.file_map = dict()

        # Load dataset
        for full_path in glob.glob(f'{data_dir}/**'):
            txn = lmdb.open(
                full_path,
                max_readers=1, readonly=True,
                lock=False, readahead=False, meminit=False).begin(write=False)
            
            n = int(txn.get('len'.encode()))
            if n < self.T+1:
                print (full_path, 'is too small. consider deleting it.')
                txn.__exit__()
            else:
                offset = self.num_frames
                for i in range(n-self.T):
                    self.num_frames += 1
                    self.txn_map[offset+i] = txn
                    self.idx_map[offset+i] = i
                    self.file_map[offset+i] = full_path

        print(f'{data_dir}: {self.num_frames} frames')
    
    def __len__(self):
        return self.num_frames
        
    def __getitem__(self, idx):

        lmdb_txn = self.txn_map[idx]
        index = self.idx_map[idx]

        locs = self.__class__.access('loc', lmdb_txn, index, self.T+1, dtype=np.float32)
        rots = self.__class__.access('rot', lmdb_txn, index, self.T, dtype=np.float32)
        spds = self.__class__.access('spd', lmdb_txn, index, self.T, dtype=np.float32).flatten()
        lbls = self.__class__.access('lbl', lmdb_txn, index+1, self.T, dtype=np.uint8).reshape(-1,96,96,12)
        maps = self.__class__.access('map', lmdb_txn, index+1, self.T, dtype=np.uint8).reshape(-1,1536,1536,12)

        rgb = self.__class__.access('rgb',  lmdb_txn, index, 1, dtype=np.uint8).reshape(720,1280,3)
        cmd = self.__class__.access('cmd', lmdb_txn, index, 1, dtype=np.float32).flatten()
        act = self.__class__.access('act', lmdb_txn, index, 1, dtype=np.float32).flatten()

        # Crop cameras
        rgb = rgb[:,:,::-1]

        return rgb, maps, lbls, locs, rots, spds, int(cmd), act

    @staticmethod
    def access(tag, lmdb_txn, index, T, dtype=np.float32):
        return np.stack([np.frombuffer(lmdb_txn.get((f'{tag}_{t:05d}').encode()), dtype) for t in range(index,index+T)])


class MainDataset(Dataset):
    def __init__(self, data_dir, config_path):
        super().__init__()

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        self.T = config['num_plan']
        self.camera_yaws = config['camera_yaws']
        self.wide_crop_top = config['wide_crop_top']
        self.narr_crop_bottom = config['narr_crop_bottom']
        self.seg_channels = config['seg_channels']
        
        self.num_speeds = config['num_speeds']
        self.num_steers = config['num_steers']
        self.num_throts = config['num_throts']

        # Ablation options
        self.multi_cam = config['multi_cam']

        self.num_frames = 0
        self.txn_map = dict()
        self.idx_map = dict()
        self.yaw_map = dict()
        self.file_map = dict()

        # Load dataset
        for full_path in glob.glob(f'{data_dir}/**'):
            txn = lmdb.open(
                full_path,
                max_readers=1, readonly=True,
                lock=False, readahead=False, meminit=False).begin(write=False)
            
            n = int(txn.get('len'.encode()))
            if n < self.T+1:
                print (full_path, ' is too small. consider deleting it.')
                txn.__exit__()
            else:
                offset = self.num_frames
                for i in range(n-self.T):
                    self.num_frames += 1
                    for j in range(len(self.camera_yaws)):
                        self.txn_map[(offset+i)*len(self.camera_yaws)+j] = txn
                        self.idx_map[(offset+i)*len(self.camera_yaws)+j] = i
                        self.yaw_map[(offset+i)*len(self.camera_yaws)+j] = j
                        self.file_map[(offset+i)*len(self.camera_yaws)+j] = full_path

        print(f'{data_dir}: {self.num_frames} frames (x{len(self.camera_yaws)})')
        
    def __len__(self):
        print('this is called')
        if self.multi_cam:
            return self.num_frames*len(self.camera_yaws)
        else:
            return self.num_frames
        
    def __getitem__(self, idx):
        
        if not self.multi_cam:
            idx *= len(self.camera_yaws)

        lmdb_txn = self.txn_map[idx]
        index = self.idx_map[idx]
        cam_index = self.yaw_map[idx]
        
        locs = self.__class__.access('loc', lmdb_txn, index, self.T+1, dtype=np.float32)
        rots = self.__class__.access('rot', lmdb_txn, index, self.T, dtype=np.float32)
        spds = self.__class__.access('spd', lmdb_txn, index, self.T, dtype=np.float32).flatten()
        lbls = self.__class__.access('lbl', lmdb_txn, index+1, self.T, dtype=np.uint8).reshape(-1,96,96,12)

        wide_rgb = self.__class__.access('wide_rgb_{}'.format(cam_index),  lmdb_txn, index, 1, dtype=np.uint8).reshape(240,480,3)
        wide_sem = self.__class__.access('wide_sem_{}'.format(cam_index),  lmdb_txn, index, 1, dtype=np.uint8).reshape(240,480)
        narr_rgb = self.__class__.access('narr_rgb_{}'.format(cam_index),  lmdb_txn, index, 1, dtype=np.uint8).reshape(240,384,3)
        cmd = self.__class__.access('cmd', lmdb_txn, index, 1, dtype=np.float32).flatten()

        wide_sem = filter_sem(wide_sem, self.seg_channels)

        # Crop cameras
        wide_rgb = wide_rgb[self.wide_crop_top:,:,::-1]
        wide_sem = wide_sem[self.wide_crop_top:]
        narr_rgb = narr_rgb[:-self.narr_crop_bottom,:,::-1]

        return wide_rgb, wide_sem, narr_rgb, lbls, locs, rots, spds, int(cmd)

    @staticmethod
    def access(tag, lmdb_txn, index, T, dtype=np.float32):
        return np.stack([np.frombuffer(lmdb_txn.get((f'{tag}_{t:05d}').encode()), dtype) for t in range(index,index+T)])


class MainDatasetUnravel():
    def __init__(self, data_dir, config_path):
        super().__init__()

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        for key, value in config.items():
            setattr(self, key, value)

        self.T = config['num_plan']
        self.camera_yaws = config['camera_yaws']
        self.wide_crop_top = config['wide_crop_top']
        self.narr_crop_bottom = config['narr_crop_bottom']
        self.seg_channels = config['seg_channels']

        self.num_speeds = config['num_speeds']
        self.num_steers = config['num_steers']
        self.num_throts = config['num_throts']

        # Ablation options
        self.multi_cam = config['multi_cam']

        self.vizs = []
        self.wide_rgbs = []
        self.narr_rgbs = []
        # self.annotated_narr_rgbs = []
        # self.raw_gaze_maps = []
        self.wide_sems = []
        self.narr_sems = []

        self.lbls = []
        self.locs = []
        self.rots = []
        self.spds = []
        self.cmds = []


        # Load dataset
        for full_path in glob.glob(f'{data_dir}/**')[:5]:
            print(full_path)
            # if self.multi_cam:
            image_path = full_path + '/rgbs'
            data_file = full_path + '/data.json'

            log_fh = open(data_file)
            log_data = json.load(log_fh)

            # for image in glob.glob(f'{image_path}/**'):
            #     print(image)
            # print(len(glob.glob(f'{image_path}/**')))
            length = log_data['len']
            self.num_frames = length
            for i in range(length):
                frame_data = log_data[str(i)]
                loc = frame_data['loc']
                yaw = frame_data['rot']
                spd = frame_data['spd']
                cmd_value = frame_data['cmd']
                wide_rgbs = []
                narr_rgbs = []
                annotated_narr_rgbs = []
                raw_gaze = []
                wide_sems = []
                narr_sems = []
                lbl = []
                lbl_imgs = sorted(glob.glob(f'{image_path}/lbl_*_{i:05d}.png'))
                # print(lbl_imgs)
                for img in lbl_imgs:
                    lbl.append(cv2.imread(img))


                for j in range(len(self.camera_yaws)):
                    print(i, j, f'{image_path}/wide_{j}_{i:05d}.jpg')
                    wide_rgb = cv2.imread(f'{image_path}/wide_{j}_{i:05d}.jpg')
                    narr_rgb = cv2.imread(f'{image_path}/narr_{j}_{i:05d}.jpg')
                    wide_sem = cv2.imread(f'{image_path}/wide_sem_{j}_{i:05d}.png')
                    narr_sem = cv2.imread(f'{image_path}/narr_sem_{j}_{i:05d}.png')

                    wide_rgbs.append(wide_rgb[..., :3])
                    narr_rgbs.append(narr_rgb[..., :3])
                    # annotated, raw = self.get_gaze(narr_rgb[...,:3])
                    # annotated_narr_rgbs.append(annotated)
                    # raw_gaze.append(raw)
                    wide_sems.append(wide_sem)
                    narr_sems.append(narr_sem)



                if self.num_frames % (self.num_repeat + 1) == 0 and self.stop_count < MAX_STOP:
                    # Note: basically, this is like fast-forwarding. should be okay tho as spd is 0.
                    self.wide_rgbs.append(wide_rgbs)
                    self.narr_rgbs.append(narr_rgbs)
                    # self.annotated_narr_rgbs.append(annotated_narr_rgbs)
                    # self.raw_gaze_maps.append(raw_gaze)
                    self.wide_sems.append(wide_sems)
                    self.narr_sems.append(narr_sems)
                    self.lbls.append(lbl)
                    self.locs.append(loc)
                    self.rots.append(yaw)
                    self.spds.append(spd)
                    self.cmds.append(cmd_value)

            # self.flush_data(full_path)

                # self.num_frames += 1

    def flush_data(self, data_path):
        # Save data
        # data_path = os.path.join(self.main_data_dir, _random_string())
        print('Saving to {}'.format(data_path))

        lmdb_env = lmdb.open(data_path, map_size=int(1e10))

        length = len(self.lbls)
        with lmdb_env.begin(write=True) as txn:

            txn.put('len'.encode(), str(length).encode())

            for i in range(length):

                for idx in range(len(self.camera_yaws)):
                    txn.put(
                        f'wide_rgb_{idx}_{i:05d}'.encode(),
                        np.ascontiguousarray(self.wide_rgbs[i][idx]).astype(np.uint8),
                    )

                    txn.put(
                        f'narr_rgb_{idx}_{i:05d}'.encode(),
                        np.ascontiguousarray(self.narr_rgbs[i][idx]).astype(np.uint8),
                    )

                    # txn.put(
                    #     f'annotated_narr_rgb_{idx}_{i:05d}'.encode(),
                    #     np.ascontiguousarray(self.annotated_narr_rgbs[i][idx]).astype(np.uint8),
                    # )

                    txn.put(
                        f'wide_sem_{idx}_{i:05d}'.encode(),
                        np.ascontiguousarray(self.wide_sems[i][idx]).astype(np.uint8),
                    )

                    txn.put(
                        f'narr_sem_{idx}_{i:05d}'.encode(),
                        np.ascontiguousarray(self.narr_sems[i][idx]).astype(np.uint8),
                    )

                txn.put(
                    f'lbl_{i:05d}'.encode(),
                    np.ascontiguousarray(self.lbls[i]).astype(np.uint8),
                )

                txn.put(
                    f'loc_{i:05d}'.encode(),
                    np.ascontiguousarray(self.locs[i]).astype(np.float32)
                )

                txn.put(
                    f'rot_{i:05d}'.encode(),
                    np.ascontiguousarray(self.rots[i]).astype(np.float32)
                )

                txn.put(
                    f'spd_{i:05d}'.encode(),
                    np.ascontiguousarray(self.spds[i]).astype(np.float32)
                )

                txn.put(
                    f'cmd_{i:05d}'.encode(),
                    np.ascontiguousarray(self.cmds[i]).astype(np.float32)
                )

        self.vizs.clear()
        self.wide_rgbs.clear()
        self.narr_rgbs.clear()
        # self.annotated_narr_rgbs.clear()
        # self.raw_gaze_maps.clear()
        self.wide_sems.clear()
        self.narr_sems.clear()
        self.lbls.clear()
        self.locs.clear()
        self.rots.clear()
        self.spds.clear()
        self.cmds.clear()

        lmdb_env.close()

class LabeledMainDataset(MainDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmenter = augment(0.5)
        
    def __getitem__(self, idx):
        lmdb_txn = self.txn_map[idx]
        index = self.idx_map[idx]
        cam_index = self.yaw_map[idx]

        wide_rgb = self.__class__.access('wide_rgb_{}'.format(cam_index),  lmdb_txn, index, 1, dtype=np.uint8).reshape(240,480,3)
        wide_sem = self.__class__.access('wide_sem_{}'.format(cam_index),  lmdb_txn, index, 1, dtype=np.uint8).reshape(240,480)
        narr_rgb = self.__class__.access('narr_rgb_{}'.format(cam_index),  lmdb_txn, index, 1, dtype=np.uint8).reshape(240,384,3)
        narr_sem = self.__class__.access('narr_sem_{}'.format(cam_index),  lmdb_txn, index, 1, dtype=np.uint8).reshape(240,384,4)[...,2]
        cmd = self.__class__.access('cmd', lmdb_txn, index, 1, dtype=np.float32).flatten()
        spd = self.__class__.access('spd', lmdb_txn, index, 1, dtype=np.float32).flatten()

        act_val = self.__class__.access('act_val_{}'.format(cam_index), lmdb_txn, index, 1, dtype=np.float32).reshape(6,self.num_steers*self.num_throts+1,self.num_speeds)

        wide_sem = filter_sem(wide_sem, self.seg_channels)
        narr_sem = filter_sem(narr_sem, self.seg_channels)

        # Crop cameras
        wide_rgb = wide_rgb[self.wide_crop_top:,:,::-1]
        wide_sem = wide_sem[self.wide_crop_top:]
        narr_rgb = narr_rgb[:-self.narr_crop_bottom,:,::-1]
        narr_sem = narr_sem[:-self.narr_crop_bottom]



        # Augment
        wide_rgb = self.augmenter(images=wide_rgb[None])[0]
        narr_rgb = self.augmenter(images=narr_rgb[None])[0]

        # print(wide_rgb.shape)
        # cv2.imshow('wide', wide_rgb)
        # cv2.imshow('narr', narr_rgb)
        # cv2.waitKey(-1)
        # sys.exit(0)

        return wide_rgb, wide_sem, narr_rgb, narr_sem, act_val, float(spd), int(cmd)


#@ray.remote
class RemoteMainDataset(MainDatasetUnravel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.act_vals = dict()

    def num_frames(self):
        return self.num_frames

    def save(self, base_idx, action_values):
        """
        Cache the action values
        """
        for i in range(len(self.camera_yaws)):
            idx = base_idx + i
            self.act_vals[idx] = action_values[i]

    def commit(self):
        """
        Commit the saved cache
        """

        for idx, act_val in self.act_vals.items():

            file_name = self.file_map[idx]
            file_idx  = self.idx_map[idx]
            yaw_idx   = self.yaw_map[idx]

            lmdb_env = lmdb.open(file_name, map_size=int(1e10))
            with lmdb_env.begin(write=True) as txn:
                txn.put(
                    f'act_val_{yaw_idx}_{file_idx:05d}'.encode(),
                    np.ascontiguousarray(act_val).astype(np.float32),
                )

            lmdb_env.close()
            print ('wrote to {}'.format(file_name))


if __name__ == '__main__':

    root = '/mnt/e3641809-fbba-49ea-86ed-f89021679b99/home/varun/Downloads/rails_dataset/main_trajs6_converted2'
    config = '/mnt/e3641809-fbba-49ea-86ed-f89021679b99/home/varun/carla/PythonAPI/WorldOnRails/config.yaml'

    # dataset = MainDataset(root, config)
    d1 = MainDatasetUnravel(root, config)
    dataset = MainDataset(root, config)

    # wide_rgb, wide_sem, narr_rgb, lbls, locs, rots, spds, cmd = dataset[30]
    
    # print (cmd)
