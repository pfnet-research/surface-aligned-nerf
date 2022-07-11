import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("../../../")
from zju_smpl.smplmodel.body_model import SMPLlayer
from lib.config import cfg

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LearnableSMPL(nn.Module):

    def __init__(self, requires_grad=True):
        super(LearnableSMPL, self).__init__()

        self.train_frame_start = cfg.begin_ith_frame
        self.train_frame_end = cfg.begin_ith_frame + cfg.num_train_frame

        self.human = cfg.human
        self.init_learnable_smpl_params(requires_grad=requires_grad)

    def init_learnable_smpl_params(self, requires_grad=True):

        poses_train, Rh_train, Th_train, shapes_train = [], [], [], []
        # fill data
        for i in range(self.train_frame_start, self.train_frame_end):

            if self.human in [313, 315]:
                params_path = os.path.join(cfg.train_dataset.data_root, cfg.params, '{}.npy'.format(i+1))
            else:
                params_path = os.path.join(cfg.train_dataset.data_root, cfg.params, '{}.npy'.format(i * cfg.frame_interval))

            params = np.load(params_path, allow_pickle=True).item()
            Rh = params['Rh'].astype(np.float32)
            # R = cv2.Rodrigues(Rh)[0].astype(np.float32)
            Th = params['Th'].astype(np.float32)
            poses = params['poses'].astype(np.float32)
            shapes = params['shapes'].astype(np.float32)

            poses_train.append(torch.tensor(poses))
            Rh_train.append(torch.tensor(Rh))
            Th_train.append(torch.tensor(Th))
            shapes_train.append(torch.tensor(shapes))

        if requires_grad:
            self.poses_train = nn.Parameter(torch.cat(poses_train, dim=0)) # [n, 72]
            self.Rh_train = nn.Parameter(torch.cat(Rh_train, dim=0)) # [n, 3]
            self.Th_train = nn.Parameter(torch.cat(Th_train, dim=0)) # [n, 3]
        else:
            self.poses_train = torch.cat(poses_train, dim=0).to(device) # [n, 72]
            self.Rh_train = torch.cat(Rh_train, dim=0).to(device) # [n, 3]
            self.Th_train = torch.cat(Th_train, dim=0).to(device) # [n, 3]

        self.shapes_train = torch.cat(shapes_train, dim=0).to(device)

        # load smpl model
        model_folder = 'zju_smpl/smplx'
        body_model = SMPLlayer(os.path.join(model_folder, 'smpl'),
                                gender='neutral',
                                device=device,
                                regressor_path=os.path.join(model_folder,
                                                            'J_regressor_body25.npy'))

        self.body_model = body_model.to(device)

        if not cfg.optimize_smpl:
            self.freeze_learnable_smpl_params()

    def freeze_learnable_smpl_params(self):
        self.poses_train.requires_grad = False
        self.Rh_train.requires_grad = False
        self.Th_train.requires_grad = False
        
    def get_learnable_smpl_params(self, frame_index):
        params = {
            'poses': self.poses_train[frame_index],
            'Rh': self.Rh_train[frame_index],
            'Th': self.Th_train[frame_index],
            'shapes': self.shapes_train[frame_index]
        }
        return params

    def get_learnable_verts(self, frame_index, shape_control=[[]]):
        # shape_control: e.g. [[1, -4], [2, 2]]
        params = self.get_learnable_smpl_params(frame_index)

        # shape control
        if shape_control[0]:
            for pc, val in shape_control:
                params['shapes'][..., pc-1] = val # pc starts from 1
        
        verts = self.body_model(return_verts=True,
                      return_tensor=True,
                      new_params=cfg.new_params,
                      **params)
        return verts.unsqueeze(0)