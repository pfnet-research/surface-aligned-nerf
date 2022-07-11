import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg
from . import embedder

from .projection.map import SurfaceAlignedConverter
from .gconv import GCN
from .smpl_optimize import LearnableSMPL


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.train_frame_start = cfg.begin_ith_frame
        self.train_frame_end = cfg.begin_ith_frame + cfg.num_train_frame

        self.gcn = GCN()
        self.nerf = NeRF()
        self.converter = SurfaceAlignedConverter()
        self.leanable_smpl = LearnableSMPL(requires_grad=cfg.optimize_smpl)

    def pts_to_can_pts(self, pts, sp_input):
        """transform pts from the world coordinate to the smpl coordinate"""
        Th = sp_input['Th']
        pts = pts - Th
        R = sp_input['R']
        pts = torch.matmul(pts, R)
        return pts

    def get_mask(self, inp, thres=0.15):
        h = inp[..., -1]
        mask = (h < thres) & (h > -0.5) # (batch, 65536)
        return mask


    def prepare_input(self, wpts, sp_input):

        frame_index = sp_input['frame_index']
        if self.training:
            if cfg.optimize_smpl and (frame_index >= self.train_frame_start) and (frame_index < self.train_frame_end):
                frame = (frame_index - self.train_frame_start) // cfg.frame_interval
                verts = self.leanable_smpl.get_learnable_verts(frame)
            else:
                verts = sp_input['verts_world']
        else:
            if cfg.shape_control[0]:
                frame = (frame_index - self.train_frame_start) // cfg.frame_interval
                verts = self.leanable_smpl.get_learnable_verts(frame, cfg.shape_control)
            else:
                verts = sp_input['verts_world']

        # transform sample points to surface-aligned representation (inp)
        inp, local_coordinates = self.converter.xyz_to_xyzch(wpts, verts) # (batch, 65536, 4)

        return inp, local_coordinates


    def run_network(self, inp, local_coordinates, viewdir, sp_input):
        # mask points far away from mesh surface
        mask = self.get_mask(inp)
        inp_f, mask_f = inp.view(-1, inp.shape[-1]), mask.view(-1)
        raw_all = torch.zeros(*mask_f.shape, 4).to(inp.device)
        # if all points are far from body
        if not torch.any(mask_f):
            return raw_all.view(*mask.shape, 4) # rgbd
        inp_f_masked = inp_f[mask_f].unsqueeze(0) # (1, len(mask_f), 4) 

        # positional encoding for inp
        inp_f_masked = embedder.inp_embedder(inp_f_masked).transpose(1, 2) # (1, 52, len(mask_f))

        # get pose embedding
        poses = sp_input['poses'] # (batch, 24, 3)
        poses = poses.transpose(1, 2)
        pose_embed = self.gcn(poses) # (batch, 64, 24)
        pose_embed = torch.mean(pose_embed, dim=-1, keepdim=True) # (batch, 64, 1)
        pose_embed_f = pose_embed.repeat(1, 1, inp_f_masked.shape[-1])

        # get viewdir
        viewdir_f_masked = viewdir.view(-1, 3)[mask_f]
        local_coordinates = local_coordinates[mask_f]
        viewdir_local = torch.matmul(local_coordinates, viewdir_f_masked.unsqueeze(2)).squeeze(2)
        viewdir_f_masked = torch.cat((viewdir_f_masked, viewdir_local), dim=1).unsqueeze(0)

        # positional encoding for viewdir
        viewdir_f_masked = embedder.view_embedder(viewdir_f_masked).transpose(1, 2)

        # forward
        rgb, alpha = self.nerf(inp_f_masked, viewdir_f_masked, pose_embed_f)

        raw = torch.cat((rgb, alpha), dim=1)
        raw = raw.transpose(1, 2) # (1, len(mask_f), 4)

        raw_all[mask_f] = raw.squeeze(0)

        return raw_all.view(*mask.shape, 4)


    def calculate_density_color(self, wpts, viewdir, sp_input):

        inp, local_coordinates = self.prepare_input(wpts, sp_input)
        raw = self.run_network(inp, local_coordinates, viewdir, sp_input)
        
        return raw


class CondMLP(nn.Module):
    def __init__(self):
        super(CondMLP, self).__init__()

        self.l0 = nn.Conv1d(52, 256, 1)
        self.res1 = ResBlock()
        self.res2 = ResBlock()
        self.res3 = ResBlock()
        self.res4 = ResBlock()
        self.res5 = ResBlock()

    def forward(self, x, z):

        x = self.l0(x)
        x = self.res1(x, z)
        x = self.res2(x, z)
        x = self.res3(x, z)
        x = self.res4(x, z)
        x = self.res5(x, z)

        return x


class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.lz = nn.Conv1d(256, 256, 1)
        self.l1 = nn.Conv1d(256, 256, 1)
        self.l2 = nn.Conv1d(256, 256, 1)

    def forward(self, x, z):
        z = F.relu(self.lz(z))
        res = x + z
        x = F.relu(self.l1(res))
        x = F.relu(self.l2(x)) + res
        return x


class NeRF(nn.Module):
    def __init__(self):
        super(NeRF, self).__init__()
        self.mlp = CondMLP()
        self.alpha_l1 = nn.Conv1d(256, 1, 1)
        self.rgb_l1 = nn.Conv1d(310, 128, 1) # for shallow net
        self.rgb_l2 = nn.Conv1d(128, 3, 1)

    def forward(self, x, d, z):
        feat = self.mlp(x, z)
        # density
        alpha = self.alpha_l1(feat)
        # rgb
        feat = torch.cat((feat, d), dim=1)
        rgb = F.relu(self.rgb_l1(feat))
        rgb = self.rgb_l2(rgb)
        return rgb, alpha