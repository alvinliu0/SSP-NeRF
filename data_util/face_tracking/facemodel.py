import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F


class Face_3DMM(nn.Module):
    def __init__(self, modelpath, id_dim, exp_dim, tex_dim, point_num):
        super(Face_3DMM, self).__init__()
        # id_dim = 100
        # exp_dim = 79
        # tex_dim = 100
        self.point_num = point_num
        DMM_info = np.load(os.path.join(
            modelpath, '3DMM_info.npy'), allow_pickle=True).item()
        base_id = DMM_info['b_shape'][:id_dim, :]
        mu_id = DMM_info['mu_shape']
        base_exp = DMM_info['b_exp'][:exp_dim, :]
        mu_exp = DMM_info['mu_exp']
        mu = mu_id + mu_exp
        mu = mu.reshape(-1, 3)
        for i in range(3):
            mu[:, i] -= np.mean(mu[:, i])
        mu = mu.reshape(-1)
        self.base_id = torch.as_tensor(base_id).cuda()/100000.0
        self.base_exp = torch.as_tensor(base_exp).cuda()/100000.0
        self.mu = torch.as_tensor(mu).cuda()/100000.0
        self.mu_id = torch.as_tensor(mu_id).cuda()/100000.0
        base_tex = DMM_info['b_tex'][:tex_dim, :]
        mu_tex = DMM_info['mu_tex']
        self.base_tex = torch.as_tensor(base_tex).cuda()
        self.mu_tex = torch.as_tensor(mu_tex).cuda()
        sig_id = DMM_info['sig_shape'][:id_dim]
        sig_tex = DMM_info['sig_tex'][:tex_dim]
        sig_exp = DMM_info['sig_exp'][:exp_dim]
        self.sig_id = torch.as_tensor(sig_id).cuda()
        self.sig_tex = torch.as_tensor(sig_tex).cuda()
        self.sig_exp = torch.as_tensor(sig_exp).cuda()

        keys_info = np.load(os.path.join(
            modelpath, 'keys_info.npy'), allow_pickle=True).item()
        self.keyinds = torch.as_tensor(keys_info['keyinds']).cuda()
        self.left_contours = torch.as_tensor(keys_info['left_contour']).cuda()
        self.right_contours = torch.as_tensor(
            keys_info['right_contour']).cuda()
        self.rigid_ids = torch.as_tensor(keys_info['rigid_ids']).cuda()

    def get_3dlandmarks(self, id_para, exp_para, euler_angle, trans, focal_length, cxy):
        id_para = id_para*self.sig_id
        exp_para = exp_para*self.sig_exp
        batch_size = id_para.shape[0]
        num_per_contour = self.left_contours.shape[1]
        left_contours_flat = self.left_contours.reshape(-1)
        right_contours_flat = self.right_contours.reshape(-1)
        sel_index = torch.cat((3*left_contours_flat.unsqueeze(1), 3*left_contours_flat.unsqueeze(1)+1,
                               3*left_contours_flat.unsqueeze(1)+2), dim=1).reshape(-1)
        left_geometry = torch.mm(id_para, self.base_id[:, sel_index]) + \
            torch.mm(exp_para, self.base_exp[:,
                     sel_index]) + self.mu[sel_index]
        left_geometry = left_geometry.view(batch_size, -1, 3)
        proj_x = forward_transform(
            left_geometry, euler_angle, trans, focal_length, cxy)[:, :, 0]
        proj_x = proj_x.reshape(batch_size, 8, num_per_contour)
        arg_min = proj_x.argmin(dim=2)
        left_geometry = left_geometry.view(batch_size*8, num_per_contour, 3)
        left_3dlands = left_geometry[torch.arange(
            batch_size*8), arg_min.view(-1), :].view(batch_size, 8, 3)

        sel_index = torch.cat((3*right_contours_flat.unsqueeze(1), 3*right_contours_flat.unsqueeze(1)+1,
                               3*right_contours_flat.unsqueeze(1)+2), dim=1).reshape(-1)
        right_geometry = torch.mm(id_para, self.base_id[:, sel_index]) + \
            torch.mm(exp_para, self.base_exp[:,
                     sel_index]) + self.mu[sel_index]
        right_geometry = right_geometry.view(batch_size, -1, 3)
        proj_x = forward_transform(
            right_geometry, euler_angle, trans, focal_length, cxy)[:, :, 0]
        proj_x = proj_x.reshape(batch_size, 8, num_per_contour)
        arg_max = proj_x.argmax(dim=2)
        right_geometry = right_geometry.view(batch_size*8, num_per_contour, 3)
        right_3dlands = right_geometry[torch.arange(
            batch_size*8), arg_max.view(-1), :].view(batch_size, 8, 3)

        sel_index = torch.cat((3*self.keyinds.unsqueeze(1), 3*self.keyinds.unsqueeze(1)+1,
                               3*self.keyinds.unsqueeze(1)+2), dim=1).reshape(-1)
        geometry = torch.mm(id_para, self.base_id[:, sel_index]) + \
            torch.mm(exp_para, self.base_exp[:,
                                             sel_index]) + self.mu[sel_index]
        lands_3d = geometry.view(-1, self.keyinds.shape[0], 3)
        lands_3d[:, :8, :] = left_3dlands
        lands_3d[:, 9:17, :] = right_3dlands
        return lands_3d

    def forward_geo_sub(self, id_para, exp_para, sub_index):
        id_para = id_para*self.sig_id
        exp_para = exp_para*self.sig_exp
        sel_index = torch.cat((3*sub_index.unsqueeze(1), 3*sub_index.unsqueeze(1)+1,
                               3*sub_index.unsqueeze(1)+2), dim=1).reshape(-1)
        geometry = torch.mm(id_para, self.base_id[:, sel_index]) + \
            torch.mm(exp_para, self.base_exp[:,
                                             sel_index]) + self.mu[sel_index]
        return geometry.reshape(-1, sub_index.shape[0], 3)

    def forward_geo(self, id_para, exp_para):
        id_para = id_para*self.sig_id
        exp_para = exp_para*self.sig_exp
        geometry = torch.mm(id_para, self.base_id) + \
            torch.mm(exp_para, self.base_exp) + self.mu
        return geometry.reshape(-1, self.point_num, 3)

    def forward_id(self, id_para):
        id_para = id_para*self.sig_id
        geometry = torch.mm(id_para, self.base_id) + self.mu
        return geometry.reshape(-1, self.point_num, 3)

    def forward_tex(self, tex_para):
        tex_para = tex_para*self.sig_tex
        texture = torch.mm(tex_para, self.base_tex) + self.mu_tex
        return texture.reshape(-1, self.point_num, 3)


def compute_tri_normal(geometry, tris):
    tri_1 = tris[:, 0]
    tri_2 = tris[:, 1]
    tri_3 = tris[:, 2]
    vert_1 = torch.index_select(geometry, 1, tri_1)
    vert_2 = torch.index_select(geometry, 1, tri_2)
    vert_3 = torch.index_select(geometry, 1, tri_3)
    nnorm = torch.cross(vert_2-vert_1, vert_3-vert_1, 2)
    normal = nn.functional.normalize(nnorm)
    return normal


def euler2rot(euler_angle):
    batch_size = euler_angle.shape[0]
    theta = euler_angle[:, 0].reshape(-1, 1, 1)
    phi = euler_angle[:, 1].reshape(-1, 1, 1)
    psi = euler_angle[:, 2].reshape(-1, 1, 1)
    one = torch.ones(batch_size, 1, 1).to(euler_angle.device)
    zero = torch.zeros(batch_size, 1, 1).to(euler_angle.device)
    rot_x = torch.cat((
        torch.cat((one, zero, zero), 1),
        torch.cat((zero, theta.cos(), theta.sin()), 1),
        torch.cat((zero, -theta.sin(), theta.cos()), 1),
    ), 2)
    rot_y = torch.cat((
        torch.cat((phi.cos(), zero, -phi.sin()), 1),
        torch.cat((zero, one, zero), 1),
        torch.cat((phi.sin(), zero, phi.cos()), 1),
    ), 2)
    rot_z = torch.cat((
        torch.cat((psi.cos(), -psi.sin(), zero), 1),
        torch.cat((psi.sin(), psi.cos(), zero), 1),
        torch.cat((zero, zero, one), 1)
    ), 2)
    return torch.bmm(rot_x, torch.bmm(rot_y, rot_z))


def rot_trans_pts(geometry, rot, trans):
    rott_geo = torch.bmm(rot, geometry.permute(0, 2, 1)) + trans[:, :, None]
    return rott_geo.permute(0, 2, 1)


def cal_lap_loss(tensor_list, weight_list):
    lap_kernel = torch.Tensor(
        (-0.5, 1.0, -0.5)).unsqueeze(0).unsqueeze(0).float().to(tensor_list[0].device)
    loss_lap = 0
    for i in range(len(tensor_list)):
        in_tensor = tensor_list[i]
        in_tensor = in_tensor.view(-1, 1, in_tensor.shape[-1])
        out_tensor = F.conv1d(in_tensor, lap_kernel)
        loss_lap += torch.mean(out_tensor**2)*weight_list[i]
    return loss_lap


def proj_pts(rott_geo, focal_length, cxy):
    cx, cy = cxy[0], cxy[1]
    X = rott_geo[:, :, 0]
    Y = rott_geo[:, :, 1]
    Z = rott_geo[:, :, 2]
    fxX = focal_length*X
    fyY = focal_length*Y
    proj_x = -fxX/Z + cx
    proj_y = fyY/Z + cy
    return torch.cat((proj_x[:, :, None], proj_y[:, :, None], Z[:, :, None]), 2)

def forward_rott(geometry, euler_angle, trans):
    rot = euler2rot(euler_angle)
    rott_geo = rot_trans_pts(geometry, rot, trans)
    return rott_geo

def forward_transform(geometry, euler_angle, trans, focal_length, cxy):
    rot = euler2rot(euler_angle)
    rott_geo = rot_trans_pts(geometry, rot, trans)
    proj_geo = proj_pts(rott_geo, focal_length, cxy)
    return proj_geo

def reverse_proj_pts(rott_geo, focal_length, cxy):
    proj_x = rott_geo[:, :, 0]
    proj_y = rott_geo[:, :, 1]
    Z = rott_geo[:, :, 2]
    cx, cy = cxy[0], cxy[1]
    fxX = (cx - proj_x) * Z
    fyY = (proj_y - cy) * Z
    X = fxX / focal_length
    Y = fyY / focal_length
    return torch.cat((X[:, :, None], Y[:, :, None], Z[:, :, None]), 2)

def forward_world2mm(geometry, euler_angle, trans, focal_length, cxy):
    rot = euler2rot(euler_angle)
    rott_geo = reverse_proj_pts(geometry, focal_length, cxy)
    rott_geo = rott_geo.permute(0, 2, 1) - trans[:, :, None]
    rott_geo = torch.bmm(rot.permute(0, 2, 1), rott_geo).permute(0, 2, 1)
    return rott_geo


def cal_lan_loss(proj_lan, gt_lan):
    return torch.mean((proj_lan-gt_lan)**2)

def cal_col_loss(pred_img, gt_img, img_mask):
    pred_img = pred_img.float()
    loss = torch.sqrt(torch.sum(torch.square(pred_img - gt_img), 3))*img_mask/255
    loss = torch.sum(loss, dim=(1, 2)) / torch.sum(img_mask, dim=(1, 2))
    loss = torch.mean(loss)
    return loss
