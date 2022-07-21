###################################################################
# The code is developed based on AD-NeRF of Guo et al. (ICCV 2021)
# Github Link: https://github.com/YudongGuo/AD-NeRF
###################################################################

import cv2
import numpy as np
import face_alignment
from skimage import io
import torch
import torch.nn.functional as F
import json
import os
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
import argparse
from face_tracking.facemodel import Face_3DMM, forward_transform, forward_world2mm

def euler2rot(euler_angle):
    batch_size = euler_angle.shape[0]
    theta = euler_angle[:, 0].reshape(-1, 1, 1)
    phi = euler_angle[:, 1].reshape(-1, 1, 1)
    psi = euler_angle[:, 2].reshape(-1, 1, 1)
    one = torch.ones((batch_size, 1, 1), dtype=torch.float32,
                     device=euler_angle.device)
    zero = torch.zeros((batch_size, 1, 1), dtype=torch.float32,
                       device=euler_angle.device)
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

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str,
                    default='obama', help='identity of target person')
parser.add_argument('--step', type=int,
                    default=0, help='step for running')

args = parser.parse_args()
id = args.id
vid_file = os.path.join('dataset', 'vids', id+'.mp4')
if not os.path.isfile(vid_file):
    print('no video')
    exit()


id_dir = os.path.join('dataset', id)
Path(id_dir).mkdir(parents=True, exist_ok=True)
ori_imgs_dir = os.path.join('dataset', id, 'ori_imgs')
Path(ori_imgs_dir).mkdir(parents=True, exist_ok=True)
parsing_dir = os.path.join(id_dir, 'parsing')
Path(parsing_dir).mkdir(parents=True, exist_ok=True)
head_imgs_dir = os.path.join('dataset', id, 'head_imgs')
Path(head_imgs_dir).mkdir(parents=True, exist_ok=True)
com_imgs_dir = os.path.join('dataset', id, 'com_imgs')
Path(com_imgs_dir).mkdir(parents=True, exist_ok=True)
vertex_dir = os.path.join('dataset', id, 'vertices')
Path(vertex_dir).mkdir(parents=True, exist_ok=True)
vertex_mean_dir = os.path.join('dataset', id, 'vertices_mean')
Path(vertex_mean_dir).mkdir(parents=True, exist_ok=True)

running_step = args.step

# # Step 0: extract wav & deepspeech feature, better run in terminal to parallel with
# below commands since this may take a few minutes
if running_step == 0:
    print('--- Step 0: extract deepspeech feature ---')
    wav_file = os.path.join(id_dir, 'aud.wav')
    extract_wav_cmd = 'ffmpeg -i ' + vid_file + ' -f wav -ar 16000 ' + wav_file
    os.system(extract_wav_cmd)
    extract_ds_cmd = 'python data_util/deepspeech_features/extract_ds_features.py --input=' + id_dir
    os.system(extract_ds_cmd)
    exit()

# Step 1: extract images
if running_step == 1:
    print('--- Step 1: extract images from vids ---')
    cap = cv2.VideoCapture(vid_file)
    frame_num = 0
    while(True):
        _, frame = cap.read()
        if frame is None:
            break
        frame = cv2.resize(frame, (450, 450))
        cv2.imwrite(os.path.join(ori_imgs_dir, str(frame_num) + '.png'), frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        frame_num = frame_num + 1
    cap.release()
    exit()

# Step 2: detect lands
if running_step == 2:
    print('--- Step 2: detect landmarks ---')
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D, flip_input=False)
    for image_path in os.listdir(ori_imgs_dir):
        if image_path.endswith('.png'):
            input = io.imread(os.path.join(ori_imgs_dir, image_path))[:, :, :3]
            preds = fa.get_landmarks(input)
            if len(preds) > 0:
                lands = preds[0].reshape(-1, 2)[:,:2]
                np.savetxt(os.path.join(ori_imgs_dir, image_path[:-3] + 'lms'), lands, '%f')
        
max_frame_num = 15000
valid_img_ids = []
for i in range(max_frame_num):
    if os.path.isfile(os.path.join(ori_imgs_dir, str(i) + '.lms')):
        valid_img_ids.append(i)
valid_img_num = len(valid_img_ids)
tmp_img = cv2.imread(os.path.join(ori_imgs_dir, str(valid_img_ids[0])+'.png'))
h, w = tmp_img.shape[0], tmp_img.shape[1]

# Step 3: face parsing
if running_step == 3:
    print('--- Step 3: face parsing ---')
    face_parsing_cmd = 'python data_util/face_parsing/test.py --respath=dataset/' + \
        id + '/parsing --imgpath=dataset/' + id + '/ori_imgs'
    os.system(face_parsing_cmd)

# Step 4: extract bc image
if running_step == 4:
    print('--- Step 4: extract background image ---')
    sel_ids = np.array(valid_img_ids)[np.arange(0, valid_img_num, 20)]
    all_xys = np.mgrid[0:h, 0:w].reshape(2, -1).transpose()
    distss = []
    for i in sel_ids:
        parse_img = cv2.imread(os.path.join(id_dir, 'parsing', str(i) + '.png'))
        bg = (parse_img[..., 0] == 255) & (
            parse_img[..., 1] == 255) & (parse_img[..., 2] == 255)
        fg_xys = np.stack(np.nonzero(~bg)).transpose(1, 0)
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(fg_xys)
        dists, _ = nbrs.kneighbors(all_xys)
        distss.append(dists)
    distss = np.stack(distss)
    print(distss.shape)
    max_dist = np.max(distss, 0)
    max_id = np.argmax(distss, 0)
    bc_pixs = max_dist > 5
    bc_pixs_id = np.nonzero(bc_pixs)
    bc_ids = max_id[bc_pixs]
    imgs = []
    num_pixs = distss.shape[1]
    for i in sel_ids:
        img = cv2.imread(os.path.join(ori_imgs_dir, str(i) + '.png'))
        imgs.append(img)
    imgs = np.stack(imgs).reshape(-1, num_pixs, 3)
    bc_img = np.zeros((h*w, 3), dtype=np.uint8)
    bc_img[bc_pixs_id, :] = imgs[bc_ids, bc_pixs_id, :]
    bc_img = bc_img.reshape(h, w, 3)
    max_dist = max_dist.reshape(h, w)
    bc_pixs = max_dist > 5
    bg_xys = np.stack(np.nonzero(~bc_pixs)).transpose()
    fg_xys = np.stack(np.nonzero(bc_pixs)).transpose()
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(fg_xys)
    distances, indices = nbrs.kneighbors(bg_xys)
    bg_fg_xys = fg_xys[indices[:, 0]]
    print(fg_xys.shape)
    print(np.max(bg_fg_xys), np.min(bg_fg_xys))
    bc_img[bg_xys[:, 0], bg_xys[:, 1],
        :] = bc_img[bg_fg_xys[:, 0], bg_fg_xys[:, 1], :]
    cv2.imwrite(os.path.join(id_dir, 'bc.png'), bc_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

# Step 5: save training images
if running_step == 5:
    print('--- Step 5: save training images ---')
    bc_img = cv2.imread(os.path.join(id_dir, 'bc.png'))
    for i in valid_img_ids:
        parsing_img = cv2.imread(os.path.join(parsing_dir, str(i) + '.png'))
        head_part = (parsing_img[:, :, 0] == 255) & (
            parsing_img[:, :, 1] == 0) & (parsing_img[:, :, 2] == 0)
        bc_part = (parsing_img[:, :, 0] == 255) & (
            parsing_img[:, :, 1] == 255) & (parsing_img[:, :, 2] == 255)
        img = cv2.imread(os.path.join(ori_imgs_dir, str(i) + '.png'))
        img[bc_part] = bc_img[bc_part]
        cv2.imwrite(os.path.join(com_imgs_dir, str(i) + '.png'), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        img[~head_part] = bc_img[~head_part]
        cv2.imwrite(os.path.join(head_imgs_dir, str(i) + '.png'), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

# Step 6: estimate head pose
if running_step == 6:
    print('--- Step 6: Estimate Head Pose ---')
    est_pose_cmd = 'python data_util/face_tracking/face_tracker.py --idname=' + \
        id + ' --img_h=' + str(h) + ' --img_w=' + str(w) + \
        ' --frame_num=' + str(max_frame_num)
    os.system(est_pose_cmd)
    exit()

# Step 7: face parsing (11 class)
if running_step == 7:
    print('--- Step 7: face parsing 11 class ---')
    face_parsing_cmd = 'python data_util/face_parsing/test_11_class.py --respath=dataset/' + \
        id + '/parsing --imgpath=dataset/' + id + '/ori_imgs'
    os.system(face_parsing_cmd)

# Step 8: save transform param & write config file
if running_step == 8:
    print('--- Step 8: Save Transform Param ---')
    params_dict = torch.load(os.path.join(id_dir, 'track_params.pt'))
    focal_len = params_dict['focal']
    euler_angle = params_dict['euler']
    raw_euler = params_dict['euler']
    raw_trans = params_dict['trans'] / 10.0
    trans = params_dict['trans'] / 10.0
    exp_para = params_dict['exp']
    id_para = params_dict['id']
    valid_num = euler_angle.shape[0]
    train_val_split = int(valid_num*10/11)
    train_ids = torch.arange(0, train_val_split)
    val_ids = torch.arange(train_val_split, valid_num)
    rot = euler2rot(euler_angle)
    rot_inv = rot.permute(0, 2, 1)
    trans_inv = -torch.bmm(rot_inv, trans.unsqueeze(2))
    pose = torch.eye(4, dtype=torch.float32)
    save_ids = ['train', 'val']
    train_val_ids = [train_ids, val_ids]
    mean_z = -float(torch.mean(trans[:, 2]).item())
    for i in range(2):
        transform_dict = dict()
        transform_dict['focal_len'] = float(focal_len[0])
        transform_dict['cx'] = float(w/2.0)
        transform_dict['cy'] = float(h/2.0)
        transform_dict['id_para'] = id_para[0].numpy().tolist()
        transform_dict['frames'] = []
        ids = train_val_ids[i]
        save_id = save_ids[i]
        for i in ids:
            i = i.item()
            frame_dict = dict()
            frame_dict['img_id'] = int(valid_img_ids[i])
            frame_dict['aud_id'] = int(valid_img_ids[i])
            pose[:3, :3] = rot_inv[i]
            pose[:3, 3] = trans_inv[i, :, 0]
            frame_dict['transform_matrix'] = pose.numpy().tolist()
            frame_dict['exp_para'] = exp_para[i].numpy().tolist()
            frame_dict['euler'] = raw_euler[i].numpy().tolist()
            frame_dict['trans'] = raw_trans[i].numpy().tolist()
            lms = np.loadtxt(os.path.join(
                ori_imgs_dir, str(valid_img_ids[i]) + '.lms'))
            min_x, max_x = np.min(lms, 0)[0], np.max(lms, 0)[0]
            cx = int((min_x+max_x)/2.0)
            cy = int(lms[27, 1])
            h_w = int((max_x-cx)*1.5)
            h_h = int((lms[8, 1]-cy)*1.15)
            rect_x = cx - h_w
            rect_y = cy - h_h
            if rect_x < 0:
                rect_x = 0
            if rect_y < 0:
                rect_y = 0
            rect_w = min(w-1-rect_x, 2*h_w)
            rect_h = min(h-1-rect_y, 2*h_h)
            rect = np.array((rect_x, rect_y, rect_w, rect_h), dtype=np.int32)
            frame_dict['face_rect'] = rect.tolist()
            transform_dict['frames'].append(frame_dict)
        with open(os.path.join(id_dir, 'transforms_' + save_id + '.json'), 'w') as fp:
            json.dump(transform_dict, fp, indent=2, separators=(',', ': '))

    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    testskip = int(val_ids.shape[0]/7)

    HeadNeRF_config_file = os.path.join(id_dir, 'HeadNeRF_config.txt')
    with open(HeadNeRF_config_file, 'w') as file:
        file.write('expname = ' + id + '\n')
        file.write('datadir = ' + os.path.join(dir_path, 'dataset', id) + '\n')
        file.write('basedir = ' + os.path.join(dir_path,
                                            'dataset', id, 'logs') + '\n')
        file.write('near = ' + str(mean_z-0.2) + '\n')
        file.write('far = ' + str(mean_z+0.4) + '\n')
        file.write('testskip = ' + str(testskip) + '\n')
        # file.write('test_pose_file = transforms_val.json' + '\n')
    Path(os.path.join(dir_path, 'dataset', id, 'logs')
        ).mkdir(parents=True, exist_ok=True)

# Step 9: save 3D point coordinate
if running_step == 9:
    print('--- Step 9: Save Point Coordinate ---')
    id_dim, exp_dim, tex_dim, point_num = 100, 79, 100, 34650
    model_3dmm = Face_3DMM(os.path.join('data_util', 'face_tracking', '3DMM'),
                        id_dim, exp_dim, tex_dim, point_num)
    params_dict = torch.load(os.path.join(id_dir, 'track_params.pt'))
    id_para = params_dict['id'].cuda()
    focal_length = params_dict['focal'].unsqueeze(0).cuda()

    exp_para = params_dict['exp'].cuda()
    euler = params_dict['euler'].cuda()
    trans = params_dict['trans'].cuda()
    id_para = id_para.expand(exp_para.shape[0], -1)

    geometry = model_3dmm.forward_geo(id_para, exp_para)

    print(id_para.shape, exp_para.shape, geometry.shape)

    for i in range(exp_para.shape[0]):
        img_id = valid_img_ids[i]
        np.save(os.path.join(vertex_dir, str(img_id) + '.npy'), geometry[i].cpu().numpy())

# Step 10: save mean exp 3D point coordinate
if running_step == 10:
    print('--- Step 10: Save Mean Exp Point Coordinate ---')
    id_dim, exp_dim, tex_dim, point_num = 100, 79, 100, 34650
    model_3dmm = Face_3DMM(os.path.join('data_util', 'face_tracking', '3DMM'),
                        id_dim, exp_dim, tex_dim, point_num)
    params_dict = torch.load(os.path.join(id_dir, 'track_params.pt'))
    id_para = params_dict['id'].cuda()
    focal_length = params_dict['focal'].unsqueeze(0).cuda()

    # set the exp parameters as 0 to get mean exp
    exp_para = torch.zeros_like(params_dict['exp']).cuda()
    euler = params_dict['euler'].cuda()
    trans = params_dict['trans'].cuda()
    id_para = id_para.expand(exp_para.shape[0], -1)

    geometry = model_3dmm.forward_geo(id_para, exp_para)

    print(id_para.shape, exp_para.shape, geometry.shape)

    for i in range(exp_para.shape[0]):
        img_id = valid_img_ids[i]
        np.save(os.path.join(vertex_mean_dir, str(img_id) + '.npy'), geometry[i].cpu().numpy())

    print(id + ' data processed done!')