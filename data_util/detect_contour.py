import cv2
import numpy as np
import os
import glob
import argparse
from pathlib import Path
from tqdm import tqdm

TORSO_RANGE_DOWN = 10
TORSO_RANGE_UP = 20

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str,
                    default='Obama', help='identity of target person')

args = parser.parse_args()

all_imgs = glob.glob(os.path.join('dataset', args.id, 'parsing/*.png'))

contour_dir = os.path.join('dataset', args.id, 'contours')
Path(contour_dir).mkdir(parents=True, exist_ok=True)
contour_img_dir = os.path.join('dataset', args.id, 'contours_img')
Path(contour_img_dir).mkdir(parents=True, exist_ok=True)

print(len(all_imgs))

for file in tqdm(all_imgs):
    img = cv2.imread(file)
    img_id = file.split('/')[-1][:-4]
    res = []
    coords = np.stack(np.meshgrid(np.arange(450), np.arange(450)), -1)
    torso_index = np.all(img[..., :] == [0, 0, 255], axis = -1)
    torso_coord = coords[torso_index]
    # print(torso_index.shape)
    for x in range(img.shape[0]):
        torso_coord_line = torso_coord[torso_coord[:, 0] == x][:, 1]
        if torso_coord_line.shape[0] != 0:
            y = np.min(torso_coord_line)
            if (img[y - 1, x] == [255, 255, 255]).all():
                res = res + [[x, y]]
                for i in range(1, TORSO_RANGE_DOWN):
                    res = res + [[x, y + i]]
                for i in range(1, TORSO_RANGE_UP):
                    res = res + [[x, y - i]]
    res = np.array(res)
    np.savetxt(os.path.join(contour_dir, img_id + '.ctr'), res, '%d')

    new_img = np.zeros((450, 450, 3))
    for i in range(res.shape[0]):
        entry = res[i]
        new_img[entry[1], entry[0]] = [255, 255, 255]
    cv2.imwrite(os.path.join(contour_img_dir, img_id + '.png'), new_img)