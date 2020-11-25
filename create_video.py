import logging
import os
import cv2
import torch
from tqdm import tqdm
import numpy as np
import pickle
import argparse
import glob

orig_img = cv2.imread('data/000002.jpg')
h, w = orig_img.shape[:2]
files = glob.glob('out_*.jpg')
files = sorted(files, key=lambda x: int(x.split('_')[-1][:-4]))


# print('start creating video with fitted model')
out = cv2.VideoWriter('outvideo.avi', cv2.VideoWriter_fourcc(*'DIVX'), 25, (w*2, h))
for i, f in enumerate(files):
	if i % 4 == 0:
		cur_img = cv2.imread(f)
		out_img = np.concatenate([orig_img, cur_img], axis=1)
		out.write(out_img)
out.release()

# 	with torch.no_grad():
# 		for k in tqdm(keys):
# 			fa_params = fa_dict[k]
# 			bbox = fa_params['bbox']
# 			l, t, r, b = bbox
# 			scale = float(fa_params['scale'])
# 			frame_img = cv2.imread(os.path.join(tar_folder, 'frames', 'frame_%d.png'%k))

# 			exp_tensor = torch.tensor(params_dict[k]['exp'], dtype=torch.float32, device='cuda')
# 			rot_tensor = torch.tensor(params_dict[k]['rot'], dtype=torch.float32, device='cuda')
# 			gamma_tensor = torch.tensor(params_dict[k]['gamma'], dtype=torch.float32, device='cuda')
# 			tran_tensor = torch.tensor(params_dict[k]['trans'], dtype=torch.float32, device='cuda')

# 			coeff = torch.cat([id_tensor,
# 							exp_tensor,
# 							tex_tensor,
# 							rot_tensor,
# 							gamma_tensor,
# 							tran_tensor]).unsqueeze(0)
# 			rendered_img, pred_lms, face_texture = model(coeff)
# 			pred = rendered_img.cpu().numpy().squeeze()
# 			out_img = pred[:, :, :3].astype(np.uint8)
# 			out_mask = (pred[:, :, 3]>0).astype(np.uint8)
# 			tar_size = r - l
# 			resized_img = cv2.resize(out_img, (tar_size, tar_size))[:, :, ::-1]
# 			resized_mask = cv2.resize(out_mask, (tar_size, tar_size), cv2.INTER_NEAREST)[..., None]
			
# 			frame_img[t:b, l:r, :] = frame_img[t:b, l:r, :] * (1 - resized_mask) + resized_img * resized_mask
# 			out.write(frame_img)
# 	out.release()
# 	print('video saved at %s' % outpath)


