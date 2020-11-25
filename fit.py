import os
import torch
from core.utils import pad_bbox
from tqdm import tqdm
from core.losses import photo_loss, lm_loss, reg_loss, reflectance_loss, gamma_loss
from core.models import ReconModel
from scipy.io import loadmat
import numpy as np
import argparse
import face_alignment
import cv2

TAR_SIZE = 256 # size for rendering window
PADDING_RATIO = 0.3 # enlarge the face detection bbox by a margin
FACE_MODEL_PATH = 'BFM/BFM_model_front.mat'

#params at rigid fitting stage
RF_ITERS = 300 # iter number for the first frame
RF_LR = 0.01 #learning rate

#params at non-rigid fitting stage
NRF_ITERS = 200 #epoch number
NRF_LR = 0.01 #learning rate
NRF_PHOTO_LOSS_W = 1.6
NRF_LM_LOSS_W = 100
NRF_REG_W = 1e-3
NRF_TEX_LOSS_W = 1

def train(args):
	img_path = args.img
	img_output_path = args.outimg

	fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)
	print('loading images')
	orig_img = cv2.imread(img_path)[:, :, ::-1]
	h, w = orig_img.shape[:2]
	print('image is loaded. width: %d, height: %d' % (w, h))
	print('processing the image')
	lms = fa.get_landmarks_from_image(orig_img)[0]
	bbox = [int(lms[:, 0].min()), int(lms[:, 1].min()), 
            int(lms[:, 0].max()), int(lms[:, 1].max())] # left, top, right, bottom
	padded_bbox = pad_bbox(bbox, (w, h), padding_ratio=PADDING_RATIO)
	l, t, r, b = padded_bbox
	cropped_img = orig_img[t:b, l:r, :]
	crop_size = cropped_img.shape[0]
	scale = TAR_SIZE / float(crop_size)
	cropped_img = cv2.resize(cropped_img, (TAR_SIZE, TAR_SIZE))
	lms[:, 0] -= l
	lms[:, 1] -= t
	lms *= scale
	lms = lms[:, :2][None, ...]
	lms = torch.tensor(lms, dtype=torch.float32).cuda()
	img_tensor = torch.tensor(cropped_img[None, ...], dtype=torch.float32).cuda()

	print('loading facemodel')
	try:
		facemodel = loadmat(FACE_MODEL_PATH)
	except Exception as e:
		print('failed to load %s' % FACE_MODEL_PATH)
	skinmask = torch.tensor(facemodel['skinmask']).cuda()

	model = ReconModel(facemodel, img_size=TAR_SIZE)
	model.train()
	model.cuda()

	id_tensor = torch.zeros((1, 80), dtype=torch.float32, requires_grad=True, device='cuda')
	tex_tensor = torch.zeros((1, 64), dtype=torch.float32, requires_grad=True, device='cuda')
	exp_tensor = torch.zeros((1, 80), dtype=torch.float32, requires_grad=True, device='cuda')
	rot_tensor = torch.zeros((1, 3), dtype=torch.float32, requires_grad=True, device='cuda')
	gamma_tensor = torch.zeros((1, 27), dtype=torch.float32, requires_grad=True, device='cuda')
	trans_tensor = torch.zeros((1, 3), dtype=torch.float32, requires_grad=True, device='cuda')

	print('start rigid fitting')
	rigid_optimizer = torch.optim.Adam([rot_tensor, trans_tensor], lr=RF_LR)
	for i in tqdm(range(RF_ITERS)):
		rigid_optimizer.zero_grad()
		coeff = torch.cat([id_tensor, exp_tensor,
						tex_tensor, rot_tensor,
						gamma_tensor, trans_tensor], dim=1)
		_, pred_lms, _, _ = model(coeff)
		lm_loss_val = lm_loss(pred_lms, lms, img_size=TAR_SIZE)
		lm_loss_val.backward()
		rigid_optimizer.step()
	print('start non-rigid fitting')
	nonrigid_optimizer = torch.optim.Adam([id_tensor, tex_tensor,
										exp_tensor, rot_tensor,
										gamma_tensor, trans_tensor], lr=NRF_LR)
	for i in tqdm(range(NRF_ITERS)):
		nonrigid_optimizer.zero_grad()
		coeff = torch.cat([id_tensor, exp_tensor,
						tex_tensor, rot_tensor,
						gamma_tensor, trans_tensor], dim=1)
		rendered_img, pred_lms, face_texture, _ = model(coeff)
		mask = rendered_img[:, :, :, 3].detach()
		photo_loss_val = photo_loss(rendered_img[:, :, :, :3], img_tensor, mask>0)
		lm_loss_val = lm_loss(pred_lms, lms, img_size=TAR_SIZE)
		reg_loss_val = reg_loss(id_tensor, exp_tensor, tex_tensor)
		tex_loss_val = reflectance_loss(face_texture, skinmask)
		loss = photo_loss_val*NRF_PHOTO_LOSS_W + \
						lm_loss_val*NRF_LM_LOSS_W + \
						reg_loss_val*NRF_REG_W + \
						tex_loss_val*NRF_TEX_LOSS_W
		loss.backward()
		nonrigid_optimizer.step()
	
	with torch.no_grad():
		coeff = torch.cat([id_tensor, exp_tensor,
						tex_tensor, rot_tensor,
						gamma_tensor, trans_tensor], dim=1)
		rendered_img, pred_lms, face_texture, mesh = model(coeff)
		print('saving results')
		rendered_img = rendered_img.cpu().numpy().squeeze()
		out_img = rendered_img[:, :, :3].astype(np.uint8)
		out_mask = (rendered_img[:, :, 3]>0).astype(np.uint8)
		resized_img = cv2.resize(out_img, (crop_size, crop_size))
		resized_mask = cv2.resize(out_mask, (crop_size, crop_size), cv2.INTER_NEAREST)[..., None]
		composed_img = orig_img.copy()
		l, t, r, b = padded_bbox
		composed_img[t:b, l:r, :] = resized_img * resized_mask + \
									composed_img[t:b, l:r, :] * (1 - resized_mask)
		composed_img = composed_img[:, :, ::-1]
		if args.outimg is None:
			out_img_path = args.img.replace('.jpg', '_composed.jpg')
		else:
			out_img_path = args.outimg
		cv2.imwrite(out_img_path, composed_img)
		print('composed image is saved at %s' % out_img_path)
		


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--img', type=str, help='image path')
	parser.add_argument('--outimg', type=str, default=None, help='output path for rendered image')
	args = parser.parse_args()
	train(args)
