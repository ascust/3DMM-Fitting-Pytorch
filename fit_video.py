from facenet_pytorch import MTCNN
from core.options import VideoFittingOptions
from PIL import Image
import cv2
import face_alignment
import numpy as np
from core import get_recon_model
import os
import torch
import core.utils as utils
from tqdm import tqdm
import core.losses as losses
import shutil
import random
import glob
import pickle
from multiprocessing import Process, set_start_method
from core.fitting_dataset import FittingDataset


def fit_coeffs(args, device, worker_ind):
    id_coeff = np.load(args.id_npy_path)
    tex_coeff = np.load(args.tex_npy_path)
    recon_model = get_recon_model(model=args.recon_model,
                                  device=device,
                                  batch_size=1,
                                  img_size=args.tar_size)
    recon_model.init_coeff_tensors(id_coeff=id_coeff, tex_coeff=tex_coeff)
    fitting_dataset = FittingDataset(
        args.tmp_face_folder, args.lm_pkl_path,
        worker_num=args.nworkers, worker_ind=worker_ind)
    fitting_data_loader = torch.utils.data.DataLoader(dataset=fitting_dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      num_workers=1,
                                                      drop_last=False)
    lm_weights = utils.get_lm_weights(device)
    res_dict = {}
    num_imgs = len(fitting_dataset)
    last_rot = None
    last_trans = None
    for batch_ind, cur_batch in tqdm(enumerate(fitting_data_loader)):
        print('fitting %d/%d' % (batch_ind, num_imgs))
        lms, img, img_keys = cur_batch
        lms = lms.to(device)
        imgs = img.to(device)
        rigid_optimizer = torch.optim.Adam([recon_model.get_rot_tensor(),
                                            recon_model.get_trans_tensor()],
                                           lr=args.rf_lr)
        num_iters = args.first_rf_iters if batch_ind == 0 else args.rest_rf_iters
        for i in range(num_iters):
            rigid_optimizer.zero_grad()
            pred_dict = recon_model(
                recon_model.get_packed_tensors(), render=False)
            lm_loss_val = losses.lm_loss(
                pred_dict['lms_proj'], lms, lm_weights, img_size=args.tar_size)
            total_loss = args.lm_loss_w * lm_loss_val
            total_loss.backward()
            rigid_optimizer.step()
        print('done rigid fitting. lm_loss: %f' %
              lm_loss_val.detach().cpu().numpy())

        print('start non-rigid fitting')
        nonrigid_optimizer = torch.optim.Adam(
            [recon_model.get_id_tensor(), recon_model.get_exp_tensor(),
             recon_model.get_gamma_tensor(), recon_model.get_tex_tensor(),
             recon_model.get_rot_tensor(), recon_model.get_trans_tensor()],
            lr=args.nrf_lr)
        num_iters = args.first_nrf_iters if batch_ind == 0 else args.rest_nrf_iters
        for i in range(num_iters):
            nonrigid_optimizer.zero_grad()

            pred_dict = recon_model(
                recon_model.get_packed_tensors(), render=True)
            rendered_img = pred_dict['rendered_img']
            lms_proj = pred_dict['lms_proj']
            face_texture = pred_dict['face_texture']

            mask = rendered_img[:, :, :, 3].detach()

            photo_loss_val = losses.photo_loss(
                rendered_img[:, :, :, :3], imgs, mask > 0)

            lm_loss_val = losses.lm_loss(lms_proj, lms, lm_weights,
                                         img_size=args.tar_size)
            id_reg_loss = losses.get_l2(recon_model.get_id_tensor())
            exp_reg_loss = losses.get_l2(recon_model.get_exp_tensor())
            tex_reg_loss = losses.get_l2(recon_model.get_tex_tensor())
            tex_loss_val = losses.reflectance_loss(
                face_texture, recon_model.get_skinmask())

            loss = lm_loss_val*args.lm_loss_w + \
                id_reg_loss*args.id_reg_w + \
                exp_reg_loss*args.exp_reg_w + \
                tex_reg_loss*args.tex_reg_w + \
                tex_loss_val*args.tex_w + \
                photo_loss_val*args.rgb_loss_w

            # regularizers for rotation and translation
            if last_rot is not None:
                rot_diff = recon_model.get_rot_tensor() - last_rot
                trans_diff = recon_model.get_trans_tensor() - last_trans

                rot_reg = torch.square(rot_diff).sum()
                trans_reg = torch.square(trans_diff).sum()
                loss += rot_reg * args.rot_reg_w
                loss += trans_reg * args.trans_reg_w

            loss.backward()
            nonrigid_optimizer.step()
        loss_str = ''
        loss_str += 'lm_loss: %f\t' % lm_loss_val.detach().cpu().numpy()
        loss_str += 'photo_loss: %f\t' % photo_loss_val.detach().cpu().numpy()
        loss_str += 'tex_loss: %f\t' % tex_loss_val.detach().cpu().numpy()
        loss_str += 'id_reg_loss: %f\t' % id_reg_loss.detach().cpu().numpy()
        loss_str += 'exp_reg_loss: %f\t' % exp_reg_loss.detach().cpu().numpy()
        loss_str += 'tex_reg_loss: %f\t' % tex_reg_loss.detach().cpu().numpy()
        if last_rot is not None:
            loss_str += 'rot_reg_loss: %f\t' % rot_reg.detach().cpu().numpy()
            loss_str += 'trans_reg_loss: %f\t' % trans_reg.detach().cpu().numpy()
        print('done non rigid fitting.', loss_str)

        if last_rot is None:
            last_rot = recon_model.get_rot_tensor().detach().clone()
            last_trans = recon_model.get_trans_tensor().detach().clone()

        cur_k = img_keys[0].numpy().item()
        cur_rot = recon_model.get_rot_tensor().detach().cpu().numpy().reshape(1, -1)
        cur_trans = recon_model.get_trans_tensor().detach().cpu().numpy().reshape(1, -1)
        cur_gamma = recon_model.get_gamma_tensor().detach().cpu().numpy().reshape(1, -1)
        cur_exp = recon_model.get_exp_tensor().detach().cpu().numpy().reshape(1, -1)
        res_dict[cur_k] = {'rot': cur_rot,
                           'trans': cur_trans,
                           'gamma': cur_gamma,
                           'exp': cur_exp}

    output_dict = {'id': id_coeff, 'tex': tex_coeff, 'fitting_res': res_dict}
    with open(os.path.join(args.cache_folder, '%d_fitting.pkl' % worker_ind), 'wb') as f:
        pickle.dump(output_dict, f)


def gen_composed_video(args, device):
    with open(args.fitting_pkl_path, 'rb') as f:
        fitting_dict = pickle.load(f)
    id_tensor = torch.tensor(fitting_dict['id'], device=device)
    tex_tensor = torch.tensor(fitting_dict['tex'], device=device)
    with open(args.v_info_path, 'rb') as f:
        video_info = pickle.load(f)

    bbox = video_info['bbox']
    face_w = bbox[2] - bbox[0]
    face_h = bbox[3] - bbox[1]
    res_dict = fitting_dict['fitting_res']
    recon_model = get_recon_model(model=args.recon_model,
                                  device=device,
                                  batch_size=1,
                                  img_size=args.tar_size)
    keys = list(res_dict.keys())
    keys = sorted(keys)

    video = cv2.VideoWriter(args.out_video_path, cv2.VideoWriter_fourcc(
        *'XVID'), video_info['fps'], (video_info['frame_w'], video_info['frame_h']))
    for k in tqdm(keys):
        orig_frame = cv2.imread(os.path.join(
            args.tmp_frame_folder, str(k)+'.png'))
        cur_rot_tensor = torch.tensor(res_dict[k]['rot'], device=device)
        cur_trans_tensor = torch.tensor(res_dict[k]['trans'], device=device)
        cur_gamma_tensor = torch.tensor(res_dict[k]['gamma'], device=device)
        cur_exp_tensor = torch.tensor(res_dict[k]['exp'], device=device)
        pred_dict = recon_model(recon_model.merge_coeffs(
            id_tensor, cur_exp_tensor, tex_tensor,
            cur_rot_tensor, cur_gamma_tensor, cur_trans_tensor), render=True)

        rendered_img = pred_dict['rendered_img']
        rendered_img = rendered_img.cpu().numpy().squeeze()
        out_img = rendered_img[:, :, :3].astype(np.uint8)
        out_mask = (rendered_img[:, :, 3] > 0).astype(np.uint8)
        resized_out_img = cv2.resize(out_img, (face_w, face_h))[:, :, ::-1]
        resized_mask = cv2.resize(
            out_mask, (face_w, face_h), cv2.INTER_NEAREST)[..., None]

        composed_face = orig_frame[bbox[1]:bbox[3], bbox[0]:bbox[2], :] * \
            (1 - resized_mask) + resized_out_img * resized_mask
        orig_frame[bbox[1]:bbox[3], bbox[0]:bbox[2], :] = composed_face

        video.write(orig_frame)
    video.release()
    print('video saved in %s' % args.out_video_path)


def fit_shape(args, device):
    recon_model = get_recon_model(model=args.recon_model,
                                  device=device,
                                  batch_size=args.nframes_shape,
                                  img_size=args.tar_size)
    fitting_dataset = FittingDataset(
        args.tmp_face_folder, args.lm_pkl_path)
    fitting_data_loader = torch.utils.data.DataLoader(dataset=fitting_dataset,
                                                      batch_size=args.nframes_shape,
                                                      shuffle=True,
                                                      num_workers=1,
                                                      drop_last=False)
    for cur_batch in fitting_data_loader:
        lms, img, img_keys = cur_batch
        lms = lms.to(device)
        imgs = img.to(device)
        break
    lm_weights = utils.get_lm_weights(device)
    print('start rigid fitting')
    rigid_optimizer = torch.optim.Adam([recon_model.get_rot_tensor(),
                                        recon_model.get_trans_tensor()],
                                       lr=args.rf_lr)
    for i in tqdm(range(args.first_rf_iters)):
        rigid_optimizer.zero_grad()
        pred_dict = recon_model(recon_model.get_packed_tensors(), render=False)
        lm_loss_val = losses.lm_loss(
            pred_dict['lms_proj'], lms, lm_weights, img_size=args.tar_size)
        total_loss = args.lm_loss_w * lm_loss_val
        total_loss.backward()
        rigid_optimizer.step()
    print('done rigid fitting. lm_loss: %f' %
          lm_loss_val.detach().cpu().numpy())

    print('start non-rigid fitting')
    nonrigid_optimizer = torch.optim.Adam(
        [recon_model.get_id_tensor(), recon_model.get_exp_tensor(),
         recon_model.get_gamma_tensor(), recon_model.get_tex_tensor(),
         recon_model.get_rot_tensor(), recon_model.get_trans_tensor()],
        lr=args.nrf_lr)
    for i in tqdm(range(args.first_nrf_iters)):
        nonrigid_optimizer.zero_grad()

        pred_dict = recon_model(recon_model.get_packed_tensors(), render=True)
        rendered_img = pred_dict['rendered_img']
        lms_proj = pred_dict['lms_proj']
        face_texture = pred_dict['face_texture']

        mask = rendered_img[:, :, :, 3].detach()

        photo_loss_val = losses.photo_loss(
            rendered_img[:, :, :, :3], imgs, mask > 0)

        lm_loss_val = losses.lm_loss(lms_proj, lms, lm_weights,
                                     img_size=args.tar_size)
        id_reg_loss = losses.get_l2(recon_model.get_id_tensor())
        exp_reg_loss = losses.get_l2(recon_model.get_exp_tensor())
        tex_reg_loss = losses.get_l2(recon_model.get_tex_tensor())
        tex_loss_val = losses.reflectance_loss(
            face_texture, recon_model.get_skinmask())

        loss = lm_loss_val*args.lm_loss_w + \
            id_reg_loss*args.id_reg_w + \
            exp_reg_loss*args.exp_reg_w + \
            tex_reg_loss*args.tex_reg_w + \
            tex_loss_val*args.tex_w + \
            photo_loss_val*args.rgb_loss_w

        loss.backward()
        nonrigid_optimizer.step()

    loss_str = ''
    loss_str += 'lm_loss: %f\t' % lm_loss_val.detach().cpu().numpy()
    loss_str += 'photo_loss: %f\t' % photo_loss_val.detach().cpu().numpy()
    loss_str += 'tex_loss: %f\t' % tex_loss_val.detach().cpu().numpy()
    loss_str += 'id_reg_loss: %f\t' % id_reg_loss.detach().cpu().numpy()
    loss_str += 'exp_reg_loss: %f\t' % exp_reg_loss.detach().cpu().numpy()
    loss_str += 'tex_reg_loss: %f\t' % tex_reg_loss.detach().cpu().numpy()
    print('done non rigid fitting.', loss_str)

    np.save(args.id_npy_path, recon_model.get_id_tensor(
    ).detach().cpu().numpy().reshape(1, -1))
    np.save(args.tex_npy_path, recon_model.get_tex_tensor(
    ).detach().cpu().numpy().reshape(1, -1))


def process_video(args, device):
    mtcnn = MTCNN(device=device, select_largest=False)

    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._3D, flip_input=True, device=device)

    frame_ind = 0

    cap = cv2.VideoCapture(args.v_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = cap.read()
    if ret is False:
        print('error reading the video file %s' % args.v_path)
        return
    orig_h, orig_w = frame.shape[:2]
    bboxes, probs = mtcnn.detect(frame)

    if bboxes is None:
        print('no face detected')
    else:
        bbox = utils.pad_bbox(bboxes[0], (orig_w, orig_h), args.padding_ratio)
        face_w = bbox[2] - bbox[0]
        face_h = bbox[3] - bbox[1]
        assert face_w == face_h
    print('A face is detected. l: %d, t: %d, r: %d, b: %d'
          % (bbox[0], bbox[1], bbox[2], bbox[3]))

    lm_dict = {}
    while(cap.isOpened()):
        face_img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        resized_face_img = cv2.resize(face_img, (args.tar_size, args.tar_size))
        lms = fa.get_landmarks_from_image(resized_face_img)[0]
        lms = lms[:, :2]
        lm_dict[frame_ind] = lms

        face_out_path = os.path.join(
            args.tmp_face_folder, '%d.png' % frame_ind)
        frame_out_path = os.path.join(
            args.tmp_frame_folder, '%d.png' % frame_ind)
        cv2.imwrite(face_out_path, resized_face_img)
        cv2.imwrite(frame_out_path, frame)
        if frame_ind % 100 == 0:
            print('processing %d/%d' % (frame_ind, num_frames))
        frame_ind += 1
        ret, frame = cap.read()
        if ret is False:
            break
    with open(args.lm_pkl_path, 'wb') as f:
        pickle.dump(lm_dict, f)
    print('done processing the video. Got %d frames' % frame_ind)
    cap.release()
    v_inf_dict = {'fps': fps, 'frame_w': orig_w,
                  'frame_h': orig_h, 'bbox': bbox}
    with open(args.v_info_path, 'wb') as f:
        pickle.dump(v_inf_dict, f)


def merge_dict(args):

    final_dict = {}
    for i in range(args.nworkers):
        with open(os.path.join(args.cache_folder, '%d_fitting.pkl' % i), 'rb') as f:
            tmp_dict = pickle.load(f)
        if i == 0:
            final_dict['id'] = tmp_dict['id']
            final_dict['tex'] = tmp_dict['tex']
            final_dict['fitting_res'] = {}
        final_dict['fitting_res'].update(tmp_dict['fitting_res'])
    with open(args.fitting_pkl_path, 'wb') as f:
        pickle.dump(final_dict, f)


if __name__ == '__main__':
    args = VideoFittingOptions()
    args = args.parse()
    args.devices = ['cuda:%d' % i for i in range(args.ngpus)]
    # to avoid mult-processing runtime error
    set_start_method('spawn')

    # remove cache files and create new folders
    if os.path.exists(args.cache_folder):
        shutil.rmtree(args.cache_folder)

    args.tmp_face_folder = os.path.join(args.cache_folder, 'faces')
    args.tmp_frame_folder = os.path.join(args.cache_folder, 'frames')
    args.lm_pkl_path = os.path.join(args.cache_folder, 'lms.pkl')
    args.id_npy_path = os.path.join(args.cache_folder, 'id.npy')
    args.tex_npy_path = os.path.join(args.cache_folder, 'tex.npy')
    args.v_info_path = os.path.join(args.cache_folder, 'v_info.pkl')
    args.fitting_pkl_path = os.path.join(
        args.res_folder, os.path.basename(args.v_path)[:-4]+'_fitting_res.pkl')
    args.out_video_path = os.path.join(
        args.res_folder, os.path.basename(args.v_path)[:-4]+'_recon_video.avi')

    utils.mymkdirs(args.cache_folder)
    utils.mymkdirs(args.tmp_face_folder)
    utils.mymkdirs(args.res_folder)
    utils.mymkdirs(args.tmp_frame_folder)

    # extract frames and faces and get landmarks
    process_video(args, args.devices[0])

    fit_shape(args, args.devices[0])

    # fit frames using Process
    processes = []
    for i in range(args.nworkers):
        p = Process(target=fit_coeffs, args=(
            args, args.devices[i % args.ngpus], i))
        p.start()
        processes.append(p)

    for cur_p in processes:
        cur_p.join()
    # merge dict
    merge_dict(args)

    gen_composed_video(args, args.devices[0])
