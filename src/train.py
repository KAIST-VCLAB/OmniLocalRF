# Copyright (c) Dongyoung Choi
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2022 Anpei Chen

import os
import warnings

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)
import json
import sys
import time

from torch.utils.tensorboard import SummaryWriter

sys.path.append("code")
from dataLoader.omnilocalrf_dataset import OmniLocalRFDataset
from omnilocal_rfs import OmniLocalRFs, ids2pixel
from opt import config_parser
from renderer import render
from utils.utils import (get_cam2cams, get_fwd_bwd_cam2cams, get_fwd_bwd_cam2cams_modified, smooth_poses_spline)
from utils.utils import (N_to_reso, TVLoss, TVLossPrev, draw_poses, get_pred_flow, get_pred_flow_360,
                         proj_360, compute_depth_loss, inverse_pose)
from utils.ray_utils import (get_ray_directions_360, get_rays_lean, patch_sampling)


def save_transforms(poses_mtx, transform_path, omnilocal_rfs, train_dataset=None):
    if train_dataset is not None:
        fnames = train_dataset.all_image_paths
    else:
        fnames = [f"{i:06d}.jpg" for i in range(len(poses_mtx))]

    fl = omnilocal_rfs.focal(omnilocal_rfs.W).item()
    transforms = {
        "fl_x": fl,
        "fl_y": fl,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "cx": omnilocal_rfs.W/2,
        "cy": omnilocal_rfs.H/2,
        "w": omnilocal_rfs.W,
        "h": omnilocal_rfs.H,
        "frames": [],
    }
    for pose_mtx, fname in zip(poses_mtx, fnames):
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :] = pose_mtx
        frame_data = {
            "file_path": f"images/{fname}",
            "transform_matrix": pose.tolist(),
        }
        transforms["frames"].append(frame_data)

    with open(transform_path, "w") as outfile:
        json.dump(transforms, outfile, indent=2)


@torch.no_grad()
def render_frames(
    args, poses_mtx, omnilocal_rfs, logfolder, test_dataset, train_dataset
):
    save_transforms(poses_mtx.cpu(), f"{logfolder}/transforms.json", omnilocal_rfs, train_dataset)
    t_w2rf = torch.stack(list(omnilocal_rfs.world2rf), dim=0).detach().cpu()
    RF_mtx_inv = torch.cat([torch.stack(len(t_w2rf) * [torch.eye(3)]), t_w2rf.clone()[..., None]], axis=-1)
    save_transforms(RF_mtx_inv.cpu(), f"{logfolder}/transforms_rf.json", omnilocal_rfs)
    
    W, H = train_dataset.img_wh

    if args.render_test:
        render(
            test_dataset,
            poses_mtx,
            omnilocal_rfs,
            args,
            W=W, H=H,
            savePath=f"{logfolder}/test",
            save_frames=True,
            test=True,
            train_dataset=train_dataset,
            img_format="png",
            start=0
        )
    
    with torch.no_grad():
        # Neural feature grid        
        os.makedirs(f"{logfolder}/dynamic", exist_ok=True)   
        os.makedirs(f"{logfolder}/dynamic/alpha", exist_ok=True)
        os.makedirs(f"{logfolder}/dynamic/rgb", exist_ok=True)
        os.makedirs(f"{logfolder}/dynamic/full", exist_ok=True)
        
        i = torch.linspace(-1,1,W, device=args.device)
        j = torch.linspace(-1,1,H, device=args.device)
        for f_num in tqdm(range(train_dataset.num_images)):
            f = 2 * f_num / (train_dataset.num_images-1) -1                
            f = torch.tensor(f,device=args.device)
            ijf = torch.stack(torch.meshgrid(i,j,f,indexing ='ij')).view(3,-1).permute([1, 0])

            for level in range(omnilocal_rfs.dyn_multi_res):                    
                if level == 0:
                    dyn_feat = torch.nn.functional.grid_sample(omnilocal_rfs.dyn_fields[level][None, ...], ijf[None, :, None, None,...],
                                                    mode='bilinear', padding_mode='border', align_corners=True)
                    dyn_feat = dyn_feat[0,:,:,0,0].permute([1, 0])
                else:
                    dyn_feat = torch.cat([dyn_feat, 
                                        torch.nn.functional.grid_sample(omnilocal_rfs.dyn_fields[level][None, ...], ijf[None, :, None, None,...],
                                        mode='bilinear', padding_mode='border', align_corners=True)[0,:,:,0,0].permute([1, 0])],
                                        dim = -1
                    )
            dyn = omnilocal_rfs.dyn_mlp(dyn_feat).permute([1,0]) ## [4, -1]
            dyn_img = dyn.reshape(-1, W, H).permute([2,1,0])
            dyn_img = (255 * dyn_img).cpu().byte().numpy()
            name = train_dataset.all_image_paths[f_num].split('.')[0]
            cv2.imwrite(f"{logfolder}/dynamic/alpha/{name}.png", dyn_img[:,:,3])
            cv2.imwrite(f"{logfolder}/dynamic/rgb/{name}.png", cv2.cvtColor(dyn_img[:,:,:3], cv2.COLOR_RGB2BGR))
            cv2.imwrite(f"{logfolder}/dynamic/full/{name}.png", cv2.cvtColor(dyn_img, cv2.COLOR_RGB2BGR))
    
    if args.render_path:
        c2ws = smooth_poses_spline(poses_mtx, median_prefilter=True)
        os.makedirs(f"{logfolder}/smooth_spline", exist_ok=True)
        save_transforms(c2ws.cpu(), f"{logfolder}/smooth_spline/transforms.json", omnilocal_rfs)
        render(
            test_dataset,
            c2ws,
            omnilocal_rfs,
            args,
            W=W, H=H,
            savePath=f"{logfolder}/smooth_spline",
            train_dataset=train_dataset,
            img_format="jpg",
            save_frames=True,
            save_video=True,
            floater_thresh=0.8,
        )

    if args.render_from_file != "":
        with open(args.render_from_file, 'r') as f:
            transforms = json.load(f)
        c2ws = [transform["transform_matrix"] for transform in transforms["frames"]]
        c2ws = torch.tensor(c2ws).to(args.device)
        c2ws = c2ws[..., :3, :]
        save_path = f"{logfolder}/{os.path.splitext(os.path.basename(args.render_from_file))[0]}"
        os.makedirs(save_path, exist_ok=True)
        render(
            test_dataset,
            c2ws,
            omnilocal_rfs,
            args,
            W=W, H=H,
            savePath=save_path,
            train_dataset=train_dataset,
            img_format="jpg",
            save_frames=True,
            save_video=True,
            floater_thresh=0.5,
        )


@torch.no_grad()
def render_test(args):
    # init dataset
    train_dataset = OmniLocalRFDataset(
        f"{args.datadir}",
        split="train",
        downsampling=args.downsampling,
        test_frame_every=args.test_frame_every,
        n_init_frames=args.n_init_frames,
        with_preprocessed_poses=args.with_preprocessed_poses,
        subsequence=args.subsequence,
        frame_step=args.frame_step,
    )
    test_dataset = OmniLocalRFDataset(
        f"{args.datadir}",
        split="test",
        load_depth=args.loss_depth_weight_inital > 0,
        load_flow=args.loss_flow_weight_inital > 0,
        downsampling=args.downsampling,
        test_frame_every=args.test_frame_every,
        with_preprocessed_poses=args.with_preprocessed_poses,
        subsequence=args.subsequence,
        frame_step=args.frame_step,
    )

    if args.ckpt is None:
        logfolder = f"{args.logdir}"
        ckpt_path = f"{logfolder}/checkpoints.th"
    else:
        ckpt_path = args.ckpt

    if not os.path.isfile(ckpt_path):
        print("Backing up to intermediate checkpoints")
        ckpt_path = f"{logfolder}/checkpoints_tmp.th"
        if not os.path.isfile(ckpt_path):
            print("the ckpt path does not exists!!")
            return  

    with open(ckpt_path, "rb") as f:
        ckpt = torch.load(f, map_location=args.device)
        print("Success in Loading Checkpoints")
    kwargs = ckpt["kwargs"]
    if args.with_preprocessed_poses:
        kwargs["camera_prior"] = {
            "rel_poses": torch.from_numpy(train_dataset.rel_poses).to(args.device),
        }
    else:
        kwargs["camera_prior"] = None
    kwargs["device"] = args.device
    omnilocal_rfs = OmniLocalRFs(**kwargs)
    omnilocal_rfs.load(ckpt["state_dict"])
    omnilocal_rfs = omnilocal_rfs.to(args.device)

    logfolder = os.path.dirname(ckpt_path)
    render_frames(
        args,
        omnilocal_rfs.get_cam2world(),
        omnilocal_rfs,
        logfolder,
        test_dataset=test_dataset,
        train_dataset=train_dataset
    )


def reconstruction(args):
    # Apply speedup factors
    args.n_iters_per_frame = int(args.n_iters_per_frame / args.refinement_speedup_factor)
    args.n_iters_reg = int(args.n_iters_reg / args.refinement_speedup_factor)
    args.upsamp_list = [int(upsamp / args.refinement_speedup_factor) for upsamp in args.upsamp_list]
    args.update_AlphaMask_list = [int(update_AlphaMask / args.refinement_speedup_factor) 
                                  for update_AlphaMask in args.update_AlphaMask_list]
    
    args.add_frames_every = int(args.add_frames_every / args.prog_speedup_factor)
    args.lr_R_init = args.lr_R_init * args.prog_speedup_factor
    args.lr_t_init = args.lr_t_init * args.prog_speedup_factor
    args.loss_flow_weight_inital = args.loss_flow_weight_inital * args.prog_speedup_factor
    args.L1_weight = args.L1_weight * args.prog_speedup_factor
    args.TV_weight_density = args.TV_weight_density * args.prog_speedup_factor
    args.TV_weight_app = args.TV_weight_app * args.prog_speedup_factor
    
    # Init dataset
    train_dataset = OmniLocalRFDataset(
        f"{args.datadir}",
        split="train",
        downsampling=args.downsampling,
        test_frame_every=args.test_frame_every,
        load_depth=args.loss_depth_weight_inital > 0,
        load_flow=args.loss_flow_weight_inital > 0,
        with_preprocessed_poses=args.with_preprocessed_poses,
        n_init_frames=args.n_init_frames,
        subsequence=args.subsequence,
        frame_step=args.frame_step,
    )
    test_dataset = OmniLocalRFDataset(
        f"{args.datadir}",
        split="test",
        load_depth=args.loss_depth_weight_inital > 0,
        load_flow=args.loss_flow_weight_inital > 0,
        downsampling=args.downsampling,
        test_frame_every=args.test_frame_every,
        with_preprocessed_poses=args.with_preprocessed_poses,
        subsequence=args.subsequence,
        frame_step=args.frame_step,
    )
    near_far = train_dataset.near_far

    # Init resolution
    upsamp_list = args.upsamp_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh

    logfolder = f"{args.logdir}"

    # Init log file
    os.makedirs(logfolder, exist_ok=True)
    writer = SummaryWriter(log_dir=logfolder)

    # Init parameters
    aabb = train_dataset.scene_bbox.to(args.device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)

    # TODO: Add midpoint loading
    if args.ckpt is not None:        
        with open(args.ckpt, "rb") as f:
            ckpt = torch.load(f, map_location=args.device)
            kwargs = ckpt["kwargs"]
            if args.with_preprocessed_poses:
                kwargs["camera_prior"] = {
                    "rel_poses": torch.from_numpy(train_dataset.rel_poses).to(args.device),
                }
            else:
                kwargs["camera_prior"] = None
            kwargs["device"] = args.device
            omnilocal_rfs = OmniLocalRFs(**kwargs)
            omnilocal_rfs.load(ckpt["state_dict"])
            omnilocal_rfs = omnilocal_rfs.to(args.device)
            kwargs = ckpt["kwargs"]
            kwargs.update({"device": args.device})


    # Linear in logrithmic space
    N_voxel_list = (
        torch.round(
            torch.exp(
                torch.linspace(
                    np.log(args.N_voxel_init),
                    np.log(args.N_voxel_final),
                    len(upsamp_list) + 1,
                )
            )
        ).long()
    ).tolist()[1:]
    N_voxel_list = {
        usamp_idx: round(N_voxel**(1/3))**3 for usamp_idx, N_voxel in zip(upsamp_list, N_voxel_list)
    }

    if args.with_preprocessed_poses:
        camera_prior = {
            "rel_poses": torch.from_numpy(train_dataset.rel_poses).to(args.device),
        }
    else:
        camera_prior = None

    print("train_dataset.num_images",train_dataset.num_images)
    print("train_dataset.loaded_frames",train_dataset.loaded_frames)
    omnilocal_rfs = OmniLocalRFs(
        camera_prior=camera_prior,
        fov=args.fov,
        n_init_frames=min(args.n_init_frames, train_dataset.num_images),
        n_frames=train_dataset.num_images,
        n_overlap=args.n_overlap,
        WH=train_dataset.img_wh,
        n_iters_per_frame=args.n_iters_per_frame,
        n_iters_reg=args.n_iters_reg,
        lr_R_init=args.lr_R_init,
        lr_t_init=args.lr_t_init,
        lr_i_init=args.lr_i_init,
        lr_exposure_init=args.lr_exposure_init,
        lr_dyn=args.lr_dyn,
        lr_dyn_mlp=args.lr_dyn_mlp,
        rf_lr_init=args.lr_init,
        rf_lr_basis=args.lr_basis,
        lr_decay_target_ratio=args.lr_decay_target_ratio,
        N_voxel_list=N_voxel_list,
        fin_alpha_block=args.fin_alpha_block>0,
        update_AlphaMask_list=args.update_AlphaMask_list,
        lr_upsample_reset=args.lr_upsample_reset,
        device=args.device,
        alphaMask_thres=args.alpha_mask_thre,
        shadingMode=args.shadingMode,
        aabb=aabb,
        gridSize=reso_cur,
        density_n_comp=n_lamb_sigma,
        appearance_n_comp=n_lamb_sh,
        app_dim=args.data_dim_color,
        near_far=near_far,
        density_shift=args.density_shift,
        distance_scale=args.distance_scale,
        rayMarch_weight_thres=args.rm_weight_mask_thre,
        pos_pe=args.pos_pe,
        view_pe=args.view_pe,
        fea_pe=args.fea_pe,
        featureC=args.featureC,
        step_ratio=args.step_ratio,
        fea2denseAct=args.fea2denseAct,
    )
    omnilocal_rfs = omnilocal_rfs.to(args.device)

    torch.cuda.empty_cache()

    tvreg = TVLoss()
    tvreg_prev = TVLoss()
    W, H = train_dataset.img_wh
    
    training = True
    n_added_frames = 0
    last_add_iter = 0
    iteration = 0
    iter_aft_append_rf = 0
    metrics = {}
    start_time = time.time()
    bidirectional_optimization = (args.bidirectional_optimization == 1)
    omnilocal_rfs.dyn_block_iter = args.dyn_block_iter

    while training:
        optimize_poses = args.lr_R_init > 0 or args.lr_t_init > 0

        data_blob = train_dataset.sample(args.batch_size, omnilocal_rfs.is_refining, optimize_poses)
        view_ids = torch.from_numpy(data_blob["view_ids"]).to(args.device)
        rgb_train = torch.from_numpy(data_blob["rgbs"]).to(args.device)
        loss_weights = torch.from_numpy(data_blob["loss_weights"]).to(args.device)
        train_test_poses = data_blob["train_test_poses"]
        ray_idx = torch.from_numpy(data_blob["idx"]).to(args.device)
        reg_loss_weight = omnilocal_rfs.lr_factor ** (omnilocal_rfs.rf_iter[-1])

        rgb_map, static_rgb_map, depth_map, directions, ij, dyn_map, T_fin = omnilocal_rfs(
            ray_idx,
            view_ids,
            W,
            H,
            is_train=True,
            test_id=train_test_poses,
        )       

        ## Record iteration after appeind rf for lr_dyn_rf_factor
        omnilocal_rfs.iter_aft_append_rf += 1
        if len(omnilocal_rfs.tensorfs) < 2:
            if omnilocal_rfs.iter_aft_append_rf > omnilocal_rfs.dyn_block_iter:
                omnilocal_rfs.activate_dyn_fields()
        else:            
            if omnilocal_rfs.iter_aft_append_rf > 2 * omnilocal_rfs.dyn_block_iter:
                omnilocal_rfs.activate_dyn_fields()
        
        total_loss = 0
        ## Dynamic RGB photometric loss
        loss_dyn_weight = args.loss_dyn_weight        
        dyn_noise = 0.1 * torch.rand_like(rgb_train, device=args.device) - 0.05
        loss_main_mask_rgb = loss_dyn_weight * ((torch.abs((dyn_map[:,:3] - (rgb_train + dyn_noise).clamp(0,1))) * loss_weights) / loss_weights.mean())
        loss_main_mask_rgb = loss_main_mask_rgb.mean()
        total_loss += loss_main_mask_rgb
        writer.add_scalar("train/dyn_loss", loss_main_mask_rgb, global_step=iteration)

        ## Rendered Depth TV
        TV_row = (torch.abs(depth_map[0::4] - depth_map[1::4]) + torch.abs(depth_map[2::4] - depth_map[3::4]))[depth_map[0::4] > 30].mean()
        TV_col = (torch.abs(depth_map[0::4] - depth_map[2::4]) + torch.abs(depth_map[1::4] - depth_map[3::4]))[depth_map[0::4] > 30].mean()
        TV_depth = TV_row + TV_col
        TV_depth_weight = 1e-4
        total_loss += TV_depth_weight * TV_depth
        writer.add_scalar("train/TV_depth", TV_depth, global_step=iteration)  

        # Forward RGB at Source Frame
        loss_for_rgb_src = 0.25 * ((torch.abs(rgb_map - rgb_train) * loss_weights) / loss_weights.mean())
        loss_for_rgb_src = loss_for_rgb_src.mean()
        total_loss += loss_for_rgb_src
        writer.add_scalar("train/loss_for_rgb_src", loss_for_rgb_src, global_step=iteration)
        
        '''
        Bidirectional Optimization
        '''
        if len(omnilocal_rfs.tensorfs) > 1 and bidirectional_optimization and omnilocal_rfs.rf_iter[-1] > max(list(omnilocal_rfs.N_voxel_list.keys())):
        # if len(omnilocal_rfs.tensorfs) > 1 and bidirectional_optimization:
            '''
            Forward Refinement by distant frames
            '''         
            refine_cam2world = omnilocal_rfs.get_cam2world(starting_id=0)
            frame_src = view_ids.clone() # Absolute index                
            curr_rf_ids = len(omnilocal_rfs.tensorfs) - 1 ## Index of refining RF (Current RF)     
            prev_rf_ids = torch.randint(0, curr_rf_ids, size=(1,1), dtype=torch.int64, device=view_ids.device)[0,0]   ## Select previous radiance fields            
            dst_idx = torch.linspace(train_dataset.frame_list[prev_rf_ids][0], train_dataset.frame_list[prev_rf_ids][1], 
                                    train_dataset.frame_list[prev_rf_ids][1] - train_dataset.frame_list[prev_rf_ids][0] + 1,
                                    dtype=torch.int64, device=view_ids.device)               
            dst_idx = dst_idx[dst_idx % args.test_frame_every != 0]
            tmp_idx = torch.randint(0, len(dst_idx), size=(1,view_ids.shape[0]), dtype=torch.int64, device=view_ids.device)[0]
            frame_dst = dst_idx[tmp_idx]  

            dst_rgb_map, dst_rgbs, dst_static_rgb_map, dst_depth_map, dst_dyn_map, valid_for_rays = omnilocal_rfs.forward_step(
                ray_ids=ray_idx,
                view_ids=view_ids,
                directions=directions,
                depth_map=depth_map,
                refine_cam2world=refine_cam2world,
                curr_rf_ids=curr_rf_ids,
                prev_rf_ids=prev_rf_ids,
                frame_src=frame_src,
                frame_dst=frame_dst,
                W=W,
                H=H,
                train_dataset=train_dataset,
                forward_depth_margin=2,
                forward_depth_thres=0.05
            )
            
            loss_forward = 0
            loss_for_rgb_dst = 0.025 * ((torch.abs((dst_rgb_map - dst_rgbs)) * loss_weights.view(-1)[...,None]) 
                                * (1-dst_dyn_map[:,3:].clone().detach()) ## Exclude masked rays of src frame
                            / loss_weights.mean()) ## Compute loss only over the valid rays
            loss_for_rgb = loss_for_rgb_dst[valid_for_rays].mean()
            loss_forward += loss_for_rgb
            writer.add_scalar("train/loss_for_rgb", loss_for_rgb, global_step=iteration)          
            
            original = len(loss_for_rgb_dst)
            after = len(loss_for_rgb_dst[valid_for_rays])
            if iteration % 200 == 0:
                print("forward refining ratio: ", after/original)
            
            total_loss += loss_forward 

            '''
            Backward Refinement by Distant Frames
            '''
            # Backward ray ids for reprojection
            back_src_ray_ids = patch_sampling(W, H, args.batch_size, args.device).reshape(len(frame_dst), -1)
            back_src_ray_ids = back_src_ray_ids + frame_dst[..., None] * (H*W)                
            back_src_ray_ids = back_src_ray_ids.reshape(-1)
            i, j = ids2pixel(W, H, back_src_ray_ids)
            back_directions_src = get_ray_directions_360(i, j, W, H) 

            back_prev_src_rgb_map, back_prev_src_static_rgb_map, back_prev_src_depth_map, _, _, back_prev_src_dyn_map, back_prev_src_T = omnilocal_rfs(
                back_src_ray_ids,
                frame_dst,
                W,
                H,
                is_train=True,
                test_id=train_test_poses,
                prev_rf=True,
                prev_rf_ids=prev_rf_ids
            )

            backward_depth_margin = 0.05
            backward_depth_thres = 1
            back_prev_dst_rgb_map, back_dst_rgbs, back_prev_dst_static_rgb_map, back_prev_dst_depth_map, back_prev_dst_dyn_map, back_prev_T, valid_back_rays, back_dst_loss_weights = omnilocal_rfs.backward_step(
                directions=back_directions_src,
                depth_map=back_prev_src_depth_map,
                refine_cam2world=refine_cam2world,
                curr_rf_ids=curr_rf_ids,
                prev_rf_ids=prev_rf_ids,
                frame_src=frame_src,
                frame_dst=frame_dst,
                W=W,
                H=H,
                train_dataset=train_dataset,
                backward_depth_margin=backward_depth_margin,
                backward_depth_thres=backward_depth_thres,
            )         

            
            ''' 
            Backward refining previous RGB 
            '''
            loss_backward = 0            
            data_blob_back_src = train_dataset[back_src_ray_ids.int().cpu().numpy()]
            back_src_rgbs = torch.from_numpy(data_blob_back_src["rgbs"]).to(args.device)  ## Get gt of dst frames 00
            back_src_loss_weights = torch.from_numpy(data_blob_back_src["loss_weights"]).to(args.device)  ## Get gt of dst frames 00
            loss_back_rgb_src = 0.025 * ((torch.abs((back_prev_src_rgb_map - back_src_rgbs)) * back_src_loss_weights.view(-1)[...,None]) 
                                * (1-back_prev_src_dyn_map[:,3:].clone().detach()) ## Exclude masked rays of src frame
                            / back_src_loss_weights.mean()) ## Compute loss only over the valid rays
                            
            loss_back_rgb_src = loss_back_rgb_src.mean()
            loss_backward += loss_back_rgb_src
            writer.add_scalar("train/loss_back_rgb_src", loss_back_rgb_src, global_step=iteration)    
            
            ## Beta distribution loss over tranmittance of the last samples
            beta_weight = 0.1 * args.loss_beta_dist
            loss_back_beta = beta_weight * (torch.log(back_prev_src_T.clamp(min=1e-6)) + torch.log((1-back_prev_src_T).clamp(min=1e-6))).mean()
            loss_backward += loss_back_beta
            writer.add_scalar("train/loss_back_beta", loss_back_beta, global_step=iteration)

            '''
            Backward refining RGB
            '''                                        
            loss_back_rgb_dst = 0.025 * ((torch.abs((back_prev_dst_rgb_map - back_dst_rgbs)) * back_dst_loss_weights.view(-1)[...,None]) 
                                * (1-back_prev_dst_dyn_map[:,3:].clone().detach()) ## Exclude masked rays of src frame
                            / back_dst_loss_weights.mean()) ## Compute loss only over the valid rays
                            
            loss_back_rgb_dst = loss_back_rgb_dst[valid_back_rays].mean()
            loss_backward += loss_back_rgb_dst
            writer.add_scalar("train/loss_back_rgb_dst", loss_back_rgb_dst, global_step=iteration)            
            

            '''
            TV depth of src during backward step
            '''
            back_prev_src_depth_map = back_prev_src_depth_map.view(-1)     
            back_TV_row = (torch.abs(back_prev_src_depth_map[0::4] - back_prev_src_depth_map[1::4]) + torch.abs(back_prev_src_depth_map[2::4] - back_prev_src_depth_map[3::4]))[back_prev_src_depth_map[0::4] > 30].mean()
            back_TV_col = (torch.abs(back_prev_src_depth_map[0::4] - back_prev_src_depth_map[2::4]) + torch.abs(back_prev_src_depth_map[1::4] - back_prev_src_depth_map[3::4]))[back_prev_src_depth_map[0::4] > 30].mean()
            back_TV_depth = back_TV_row + back_TV_col
            back_TV_depth_weight = 1e-5
            loss_backward += back_TV_depth_weight * back_TV_depth
            writer.add_scalar("train/back_TV_depth", back_TV_depth, global_step=iteration) 

            # Dynamic RGB supervision on dst frame while backward refinement
            dyn_noise = 0.1 * torch.rand_like(back_dst_rgbs, device=args.device) - 0.05
            loss_back_mask_rgb = loss_dyn_weight * ((torch.abs((back_prev_dst_dyn_map[:,:3] - (back_dst_rgbs + dyn_noise).clamp(0,1))) * back_dst_loss_weights.view(-1)[...,None]) / back_dst_loss_weights.mean())
            loss_back_mask_rgb = loss_back_mask_rgb.mean()
            loss_backward += loss_back_mask_rgb
            writer.add_scalar("train/loss_back_mask_rgb", loss_back_mask_rgb, global_step=iteration)

            if  omnilocal_rfs.regularize:
                loss_prev_tv, prev_l1_loss = omnilocal_rfs.get_prev_reg_loss(tvreg_prev, args.TV_weight_density, args.TV_weight_app, args.L1_weight, rf_ids=prev_rf_ids)
                loss_backward = loss_backward + 0.1*loss_prev_tv + 0.1*prev_l1_loss
                writer.add_scalar("train/loss_prev_tv", loss_prev_tv, global_step=iteration)
                writer.add_scalar("train/prev_l1_loss", prev_l1_loss, global_step=iteration)

            total_loss += loss_backward
    
        # Get rendered rays schedule
        if omnilocal_rfs.regularize and args.loss_flow_weight_inital > 0 or args.loss_depth_weight_inital > 0:
            depth_map = depth_map.view(view_ids.shape[0], -1)
            loss_weights = loss_weights.view(view_ids.shape[0], -1)
            writer.add_scalar("train/reg_loss_weights", reg_loss_weight, global_step=iteration)

        # Optical flow            
        if omnilocal_rfs.regularize and args.loss_flow_weight_inital > 0:
            fwd_flow = torch.from_numpy(data_blob["fwd_flow"]).to(args.device)
            fwd_mask = torch.from_numpy(data_blob["fwd_mask"]).to(args.device)
            bwd_flow = torch.from_numpy(data_blob["bwd_flow"]).to(args.device)
            bwd_mask = torch.from_numpy(data_blob["bwd_mask"]).to(args.device)

            starting_frame_id = max(train_dataset.active_frames_bounds[0] - 1, 0)
            cam2world = omnilocal_rfs.get_cam2world(starting_id=starting_frame_id)            
            fwd_flow = fwd_flow.view(view_ids.shape[0], -1, 2)
            fwd_mask = fwd_mask.view(view_ids.shape[0], -1)
            fwd_mask[view_ids == len(cam2world) - 1] = 0
            bwd_flow = bwd_flow.view(view_ids.shape[0], -1, 2)
            bwd_mask = bwd_mask.view(view_ids.shape[0], -1)

            ij = ij.view(view_ids.shape[0], -1, 2)
            directions = directions.view(view_ids.shape[0], -1, 3)
            fwd_cam2cams, bwd_cam2cams = get_fwd_bwd_cam2cams(cam2world, view_ids - starting_frame_id)
            pts = directions * depth_map[..., None]

            if args.fov == 360:
                pred_fwd_flow = get_pred_flow_360(
                    pts, ij, fwd_cam2cams, W, H)
                pred_bwd_flow = get_pred_flow_360(
                    pts, ij, bwd_cam2cams, W, H)

                bwd_flow_diff = torch.abs(pred_bwd_flow - bwd_flow)
                fwd_flow_diff = torch.abs(pred_fwd_flow - fwd_flow)
                flow_loss_arr =  torch.sum(bwd_flow_diff, dim=-1) * bwd_mask * (1-dyn_map[:,3].clone().detach()).view(view_ids.shape[0], -1)
                flow_loss_arr += torch.sum(fwd_flow_diff, dim=-1) * fwd_mask * (1-dyn_map[:,3].clone().detach()).view(view_ids.shape[0], -1)
                flow_loss_arr[flow_loss_arr > torch.quantile(flow_loss_arr, 0.9, dim=1)[..., None]] = 0
            else:
                pred_fwd_flow = get_pred_flow(
                    pts, ij, fwd_cam2cams, omnilocal_rfs.focal(W), omnilocal_rfs.center(W, H))
                pred_bwd_flow = get_pred_flow(
                    pts, ij, bwd_cam2cams, omnilocal_rfs.focal(W), omnilocal_rfs.center(W, H))
                flow_loss_arr =  torch.sum(torch.abs(pred_bwd_flow - bwd_flow), dim=-1) * bwd_mask
                flow_loss_arr += torch.sum(torch.abs(pred_fwd_flow - fwd_flow), dim=-1) * fwd_mask
                flow_loss_arr[flow_loss_arr > torch.quantile(flow_loss_arr, 0.9, dim=1)[..., None]] = 0

            flow_loss = (flow_loss_arr).mean() * args.loss_flow_weight_inital * reg_loss_weight / ((W + H) / 2)
            total_loss += flow_loss
            writer.add_scalar("train/flow_loss", flow_loss, global_step=iteration)               

        # Monocular depth 
        if omnilocal_rfs.regularize and args.loss_depth_weight_inital > 0:
            invdepths = torch.from_numpy(data_blob["invdepths"]).to(args.device)
            invdepths = invdepths.view(view_ids.shape[0], -1)
            _, _, depth_loss_arr = compute_depth_loss(1 / depth_map.clamp(1e-6), invdepths)

            depth_loss_arr[depth_loss_arr > torch.quantile(depth_loss_arr, 0.8, dim=1)[..., None]] = 0
            depth_loss_arr = depth_loss_arr * (1-dyn_map[:,3].clone().detach().reshape(depth_loss_arr.shape[0],depth_loss_arr.shape[1]))
            depth_loss = (depth_loss_arr).mean() * args.loss_depth_weight_inital * reg_loss_weight
            total_loss = total_loss + depth_loss 
            writer.add_scalar("train/depth_loss", depth_loss, global_step=iteration)

        if  omnilocal_rfs.regularize:
            loss_tv, l1_loss = omnilocal_rfs.get_reg_loss(tvreg, args.TV_weight_density, args.TV_weight_app, args.L1_weight)
            total_loss = total_loss + loss_tv + l1_loss
            writer.add_scalar("train/loss_tv", loss_tv, global_step=iteration)
            writer.add_scalar("train/l1_loss", l1_loss, global_step=iteration)

        # Dynanmic total variation        
        dyn_tvx, dyn_tvy, dyn_tvt = omnilocal_rfs.get_dyn_tv(frame_begin=train_dataset.active_frames_bounds[0],
                                                            frame_end=train_dataset.active_frames_bounds[1], 
                                                            num_images=train_dataset.num_images)
        dyn_tv = args.TV_dyn * (dyn_tvx + dyn_tvy + dyn_tvt * 0.1)
        total_loss += dyn_tv        
        writer.add_scalar("train/dyn_tv", dyn_tv, global_step=iteration)

        # Dynamic binary loss
        if omnilocal_rfs.lr_dyn_rf_factor == 1:
            dyn_beta_weight = args.loss_dyn_binary
            loss_main_bin = dyn_beta_weight * (dyn_map[:,3]**2 * (1-dyn_map[:,3])**2).mean()
            total_loss = total_loss + loss_main_bin
            writer.add_scalar("train/loss_main_bin", loss_main_bin, global_step=iteration)        

        # Beta distribution loss over tranmittance of the last samples
        beta_weight = args.loss_beta_dist
        loss_beta_dist = beta_weight * (torch.log(T_fin.clamp(min=1e-6)) + torch.log((1-T_fin).clamp(min=1e-6))).mean()
        total_loss = total_loss + loss_beta_dist
        writer.add_scalar("train/loss_beta_dist", loss_beta_dist, global_step=iteration)        

        # Progressive optimization
        if train_test_poses:
            can_add_rf = False
            if optimize_poses:
                omnilocal_rfs.optimizer_step_poses_only(total_loss)
        else:
            can_add_rf = omnilocal_rfs.optimizer_step(total_loss, optimize_poses)
            training |= train_dataset.active_frames_bounds[1] != train_dataset.num_images

        if not omnilocal_rfs.is_refining:
            should_refine = (not train_dataset.has_left_frames() or (
                n_added_frames > args.n_overlap and (
                    omnilocal_rfs.get_dist_to_last_rf().cpu().item() > args.max_drift
                    or (train_dataset.active_frames_bounds[1] - train_dataset.active_frames_bounds[0]) >= args.n_max_frames
                )))
            if should_refine and (iteration - last_add_iter) >= args.add_frames_every:
                omnilocal_rfs.is_refining = True

            should_add_frame = train_dataset.has_left_frames()
            should_add_frame &= (iteration - last_add_iter + 1) % args.add_frames_every == 0

            should_add_frame &= not should_refine
            should_add_frame &= not omnilocal_rfs.is_refining

            # Add supervising frames
            if should_add_frame:
                omnilocal_rfs.append_frame()
                train_dataset.activate_frames()
                n_added_frames += 1
                last_add_iter = iteration

        # Add new RF
        if can_add_rf :
            if train_dataset.has_left_frames():
                omnilocal_rfs.append_rf(n_added_frames)
                n_added_frames = 0
                last_add_rf_iter = iteration

                # Remove supervising frames
                training_frames = (omnilocal_rfs.blending_weights[:, -1] > 0)
                train_dataset.deactivate_frames(
                    np.argmax(training_frames.cpu().numpy(), axis=0))
            else:
                training = False
                
        ## Logging
        writer.add_scalar(
            "train/density_app_plane_lr",
            omnilocal_rfs.rf_optimizers[-1].param_groups[0]["lr"],
            global_step=iteration,
        )
        writer.add_scalar(
            "train/basis_mat_lr",
            omnilocal_rfs.rf_optimizers[-1].param_groups[4]["lr"],
            global_step=iteration,
        )

        writer.add_scalar(
            "train/lr_r",
            omnilocal_rfs.r_optimizers[-1].param_groups[0]["lr"],
            global_step=iteration,
        )
        writer.add_scalar(
            "train/lr_t",
            omnilocal_rfs.t_optimizers[-1].param_groups[0]["lr"],
            global_step=iteration,
        )

        writer.add_scalar(
            "train/focal", omnilocal_rfs.focal(W), global_step=iteration
        )
        writer.add_scalar(
            "train/center0", omnilocal_rfs.center(W, H)[0].item(), global_step=iteration
        )
        writer.add_scalar(
            "train/center1", omnilocal_rfs.center(W, H)[1].item(), global_step=iteration
        )

        writer.add_scalar(
            "active_frames_bounds/0", train_dataset.active_frames_bounds[0], global_step=iteration
        )
        writer.add_scalar(
            "active_frames_bounds/1", train_dataset.active_frames_bounds[1], global_step=iteration
        )

        for index, blending_weights in enumerate(
            torch.permute(omnilocal_rfs.blending_weights, [1, 0])
        ):
            active_cam_indices = torch.nonzero(blending_weights)
            writer.add_scalar(
                f"tensorf_bounds/rf{index}_b0", active_cam_indices[0], global_step=iteration
            )
            writer.add_scalar(
                f"tensorf_bounds/rf{index}_b1", active_cam_indices[-1], global_step=iteration
            )

        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            print("active frame: ", train_dataset.active_frames_bounds)
            print("prev frame: ", train_dataset.prev_frames_bounds)
            print("frame_list: ", train_dataset.frame_list)
            # All poses visualization
            poses_mtx = omnilocal_rfs.get_cam2world().detach().cpu()
            t_w2rf = torch.stack(list(omnilocal_rfs.world2rf), dim=0).detach().cpu()
            RF_mtx_inv = torch.cat([torch.stack(len(t_w2rf) * [torch.eye(3)]), -t_w2rf.clone()[..., None]], axis=-1)

            all_poses = torch.cat([poses_mtx,  RF_mtx_inv], dim=0)
            colours = ["C1"] * poses_mtx.shape[0] + ["C2"] * RF_mtx_inv.shape[0]
            img = draw_poses(all_poses, colours)
            
            writer.add_image("poses/all", (np.transpose(img, (2, 0, 1)) / 255.0).astype(np.float32), iteration)

            # Neural feature grid            
            with torch.no_grad():
                '''         
                dyn_fields[None, ...] -> [1, feat_dim, F, H, W]
                ijf[None, :, None, None, ...] -> [1, n, 1, 1, 3], index order = [W, H, F]        
                output : [1, feat_dim, n, 1, 1]
                '''
                dyn_size = 200
                n_images = train_dataset.num_images - 1
                i = torch.linspace(-1,1,2*dyn_size, device=args.device)
                j = torch.linspace(-1,1,dyn_size, device=args.device)
                # Frame number for dynamic image displaying
                f_list = [1, 51, 101, 125]

                for num, f_num in enumerate(f_list):
                    f = torch.tensor(2*(f_num/n_images)-1.,device=args.device)
                    ijf = torch.stack(torch.meshgrid(i,j,f,indexing ='ij')).view(3,-1).permute([1, 0])

                    for level in range(omnilocal_rfs.dyn_multi_res):                    
                        if level == 0:
                            dyn_feat = torch.nn.functional.grid_sample(omnilocal_rfs.dyn_fields[level][None, ...], ijf[None, :, None, None,...],
                                                            mode='bilinear', padding_mode='border', align_corners=True)
                            dyn_feat = dyn_feat[0,:,:,0,0].permute([1, 0])
                        else:
                            dyn_feat = torch.cat([dyn_feat, 
                                                torch.nn.functional.grid_sample(omnilocal_rfs.dyn_fields[level][None, ...], ijf[None, :, None, None,...],
                                                mode='bilinear', padding_mode='border', align_corners=True)[0,:,:,0,0].permute([1, 0])],
                                                dim = -1
                            )
                    dyn = omnilocal_rfs.dyn_mlp(dyn_feat).permute([1,0]) ## [4, -1]
                    dyn_img = dyn.reshape(-1, 2*dyn_size, dyn_size).permute([0,2,1])
                    dyn_img = dyn_img.detach().cpu()
                    writer.add_image(f"dyn/dyn_{num}",dyn_img, iteration)
                    writer.add_image(f"dyn/dyn_{num}_rgb",dyn_img[:3,:,:], iteration)
                    writer.add_image(f"dyn/dyn_{num}_alpha",dyn_img[3:,:,:], iteration)
                    
            # Get runtime 
            ips = min(args.progress_refresh_rate, iteration + 1) / (time.time() - start_time)
            writer.add_scalar(f"train/iter_per_sec", ips, global_step=iteration)
            print(f"Iteration {iteration:06d}: {ips:.2f} it/s")
            start_time = time.time()

        if (iteration % args.vis_every == args.vis_every - 1):
            poses_mtx = omnilocal_rfs.get_cam2world().detach()
            rgb_maps_tb, depth_maps_tb, gt_rgbs_tb, fwd_flow_cmp_tb, bwd_flow_cmp_tb, depth_err_tb, loc_metrics = render(
                test_dataset,
                poses_mtx,
                omnilocal_rfs,
                args,
                W=W // 4, H=H // 4,
                savePath=logfolder,
                save_frames=True,
                img_format="jpg",
                test=True,
                train_dataset=train_dataset,
                start=0
            )

            if len(loc_metrics.values()):
                metrics.update(loc_metrics)
                mses = [metric["mse"] for metric in metrics.values()]
                writer.add_scalar(
                    f"test/PSNR", -10.0 * np.log(np.array(mses).mean()) / np.log(10.0), 
                    global_step=iteration
                )
                loc_mses = [metric["mse"] for metric in loc_metrics.values()]
                writer.add_scalar(
                    f"test/local_PSNR", -10.0 * np.log(np.array(loc_mses).mean()) / np.log(10.0), 
                    global_step=iteration
                )
                ssim = [metric["ssim"] for metric in metrics.values()]
                writer.add_scalar(
                    f"test/ssim", np.array(ssim).mean(), 
                    global_step=iteration
                )
                loc_ssim = [metric["ssim"] for metric in loc_metrics.values()]
                writer.add_scalar(
                    f"test/local_ssim", np.array(loc_ssim).mean(), 
                    global_step=iteration
                )

                writer.add_images(
                    "test/rgb_maps",
                    torch.stack(rgb_maps_tb, 0),
                    global_step=iteration,
                    dataformats="NHWC",
                )
                writer.add_images(
                    "test/depth_map",
                    torch.stack(depth_maps_tb, 0),
                    global_step=iteration,
                    dataformats="NHWC",
                )
                writer.add_images(
                    "test/gt_maps",
                    torch.stack(gt_rgbs_tb, 0),
                    global_step=iteration,
                    dataformats="NHWC",
                )
                
                if len(fwd_flow_cmp_tb) > 0:
                    writer.add_images(
                        "test/fwd_flow_cmp",
                        torch.stack(fwd_flow_cmp_tb, 0)[..., None],
                        global_step=iteration,
                        dataformats="NHWC",
                    )
                    
                    writer.add_images(
                        "test/bwd_flow_cmp",
                        torch.stack(bwd_flow_cmp_tb, 0)[..., None],
                        global_step=iteration,
                        dataformats="NHWC",
                    )
                
                if len(depth_err_tb) > 0:
                    writer.add_images(
                        "test/depth_cmp",
                        torch.stack(depth_err_tb, 0)[..., None],
                        global_step=iteration,
                        dataformats="NHWC",
                    )

            with open(f"{logfolder}/checkpoints_tmp.th", "wb") as f:
                omnilocal_rfs.save(f)

        iteration += 1

    with open(f"{logfolder}/checkpoints.th", "wb") as f:
        omnilocal_rfs.save(f)

    poses_mtx = omnilocal_rfs.get_cam2world().detach()
    render_frames(args, poses_mtx, omnilocal_rfs, logfolder, test_dataset=test_dataset, train_dataset=train_dataset)


if __name__ == "__main__":

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20240314)
    np.random.seed(20240314)

    args = config_parser()
    print(args)

    if args.render_only:
        render_test(args)
    else:
        reconstruction(args)
