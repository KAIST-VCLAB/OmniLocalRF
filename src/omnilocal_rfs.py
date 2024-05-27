# Copyright (c) Dongyoung Choi
# Copyright (c) Meta Platforms, Inc. and affiliates.

import math
import torch
from models.tensorBase import AlphaGridMask
from models.tensoRF import TensorVMSplit
from models.tensorBase import alpha2weights, positional_encoding
from utils.utils import mtx_to_sixD, sixD_to_mtx, N_to_reso, proj_360, get_cam2cams
from utils.ray_utils import get_ray_directions_lean, get_rays_lean, get_ray_directions_360

def ids2pixel_view(W, H, ids):
    '''
    Regress pixel coordinates from ray indices
    '''
    col = ids % W
    row = ids.div(W, rounding_mode='floor') % H
    view_ids = ids.div(W*H, rounding_mode='floor')
    return col, row, view_ids

def ids2pixel(W, H, ids):
    '''
    Regress pixel coordinates from ray indices
    '''
    col = ids % W
    row = ids.div(W, rounding_mode='floor') % H
    return col, row

class OmniLocalRFs(torch.nn.Module):
    '''
    Self calibrating local tensorfs.
    '''
    def __init__(
        self,
        fov,
        n_init_frames,
        n_frames,
        n_overlap,
        WH,
        n_iters_per_frame,
        n_iters_reg,
        lr_R_init,
        lr_t_init,
        lr_i_init,
        lr_exposure_init,
        lr_dyn,
        lr_dyn_mlp,
        rf_lr_init,
        rf_lr_basis,
        lr_decay_target_ratio,
        N_voxel_list,
        fin_alpha_block,
        update_AlphaMask_list,
        camera_prior,
        device,
        lr_upsample_reset,
        **tensorf_args,
    ):
        super(OmniLocalRFs, self).__init__()
        self.fov = fov
        self.n_init_frames = n_init_frames
        self.n_frames = n_frames
        self.n_overlap = n_overlap
        self.W, self.H = WH
        self.n_iters_per_frame = n_iters_per_frame
        self.n_iters_reg_per_frame = n_iters_reg
        self.lr_R_init, self.lr_t_init, self.lr_i_init, self.lr_exposure_init = lr_R_init, lr_t_init, lr_i_init, lr_exposure_init ## Camera Lr
        self.lr_dyn, self.lr_dyn_mlp = lr_dyn, lr_dyn_mlp                                                                         ## Dynamic Lr
        self.fin_alpha_block = fin_alpha_block
        self.rf_lr_init, self.rf_lr_basis, self.lr_decay_target_ratio = rf_lr_init, rf_lr_basis, lr_decay_target_ratio            ## RF Lr
        self.N_voxel_per_frame_list = N_voxel_list
        self.update_AlphaMask_per_frame_list = update_AlphaMask_list
        self.device = torch.device(device)
        self.camera_prior = camera_prior
        self.tensorf_args = tensorf_args
        self.is_refining = False
        self.lr_upsample_reset = lr_upsample_reset
        self.lr_factor = 1
        self.lr_dyn_rf_factor = 0
        self.regularize = True
        self.n_iters_reg = self.n_iters_reg_per_frame
        self.n_iters = self.n_iters_per_frame
        self.update_AlphaMask_list = update_AlphaMask_list
        self.N_voxel_list = N_voxel_list

        # Setup pose and camera parameters
        self.r_c2w, self.t_c2w, self.exposure = torch.nn.ParameterList(), torch.nn.ParameterList(), torch.nn.ParameterList()
        self.r_optimizers, self.t_optimizers, self.exp_optimizers, self.pose_linked_rf = [], [], [], [] 
        self.blending_weights = torch.nn.Parameter(
            torch.ones([1, 1], device=self.device, requires_grad=False), 
            requires_grad=False,
        )
        fov = fov * math.pi / 180
        focal = self.W / math.tan(fov / 2) / 2
        
        self.init_focal = torch.nn.Parameter(torch.Tensor([focal]).to(self.device))
        self.focal_offset = torch.nn.Parameter(torch.ones(1, device=device))
        self.center_rel = torch.nn.Parameter(0.5 * torch.ones(2, device=device))

        if lr_i_init > 0:
            self.intrinsic_optimizer = torch.optim.Adam([self.focal_offset, self.center_rel], betas=(0.9, 0.99), lr=self.lr_i_init)

        self.iter_aft_append_rf = 0

        # Setup dynamic fields [RGBA, time_bin, H, W]        
        dyn_res = 128
        dyn_feat_dim = 4
        self.dyn_multi_res = 4
        self.dyn_fields = torch.nn.ParameterList()
        for level in range(self.dyn_multi_res):
            self.dyn_fields.append(torch.nn.Parameter(torch.zeros((dyn_feat_dim, self.n_frames, 
                                                                    int(dyn_res/(2**level)),
                                                                    int(2 * dyn_res/(2**level))), device=device, requires_grad=True)))      
        dyn_mlp_nodes = 128
        self.dyn_mlp = torch.nn.Sequential(
            torch.nn.Linear(dyn_feat_dim * self.dyn_multi_res, dyn_mlp_nodes),
            torch.nn.Linear(dyn_mlp_nodes, dyn_mlp_nodes),
            torch.nn.Linear(dyn_mlp_nodes, 4),
            torch.nn.Sigmoid()
        ).to(device)        

        for _ in range(n_init_frames):
            self.append_frame()
        
        self.dyn_fields_optimizer = []
        self.dyn_mlp_optimizer = (torch.optim.Adam([{"params": self.dyn_mlp.parameters(), 
                                                    "lr": self.lr_dyn_mlp}],
                                                    betas=(0.9, 0.99)))
        for level in range(self.dyn_multi_res):
            self.dyn_fields_optimizer.append(torch.optim.Adam([{"params": self.dyn_fields[level],
                                                        "lr": self.lr_dyn/(level+1)}],
                                                        betas=(0.9, 0.99)))

        # Setup radiance fields
        self.tensorfs = torch.nn.ParameterList()
        self.rf_optimizers, self.rf_iter = [], []
        self.world2rf = torch.nn.ParameterList()
        self.append_rf()


    def append_rf(self, n_added_frames=1):
        self.is_refining = False
        if len(self.tensorfs) > 0:
            n_overlap = min(n_added_frames, self.n_overlap, self.blending_weights.shape[0] - 1)
            weights_overlap = 1 / n_overlap + torch.arange(
                0, 1, 1 / n_overlap
            )
            self.blending_weights.requires_grad = False
            self.blending_weights[-n_overlap :, -1] = 1 - weights_overlap
            new_blending_weights = torch.zeros_like(self.blending_weights[:, 0:1])
            new_blending_weights[-n_overlap :, 0] = weights_overlap
            self.blending_weights = torch.nn.Parameter(
                torch.cat([self.blending_weights, new_blending_weights], dim=1),
                requires_grad=False,
            )
            world2rf = -self.t_c2w[-1].clone().detach()            
        else:
            world2rf = torch.zeros(3, device=self.device)

        self.tensorfs.append(TensorVMSplit(device=self.device, **self.tensorf_args))
        self.world2rf.append(world2rf.clone().detach())        
        self.rf_iter.append(0)

        grad_vars = self.tensorfs[-1].get_optparam_groups(
            self.rf_lr_init, self.rf_lr_basis
        )
        self.rf_optimizers.append(torch.optim.Adam(grad_vars, betas=(0.9, 0.99)))

        # Make lr_dyn_rf_factor 0 to curb the reconstruction relying on the dyn fields at an initial phase
        self.lr_dyn_rf_factor = 0
        self.iter_aft_append_rf = 0
   
    def append_frame(self):
        if len(self.r_c2w) == 0:
            self.r_c2w.append(torch.eye(3, 2, device=self.device))
            self.t_c2w.append(torch.zeros(3, device=self.device))

            self.pose_linked_rf.append(0)            
        else:
            self.r_c2w.append(mtx_to_sixD(sixD_to_mtx(self.r_c2w[-1].clone().detach()[None]))[0])
            self.t_c2w.append(self.t_c2w[-1].clone().detach())

            self.blending_weights = torch.nn.Parameter(
                torch.cat([self.blending_weights, self.blending_weights[-1:, :]], dim=0),
                requires_grad=False,
            )

            rf_ind = int(torch.nonzero(self.blending_weights[-1, :])[0])
            self.pose_linked_rf.append(rf_ind)
            
            # Initialize a new mask by the previous one
            for level in range(self.dyn_multi_res):
                self.dyn_fields[level][:,len(self.r_c2w)-1,:,:].data = self.dyn_fields[level][:,len(self.r_c2w)-2,:,:].data
                
        self.exposure.append(torch.eye(3, 3, device=self.device))


        if self.camera_prior is not None:
            idx = len(self.r_c2w) - 1
            rel_pose = self.camera_prior["rel_poses"][idx]
            last_r_c2w = sixD_to_mtx(self.r_c2w[-1].clone().detach()[None])[0]
            self.r_c2w[-1] = last_r_c2w @ rel_pose[:3, :3]
            self.t_c2w[-1].data += last_r_c2w @ rel_pose[:3, 3]
            
        self.r_optimizers.append(torch.optim.Adam([self.r_c2w[-1]], betas=(0.9, 0.99), lr=self.lr_R_init)) 
        self.t_optimizers.append(torch.optim.Adam([self.t_c2w[-1]], betas=(0.9, 0.99), lr=self.lr_t_init)) 
        self.exp_optimizers.append(torch.optim.Adam([self.exposure[-1]], betas=(0.9, 0.99), lr=self.lr_exposure_init)) 

    def optimizer_step_poses_only(self, loss):
        for idx in range(len(self.r_optimizers)):
            if self.pose_linked_rf[idx] == len(self.rf_iter) - 1 and self.rf_iter[-1] < self.n_iters:
                self.r_optimizers[idx].zero_grad()
                self.t_optimizers[idx].zero_grad()
        
        loss.backward()

        # Optimize poses
        for idx in range(len(self.r_optimizers)):
            if self.pose_linked_rf[idx] == len(self.rf_iter) - 1 and self.rf_iter[-1] < self.n_iters:
                self.r_optimizers[idx].step()
                self.t_optimizers[idx].step()
                
    def optimizer_step(self, loss, optimize_poses):
        if self.rf_iter[-1] == 0:
            self.lr_factor = 1
            self.lr_cam_factor = 1
            self.lr_dynmlp_factor = 1
            self.n_iters = self.n_iters_per_frame
            self.n_iters_reg = self.n_iters_reg_per_frame
        elif self.rf_iter[-1] == 1:
            n_training_frames = (self.blending_weights[:, -1] > 0).sum()
            self.n_iters = int(self.n_iters_per_frame * n_training_frames)
            self.n_iters_reg = int(self.n_iters_reg_per_frame * n_training_frames)
            self.lr_factor = self.lr_decay_target_ratio ** (1 / self.n_iters)
            self.lr_cam_factor = (0.1 * self.lr_decay_target_ratio) ** (1 / self.n_iters)
            self.lr_dynmlp_factor = (0.01 * self.lr_decay_target_ratio) ** (1 / self.n_iters)
            self.N_voxel_list = {int(key * n_training_frames): self.N_voxel_per_frame_list[key] for key in self.N_voxel_per_frame_list}
            self.update_AlphaMask_list = [int(update_AlphaMask * n_training_frames) for update_AlphaMask in self.update_AlphaMask_per_frame_list]

        self.regularize = self.rf_iter[-1] < self.n_iters_reg

        for idx in range(len(self.r_optimizers)):
            if self.pose_linked_rf[idx] == len(self.rf_iter) - 1 and self.rf_iter[-1] < self.n_iters:
                # Poses
                if optimize_poses:
                    for param_group in self.r_optimizers[idx].param_groups:
                        param_group["lr"] *= self.lr_cam_factor
                    for param_group in self.t_optimizers[idx].param_groups:
                        param_group["lr"] *= self.lr_cam_factor
                    # Always initialize gradients of camera pose
                    self.r_optimizers[idx].zero_grad()
                    self.t_optimizers[idx].zero_grad()
                
                # Exposure
                if self.lr_exposure_init > 0:
                    for param_group in self.exp_optimizers[idx].param_groups:
                        param_group["lr"] *= self.lr_factor
                    self.exp_optimizers[idx].zero_grad()

        '''
        Initialize the gradients
        '''
        # Intrinsics
        if (
            self.lr_i_init > 0 and 
            self.blending_weights.shape[1] == 1 and 
            self.is_refining
        ):
            for param_group in self.intrinsic_optimizer.param_groups:
                param_group["lr"] *= self.lr_factor
            self.intrinsic_optimizer.zero_grad()

        # TensoRFs
        for optimizer, iteration in zip(self.rf_optimizers, self.rf_iter):
            if iteration < self.n_iters:
                optimizer.zero_grad()        
        if len(self.rf_optimizers) > 1:
            for i in range(len(self.rf_optimizers)-1):
                self.rf_optimizers[i].zero_grad()

        # Mask module
        self.dyn_mlp_optimizer.zero_grad()        
        for param_group in self.dyn_mlp_optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * self.lr_dynmlp_factor
        for level in range(self.dyn_multi_res):
            self.dyn_fields_optimizer[level].zero_grad()


        loss.backward()

        '''
        Optimize the parameters
        '''
        # Optimize RFs
        self.rf_optimizers[-1].step()
        if self.is_refining:
            for param_group in self.rf_optimizers[-1].param_groups:
                param_group["lr"] = param_group["lr"] * self.lr_factor
        if len(self.rf_optimizers) > 1:
            for i in range(len(self.rf_optimizers)-1):
                self.rf_optimizers[i].step()

        # Optimize mask module  
        # Step only if the mask module is not blocked
        if self.iter_aft_append_rf > self.dyn_block_iter:
            self.dyn_mlp_optimizer.step()    
            for level in range(self.dyn_multi_res):
                    self.dyn_fields_optimizer[level].step()
        
        # Increase RF resolution
        if self.rf_iter[-1] in self.N_voxel_list:
            n_voxels = self.N_voxel_list[self.rf_iter[-1]]
            reso_cur = N_to_reso(n_voxels, self.tensorfs[-1].aabb)
            self.tensorfs[-1].upsample_volume_grid(reso_cur)

            if self.lr_upsample_reset:
                print("reset lr to initial")
                grad_vars = self.tensorfs[-1].get_optparam_groups(
                    self.rf_lr_init, self.rf_lr_basis
                )
                self.rf_optimizers[-1] = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

        # Update alpha RF mask for computing acceleration
        if iteration in self.update_AlphaMask_list:
            reso_mask = (self.tensorfs[-1].gridSize / 2).int()
            self.tensorfs[-1].updateAlphaMask(tuple(reso_mask))

        for idx in range(len(self.r_optimizers)):
            if self.pose_linked_rf[idx] == len(self.rf_iter) - 1 and self.rf_iter[-1] < self.n_iters:
                # Optimize poses
                if optimize_poses:
                    self.r_optimizers[idx].step()
                    self.t_optimizers[idx].step()
                # Optimize exposures
                if self.lr_exposure_init > 0:
                    self.exp_optimizers[idx].step()
        
        # Optimize intrinsics
        if (
            self.lr_i_init > 0 and 
            self.blending_weights.shape[1] == 1 and
            self.is_refining 
        ):
            self.intrinsic_optimizer.step()

        if self.is_refining:
            self.rf_iter[-1] += 1

        can_add_rf = self.rf_iter[-1] >= self.n_iters - 1
        return can_add_rf

    def activate_dyn_fields(self):
        self.lr_dyn_rf_factor = 1
        return

    def get_cam2world(self, view_ids=None, starting_id=0):
        if view_ids is not None:
            r_c2w = torch.stack([self.r_c2w[view_id] for view_id in view_ids], dim=0)
            t_c2w = torch.stack([self.t_c2w[view_id] for view_id in view_ids], dim=0)
        else:
            r_c2w = torch.stack(list(self.r_c2w[starting_id:]), dim=0)
            t_c2w = torch.stack(list(self.t_c2w[starting_id:]), dim=0)
        return torch.cat([sixD_to_mtx(r_c2w), t_c2w[..., None]], dim = -1)

    def get_kwargs(self):
        kwargs = {
            "camera_prior": None,
            "fov": self.fov,
            "n_init_frames": self.n_init_frames,
            "n_overlap": self.n_overlap,
            "WH": (self.W, self.H),
            "n_iters_per_frame": self.n_iters_per_frame,
            "n_iters_reg": self.n_iters_reg_per_frame,
            "lr_R_init": self.lr_R_init,
            "lr_t_init": self.lr_t_init,
            "lr_i_init": self.lr_i_init,
            "lr_exposure_init": self.lr_exposure_init,
            "rf_lr_init": self.rf_lr_init,
            "rf_lr_basis": self.rf_lr_basis,
            "lr_decay_target_ratio": self.lr_decay_target_ratio,
            "lr_decay_target_ratio": self.lr_decay_target_ratio,
            "N_voxel_list": self.N_voxel_per_frame_list,
            "update_AlphaMask_list": self.update_AlphaMask_per_frame_list,
            "lr_upsample_reset": self.lr_upsample_reset,
            "n_frames": self.n_frames,
            "lr_dyn": self.lr_dyn,
            "lr_dyn_mlp": self.lr_dyn_mlp,
            "fin_alpha_block": self.fin_alpha_block,
        }
        kwargs.update(self.tensorfs[0].get_kwargs())

        return kwargs

    def save(self, path):
        kwargs = self.get_kwargs()
        ckpt = {"kwargs": kwargs, "state_dict": self.state_dict()}
        torch.save(ckpt, path)

    def load(self, state_dict):
        # TODO A bit hacky?
        import re
        n_frames = 0
        for key in state_dict:
            if re.fullmatch(r"r_c2w.[0-9]*", key):
                n_frames += 1
            if re.fullmatch(r"tensorfs.[1-9][0-9]*.density_plane.0", key):
                self.tensorf_args["gridSize"] = [state_dict[key].shape[2], state_dict[key].shape[3], state_dict[f"{key[:-15]}density_line.0"].shape[2]]
                self.append_rf()

        for i in range(len(self.tensorfs)):
            if f"tensorfs.{i}.alphaMask.aabb" in state_dict:
                alpha_volume = state_dict[f'tensorfs.{i}.alphaMask.alpha_volume'].to(self.device)
                aabb = state_dict[f'tensorfs.{i}.alphaMask.aabb'].to(self.device)
                self.tensorfs[i].alphaMask = AlphaGridMask(self.device, aabb, alpha_volume)


        for _ in range(n_frames - len(self.r_c2w)):
            self.append_frame()
        
        self.blending_weights = torch.nn.Parameter(
            torch.ones_like(state_dict["blending_weights"]), requires_grad=False
        )

        self.load_state_dict(state_dict)

    def get_dist_to_last_rf(self):
        return torch.norm(self.t_c2w[-1] + self.world2rf[-1])

    def get_reg_loss(self, tvreg, TV_weight_density, TV_weight_app, L1_weight_inital):
        tv_loss = 0
        l1_loss = 0
        if self.rf_iter[-1] < self.n_iters:
            if TV_weight_density > 0:
                tv_weight = TV_weight_density * (self.lr_factor ** self.rf_iter[-1])
                tv_loss += self.tensorfs[-1].TV_loss_density(tvreg).mean() * tv_weight
                
            if TV_weight_app > 0:
                tv_weight = TV_weight_app * (self.lr_factor ** self.rf_iter[-1])
                tv_loss += self.tensorfs[-1].TV_loss_app(tvreg).mean() * tv_weight
    
            if L1_weight_inital > 0:
                l1_loss += self.tensorfs[-1].density_L1() * L1_weight_inital

        return tv_loss, l1_loss

    def get_prev_reg_loss(self, tvreg, TV_weight_density, TV_weight_app, L1_weight_inital, rf_ids):
        tv_loss = 0
        l1_loss = 0
        if self.rf_iter[-1] < self.n_iters:          
            if TV_weight_density > 0:
                tv_weight = TV_weight_density * (self.lr_factor ** self.rf_iter[-2])
                tv_loss += self.tensorfs[rf_ids].TV_loss_density(tvreg).mean() * tv_weight
                
            if TV_weight_app > 0:
                tv_weight = TV_weight_app * (self.lr_factor ** self.rf_iter[-2])
                tv_loss += self.tensorfs[rf_ids].TV_loss_app(tvreg).mean() * tv_weight
    
            if L1_weight_inital > 0:
                l1_loss += self.tensorfs[rf_ids].density_L1() * L1_weight_inital

        return tv_loss, l1_loss

    def focal(self, W):
        return self.init_focal * self.focal_offset * W / self.W 
    def center(self, W, H):
        return torch.Tensor([W, H]).to(self.center_rel) * self.center_rel

    def get_dyn_tv(
        self,
        frame_begin,
        frame_end,
        num_images,
        dyn_size=200
        ):
        frame_num = int(frame_end - frame_begin + 1)
        frame_begin = 2 * (frame_begin/(num_images-1)) - 1
        frame_end = 2 * (frame_end/(num_images-1)) - 1
        
        i = torch.linspace(-1,1,2*dyn_size, device=self.device)
        j = torch.linspace(-1,1,dyn_size, device=self.device)
        f = torch.linspace(frame_begin, frame_end , frame_num, device=self.device)
        ijf = torch.stack(torch.meshgrid(i,j,f,indexing ='ij')).view(3,-1).permute([1, 0])
        
        for level in range(self.dyn_multi_res):                    
            if level == 0:
                dyn_feat = torch.nn.functional.grid_sample(self.dyn_fields[level][None, ...], ijf[None, :, None, None,...],
                                                mode='bilinear', padding_mode='border', align_corners=True)
                dyn_feat = dyn_feat[0,:,:,0,0].permute([1, 0])
            else:
                dyn_feat = torch.cat([dyn_feat, 
                                    torch.nn.functional.grid_sample(self.dyn_fields[level][None, ...], ijf[None, :, None, None,...],
                                    mode='bilinear', padding_mode='border', align_corners=True)[0,:,:,0,0].permute([1, 0])],
                                    dim = -1
                )
        dyn = self.dyn_mlp(dyn_feat).permute([1,0]) ## [4, -1]
        dyn_img = dyn.reshape(-1, 2*dyn_size, dyn_size, frame_num).permute([0,2,1,3])        
        dyn_tvx = ((dyn_img[-1,:-1,:,:] - dyn_img[-1,1:,:,:])**2).mean()
        dyn_tvy = ((dyn_img[-1,:,:-1,:] - dyn_img[-1,:,1:,:])**2).mean()
        dyn_tvt = ((dyn_img[-1,:,:,:-1] - dyn_img[-1,:,:,1:])**2).mean()
        
        return dyn_tvx, dyn_tvy, dyn_tvt
   
    def rf_forward(        
        self,
        tensorf,
        rays_chunk,
        ray_ids,
        white_bg=True,
        is_train=False,
        N_samples=-1,
        refine=True,
        detach_dyn=False,
        floater_thresh=0,
        ):        
        # Sample points
        viewdirs = rays_chunk[:, 3:6]
        viewdirs_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = viewdirs / viewdirs_norm
        xyz_sampled, z_vals, ray_valid = tensorf.sample_ray_contracted(
            rays_chunk[:, :3], viewdirs, is_train=is_train, N_samples=N_samples
        )
        dists = torch.cat(
            (z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])),
            dim=-1,
        )
        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)

        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)

        if tensorf.alphaMask is not None:
            alphas = tensorf.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= (~alpha_mask)
            ray_valid = ~ray_invalid
        
        ray_valid[:, -1] = 0
        if ray_valid.any():
            xyz_sampled = tensorf.normalize_coord(xyz_sampled)
            sigma_feature = tensorf.compute_densityfeature(
                xyz_sampled[ray_valid],
            )

            validsigma = tensorf.feature2density(sigma_feature)
            sigma[ray_valid] = validsigma

        alpha = 1.0 - torch.exp(-sigma * dists * tensorf.distance_scale)
              
        if rays_chunk.shape[0] == ray_ids.shape[0]: # training
            i, j, f = ids2pixel_view(self.W, self.H, ray_ids)
            i = i / self.W
            j = j / self.H
            f = f / (self.n_frames-1)
            ijf = torch.cat((i[...,None], j[..., None], f[...,None]), dim=-1)
            ijf = 2 * ijf - 1 # convert ij range to [-1, 1] for F.sample_grid

            '''
            Neural feature grid
            dyn_fields[None, ...] -> [1, feat_dim, F, H, W]
            ijf[None, :, None, None, ...] -> [1, n, 1, 1, 3], index order = [W, H, F]        
            output : [1, feat_dim, n, 1, 1]
            '''
            for level in range(self.dyn_multi_res):
                if level == 0:
                    dyn_feat = torch.nn.functional.grid_sample(self.dyn_fields[level][None, ...], ijf[None, :, None, None,...],
                                                    mode='bilinear', padding_mode='border', align_corners=True)
                    dyn_feat = dyn_feat[0,:,:,0,0].permute([1, 0])
                else:
                    dyn_feat = torch.cat([dyn_feat, 
                                        torch.nn.functional.grid_sample(self.dyn_fields[level][None, ...], ijf[None, :, None, None,...],
                                        mode='bilinear', padding_mode='border', align_corners=True)[0,:,:,0,0].permute([1, 0])],
                                        dim = -1
                    )

            dyn_ori = self.dyn_mlp(dyn_feat)            
            if self.lr_dyn == 0 or self.lr_dyn_rf_factor == 0:
                dyn = torch.zeros_like(dyn_ori, device=dyn_ori.device)
            else:
                dyn = dyn_ori               

            dyn_alpha = dyn[:,3][...,None]
            
            if detach_dyn:
                dyn_alpha = dyn[:,3][...,None].clone().detach()

            alpha_w_dyn = torch.cat([dyn_alpha, alpha], dim = -1)
            weight_w_dyn, T_w_dyn = alpha2weights(alpha_w_dyn, fin_alpha_block=self.fin_alpha_block)
            weight, T = weight_w_dyn[:, 1:], T_w_dyn[:, 1:]
            acc_map = torch.sum(weight, -1)            

            weight_static, T_static = alpha2weights(alpha, fin_alpha_block=self.fin_alpha_block)
            depth_map = (torch.sum(weight_static * z_vals, -1))/ viewdirs_norm[..., 0]

            if floater_thresh > 0:
                idx_map = torch.sum(weight * torch.arange(alpha.shape[1], device=alpha.device)[None], -1, keepdim=True)
                alpha[torch.arange(alpha.shape[1], device=alpha.device)[None] < idx_map * floater_thresh] = 0
                alpha_w_dyn = torch.cat([dyn_alpha, alpha], dim = -1)
                weight_w_dyn, T_w_dyn = alpha2weights(alpha_w_dyn, fin_alpha_block=self.fin_alpha_block)
                weight, T = weight_w_dyn[:, 1:], T_w_dyn[:, 1:]            
        
            if detach_dyn:
                dyn_map = dyn_alpha.clone().detach() * dyn[:,:3].clone().detach()
            else:
                dyn_map = dyn_alpha * dyn[:,:3]
            dyn_rgb = dyn_ori

        else: # Testing: Not generating dynamic objects
            weight, T = alpha2weights(alpha, fin_alpha_block=self.fin_alpha_block)
            acc_map = torch.sum(weight, -1)
            
            depth_map = (torch.sum(weight * z_vals, -1))/ viewdirs_norm[..., 0]

            if floater_thresh > 0:
                idx_map = torch.sum(weight * torch.arange(alpha.shape[1], device=alpha.device)[None], -1, keepdim=True)
                alpha[torch.arange(alpha.shape[1], device=alpha.device)[None] < idx_map * floater_thresh] = 0
                weight, T = alpha2weights(alpha, fin_alpha_block=self.fin_alpha_block)    

            dyn_map = 0
            dyn_rgb = torch.zeros_like(torch.sum(weight[..., None], -2))
        
        app_mask = weight > tensorf.rayMarch_weight_thres
        if app_mask.any():
            app_features = tensorf.compute_appfeature(
                xyz_sampled[app_mask],
            )
            valid_rgbs = tensorf.renderModule(
                xyz_sampled[app_mask], viewdirs[app_mask].clone().detach(), app_features, refine
            )
            rgb[app_mask] = valid_rgbs

        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        static_rgb_map = rgb_map.clamp(0,1)
        rgb_map = (dyn_map + rgb_map).clamp(0,1)

        return rgb_map, static_rgb_map, depth_map, dyn_rgb, T[:,-1]

        
    def rf_forward_depth(        
        self,
        tensorf,
        rays_chunk,
        white_bg=True,
        is_train=False,
        N_samples=-1,
        refine=True,
        floater_thresh=0,
        ):
        
        # sample points
        viewdirs = rays_chunk[:, 3:6]
        viewdirs_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = viewdirs / viewdirs_norm
        xyz_sampled, z_vals, ray_valid = tensorf.sample_ray_contracted(
            rays_chunk[:, :3], viewdirs, is_train=is_train, N_samples=N_samples
        )
        dists = torch.cat(
            (z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])),
            dim=-1,
        )
        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)

        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)

        if tensorf.alphaMask is not None:
            alphas = tensorf.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= (~alpha_mask)
            ray_valid = ~ray_invalid
        
        ray_valid[:, -1] = 0
        if ray_valid.any():
            xyz_sampled = tensorf.normalize_coord(xyz_sampled)
            sigma_feature = tensorf.compute_densityfeature(
                xyz_sampled[ray_valid],
            )

            validsigma = tensorf.feature2density(sigma_feature)
            sigma[ray_valid] = validsigma

        alpha = 1.0 - torch.exp(-sigma * dists * tensorf.distance_scale)             
        weight_static, T_static = alpha2weights(alpha, fin_alpha_block=self.fin_alpha_block)
        depth_map = (torch.sum(weight_static * z_vals, -1))/ viewdirs_norm[..., 0]
            
        return depth_map


    def rf_refine_forward_uv(    
        self,
        tensorf,
        rays_chunk,
        dst_ray_ids,
        uv,
        is_train=False,
        N_samples=-1,
        refine=True,
        floater_thresh=0
        ):  
        
        # sample points
        viewdirs = rays_chunk[:, 3:6]
        viewdirs_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = viewdirs / viewdirs_norm
        xyz_sampled, z_vals, ray_valid = tensorf.sample_ray_contracted(
            rays_chunk[:, :3], viewdirs, is_train=is_train, N_samples=N_samples
        )        
        dists = torch.cat(
            (z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])),
            dim=-1,
        )
        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)

        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)

        if tensorf.alphaMask is not None:
            alphas = tensorf.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= (~alpha_mask)
            ray_valid = ~ray_invalid
        
        ray_valid[:, -1] = 0
        if ray_valid.any():
            xyz_sampled = tensorf.normalize_coord(xyz_sampled)
            sigma_feature = tensorf.compute_densityfeature(
                xyz_sampled[ray_valid],
            )

            validsigma = tensorf.feature2density(sigma_feature)
            sigma[ray_valid] = validsigma

        alpha = 1.0 - torch.exp(-sigma * dists * tensorf.distance_scale)
        
        # Compute dynamic fields        
        if rays_chunk.shape[0] == dst_ray_ids.shape[0]: # training
            _, _, f = ids2pixel_view(self.W, self.H, dst_ray_ids)
            i = uv[:,0]
            i[i<0] += self.W
            i = i / self.W

            j = uv[:,1] 
            j[j<0] += self.H
            j = j / self.H

            f = f / (self.n_frames-1)
            ijf = torch.cat((i[...,None], j[..., None], f[...,None]), dim=-1)
            ijf = 2 * ijf - 1 # convert ij range to [-1, 1] for F.sample_grid

            # Multi resolution grid 
            for level in range(self.dyn_multi_res):
                if level == 0:
                    dyn_feat = torch.nn.functional.grid_sample(self.dyn_fields[level][None, ...], ijf[None, :, None, None,...],
                                                    mode='bilinear', padding_mode='border', align_corners=True)
                    dyn_feat = dyn_feat[0,:,:,0,0].permute([1, 0])
                else:
                    dyn_feat = torch.cat([dyn_feat, 
                                        torch.nn.functional.grid_sample(self.dyn_fields[level][None, ...], ijf[None, :, None, None,...],
                                        mode='bilinear', padding_mode='border', align_corners=True)[0,:,:,0,0].permute([1, 0])],
                                        dim = -1
                    )

            dyn_ori = self.dyn_mlp(dyn_feat)
            if self.lr_dyn == 0 or self.lr_dyn_rf_factor == 0:
                dyn = torch.zeros_like(dyn_ori, device=dyn_ori.device)
            else:
                dyn = dyn_ori
            dyn_alpha = dyn[:,3][...,None]

            alpha_w_dyn = torch.cat([dyn_alpha, alpha], dim = -1)
            weight_w_dyn, T_w_dyn = alpha2weights(alpha_w_dyn, fin_alpha_block=self.fin_alpha_block)
            weight, T = weight_w_dyn[:, 1:], T_w_dyn[:, 1:]
            acc_map = torch.sum(weight, -1)

            weight_static, T_static = alpha2weights(alpha, fin_alpha_block=self.fin_alpha_block)
            depth_map = (torch.sum(weight_static * z_vals, -1))/ viewdirs_norm[..., 0]

            if floater_thresh > 0:
                idx_map = torch.sum(weight * torch.arange(alpha.shape[1], device=alpha.device)[None], -1, keepdim=True)
                alpha[torch.arange(alpha.shape[1], device=alpha.device)[None] < idx_map * floater_thresh] = 0
                alpha_w_dyn = torch.cat([dyn_alpha, alpha], dim = -1)
                weight_w_dyn, T_w_dyn = alpha2weights(alpha_w_dyn, fin_alpha_block=self.fin_alpha_block)
                weight, T = weight_w_dyn[:, 1:], T_w_dyn[:, 1:]            
            
            dyn_map = dyn_alpha.clone().detach() * dyn[:,:3].clone().detach()
            dyn_rgb = dyn_ori

        else: # Testing: Not generating dynamic objects
            weight, T = alpha2weights(alpha, fin_alpha_block=self.fin_alpha_block)
            acc_map = torch.sum(weight, -1)
            
            depth_map = torch.sum(weight * z_vals, -1)/ viewdirs_norm[..., 0]

            if floater_thresh > 0:
                idx_map = torch.sum(weight * torch.arange(alpha.shape[1], device=alpha.device)[None], -1, keepdim=True)
                alpha[torch.arange(alpha.shape[1], device=alpha.device)[None] < idx_map * floater_thresh] = 0
                weight, T = alpha2weights(alpha, fin_alpha_block=self.fin_alpha_block)    

            dyn_map = 0
            dyn_rgb = torch.zeros_like(torch.sum(weight[..., None], -2))
        
        app_mask = weight > tensorf.rayMarch_weight_thres
        if app_mask.any():
            app_features = tensorf.compute_appfeature(
                xyz_sampled[app_mask],
            )
            valid_rgbs = tensorf.renderModule(
                xyz_sampled[app_mask], viewdirs[app_mask].clone().detach(), app_features, refine
            )
            rgb[app_mask] = valid_rgbs

        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        static_rgb_map = rgb_map
        rgb_map = (dyn_map + rgb_map).clamp(0,1)

        return rgb_map, static_rgb_map, depth_map, dyn_rgb, T[:,-1]

    def forward(
        self,
        ray_ids,
        view_ids,
        W,
        H,
        white_bg=True,
        is_train=True,
        cam2world=None,
        world2rf=None,
        blending_weights=None,
        prev_rf=False,
        prev_rf_ids=None,
        chunk=16384,
        test_id=False,
        floater_thresh=0,
        ):
        i, j = ids2pixel(W, H, ray_ids)
        if self.fov == 360:
            directions = get_ray_directions_360(i, j, W, H)
        else:
            directions = get_ray_directions_lean(i, j, self.focal(W), self.center(W, H))

        if blending_weights is None:
            blending_weights = self.blending_weights[view_ids].clone()
        if cam2world is None:
            cam2world = self.get_cam2world(view_ids)
        if world2rf is None:
            world2rf = self.world2rf

        # Train a single RF at a time
        if is_train:            
            if prev_rf:                
                blending_weights[:,:] = 0
                blending_weights[:, prev_rf_ids] = 1
            else:
                blending_weights[:, -1] = 1
                blending_weights[:, :-1] = 0

        if is_train:
            if prev_rf:
                active_rf_ids = [prev_rf_ids]
            else:                
                active_rf_ids = [len(self.tensorfs) - 1]
        else:
            active_rf_ids = torch.nonzero(torch.sum(blending_weights, dim=0))[:, 0].tolist()
        ij = torch.stack([i, j], dim=-1)
        if len(active_rf_ids) == 0:
            print("****** No valid RF")
            return torch.ones([ray_ids.shape[0], 3]), torch.ones_like(ray_ids).float(), torch.ones_like(ray_ids).float(), directions, ij

        cam2rfs = {}
        initial_devices = []
        for rf_id in active_rf_ids:
            cam2rf = cam2world.clone()
            cam2rf[:, :3, 3] += world2rf[rf_id]
            cam2rfs[rf_id] = cam2rf
            
            initial_devices.append(self.tensorfs[rf_id].device)
            if initial_devices[-1] != view_ids.device:
                self.tensorfs[rf_id].to(view_ids.device)

        for key in cam2rfs:
            cam2rfs[key] = cam2rfs[key].repeat_interleave(ray_ids.shape[0] // view_ids.shape[0], dim=0)
        blending_weights_expanded = blending_weights.repeat_interleave(ray_ids.shape[0] // view_ids.shape[0], dim=0)
        rgbs = torch.zeros_like(directions)
        static_rgbs = torch.zeros_like(directions)
        dyn = torch.zeros((directions.shape[0],4),device=directions.device)  ## Include dynamic alpha
        depth_maps = torch.zeros_like(directions[..., 0]) 
        T_fin = torch.zeros_like(directions[..., 0]) 
        N_rays_all = ray_ids.shape[0]
        chunk = chunk // len(active_rf_ids)
        for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
            if chunk_idx != 0:
                torch.cuda.empty_cache()
            directions_chunk = directions[chunk_idx * chunk : (chunk_idx + 1) * chunk]
            blending_weights_chunk = blending_weights_expanded[
                chunk_idx * chunk : (chunk_idx + 1) * chunk
            ]

            for rf_id in active_rf_ids:
                blending_weight_chunk = blending_weights_chunk[:, rf_id]
                cam2rf = cam2rfs[rf_id][chunk_idx * chunk : (chunk_idx + 1) * chunk]

                rays_o, rays_d = get_rays_lean(directions_chunk, cam2rf)
                rays = torch.cat([rays_o, rays_d], -1).view(-1, 6)                 
                
                rgb_map_t, static_rgb_map_t, depth_map_t, dyn_map_t, T_fin_t = self.rf_forward(
                    self.tensorfs[rf_id],
                    rays,
                    ray_ids,
                    is_train=is_train,
                    white_bg=white_bg,
                    N_samples=-1,
                    refine=self.is_refining,
                    detach_dyn=prev_rf,
                    floater_thresh=floater_thresh,
                )      

                rgbs[chunk_idx * chunk : (chunk_idx + 1) * chunk] = (
                    rgbs[chunk_idx * chunk : (chunk_idx + 1) * chunk] + 
                    rgb_map_t * blending_weight_chunk[..., None]
                )
                static_rgbs[chunk_idx * chunk : (chunk_idx + 1) * chunk] = (
                    static_rgbs[chunk_idx * chunk : (chunk_idx + 1) * chunk] + 
                    static_rgb_map_t * blending_weight_chunk[..., None]
                )
                depth_maps[chunk_idx * chunk : (chunk_idx + 1) * chunk] = (
                    depth_maps[chunk_idx * chunk : (chunk_idx + 1) * chunk] + 
                    depth_map_t * blending_weight_chunk
                )
                dyn[chunk_idx * chunk : (chunk_idx + 1) * chunk] = (
                    dyn[chunk_idx * chunk : (chunk_idx + 1) * chunk] + 
                    dyn_map_t * blending_weight_chunk[..., None]
                )
                T_fin[chunk_idx * chunk : (chunk_idx + 1) * chunk] = (
                    T_fin[chunk_idx * chunk : (chunk_idx + 1) * chunk] + 
                    T_fin_t * blending_weight_chunk
                )

        for rf_id, initial_device in zip(active_rf_ids, initial_devices):
            if initial_device != view_ids.device:
                self.tensorfs[rf_id].to(initial_device)
                torch.cuda.empty_cache()

        if self.lr_exposure_init > 0:
            # TODO: cleanup
            if test_id:
                view_ids_m = torch.maximum(view_ids - 1, torch.tensor(0, device=view_ids.device))
                view_ids_m[view_ids_m==view_ids] = 1
                
                view_ids_p = torch.minimum(view_ids + 1, torch.tensor(len(self.exposure) - 1, device=view_ids.device))
                view_ids_p[view_ids_m==view_ids] = len(self.exposure) - 2
                
                exposure_stacked = torch.stack(list(self.exposure), dim=0).clone().detach()
                exposure = (exposure_stacked[view_ids_m] + exposure_stacked[view_ids_p]) / 2  
            else:
                exposure = torch.stack(list(self.exposure), dim=0)[view_ids]
                
            exposure = exposure.repeat_interleave(ray_ids.shape[0] // view_ids.shape[0], dim=0)
            rgbs = torch.bmm(exposure, static_rgbs[..., None])[..., 0] 
            if self.lr_dyn > 0 :      
                if prev_rf:
                    rgbs = (rgbs + dyn[:,:3].clone().detach() * dyn[:,3:].clone().detach()).clamp(0, 1)
                else:
                    rgbs = (rgbs + dyn[:,:3] * dyn[:,3:]).clamp(0, 1)
            else:
                rgbs = rgbs.clamp(0,1)

        return rgbs, static_rgbs, depth_maps, directions, ij, dyn, T_fin


    def forward_step(        
        self,
        ray_ids,
        view_ids,
        directions,
        depth_map,
        refine_cam2world,
        curr_rf_ids,
        prev_rf_ids,
        frame_src,
        frame_dst,
        W,
        H,
        train_dataset,
        forward_depth_margin,
        forward_depth_thres,
        ):        
        src2dst = frame_dst - frame_src            
        src2dst_cam = get_cam2cams(refine_cam2world, frame_src, src2dst)   
        directions_src = directions.view(view_ids.shape[0], -1, 3) # NDC direction from src          
        depth_map = depth_map.view(view_ids.shape[0], -1)
        pts = directions_src * depth_map[..., None]  ## pts at NDC of src frame          

        ## Detach src2dst camera pose to prevent contaminating entire camera trajectory due to multiplication ambiguity
        project_pts = proj_360(pts, src2dst_cam, W, H).detach() ## floating points pixels coord. on dst            
        dst_rgbs, dst_ray_ids, _ = train_dataset.bilerp(project_pts=project_pts, frame_dst=frame_dst)
        
        dst_cam2curr_rf = self.get_cam2world(frame_dst).clone()
        dst_cam2curr_rf[:, :3, 3] += self.world2rf[curr_rf_ids] ## Move camera orientation with respect to the reference RF
        dst_cam2curr_rf = dst_cam2curr_rf.repeat_interleave(dst_ray_ids.shape[0] // frame_dst.shape[0], dim=0)

        project_pts = project_pts.view(-1,2)
        directions_dst = get_ray_directions_360(project_pts[:,0], project_pts[:,1], W, H) # NDC direction from dst

        rays_o, rays_d = get_rays_lean(directions_dst, dst_cam2curr_rf)
        rays = torch.cat([rays_o, rays_d], -1).view(-1, 6)                    
        dst_rgb_map, dst_static_rgb_map, dst_depth_map, dst_dyn_map, dst_T = self.rf_refine_forward_uv(
            self.tensorfs[curr_rf_ids],
            rays,
            is_train=True,
            dst_ray_ids=dst_ray_ids,
            uv=project_pts,
            N_samples=-1,
            refine=self.is_refining
        )             

        if self.lr_exposure_init > 0:
            dst_exposure = torch.stack(list(self.exposure), dim=0)[frame_dst.tolist()]                        
            dst_exposure = dst_exposure.repeat_interleave(ray_ids.shape[0] // view_ids.shape[0], dim=0)
            dst_rgb_map = torch.bmm(dst_exposure, dst_static_rgb_map[..., None])[..., 0]
            dst_rgb_map = (dst_rgb_map + dst_dyn_map[:,:3] * dst_dyn_map[:,3:]).clamp(0,1)
            dst_rgb_map = dst_rgb_map.clamp(0, 1)

        dst_cam2prev_rf = self.get_cam2world(frame_dst).clone()
        dst_cam2prev_rf[:, :3, 3] += self.world2rf[prev_rf_ids] ## Move camera orientation with respect to the reference RF
        dst_cam2prev_rf = dst_cam2prev_rf.repeat_interleave(dst_ray_ids.shape[0] // frame_dst.shape[0], dim=0)
        rays_o, rays_d = get_rays_lean(directions_dst, dst_cam2prev_rf)
        rays = torch.cat([rays_o, rays_d], -1).view(-1, 6) 

        prev_dst_depth_map = self.rf_forward_depth(
            self.tensorfs[prev_rf_ids],
            rays,
            is_train=True,
            N_samples=-1,
            refine=self.is_refining
        )
                        
        dst_depth_map = dst_depth_map.view(-1)
        prev_dst_depth_map = prev_dst_depth_map.view(-1)
        valid_ray_mask = ( (prev_dst_depth_map * (1-forward_depth_margin) < dst_depth_map)
                            & (prev_dst_depth_map * (1+forward_depth_margin) > dst_depth_map)
                            & (dst_depth_map < forward_depth_thres)
                            )

        return dst_rgb_map, dst_rgbs, dst_static_rgb_map, dst_depth_map, dst_dyn_map, valid_ray_mask

    def backward_step(
        self,
        directions,
        depth_map,
        refine_cam2world,
        curr_rf_ids,
        prev_rf_ids,
        frame_src,
        frame_dst,
        W,
        H,
        train_dataset,
        backward_depth_margin,
        backward_depth_thres,
        ):        
        directions = directions.view(frame_dst.shape[0], -1, 3) # NDC direction from src          
        depth_map = depth_map.view(frame_dst.shape[0], -1)
        back_pts = directions * depth_map[..., None]  ## pts at NDC of src frame

        back_src2dst_cam = get_cam2cams(refine_cam2world, frame_dst, frame_src-frame_dst)   
        back_project_pts = proj_360(back_pts, back_src2dst_cam, W, H).detach() ## floating points pixels coord. on src
        back_dst_rgbs, back_dst_ray_ids, back_dst_loss_weights = train_dataset.bilerp(project_pts=back_project_pts, frame_dst=frame_src)
        
        back_dst_cam2prev_rf = self.get_cam2world(frame_src).clone()
        back_dst_cam2prev_rf[:, :3, 3] += self.world2rf[prev_rf_ids] ## Move camera orientation with respect to the reference RF
        back_dst_cam2prev_rf = back_dst_cam2prev_rf.repeat_interleave(back_dst_ray_ids.shape[0] // frame_src.shape[0], dim=0)
        back_project_pts = back_project_pts.view(-1,2)
        back_directions_dst = get_ray_directions_360(back_project_pts[:,0], back_project_pts[:,1], W, H) # NDC direction from dst

        rays_o, rays_d = get_rays_lean(back_directions_dst, back_dst_cam2prev_rf)
        rays = torch.cat([rays_o, rays_d], -1).view(-1, 6)    
        back_prev_dst_rgb_map, back_prev_dst_static_rgb_map, back_prev_dst_depth_map, back_prev_dst_dyn_map, back_prev_T = self.rf_refine_forward_uv(
            self.tensorfs[prev_rf_ids],
            rays,
            is_train=True,
            dst_ray_ids=back_dst_ray_ids,
            uv=back_project_pts,
            N_samples=-1,
            refine=self.is_refining
        )  
        
        if self.lr_exposure_init > 0:
            back_dst_exposure = torch.stack(list(self.exposure), dim=0)[frame_src.tolist()]   
            back_dst_exposure = back_dst_exposure.repeat_interleave(back_dst_ray_ids.shape[0] // frame_src.shape[0], dim=0)
            back_prev_dst_rgb_map = torch.bmm(back_dst_exposure, back_prev_dst_static_rgb_map[..., None])[..., 0]
            back_prev_dst_rgb_map = (back_prev_dst_rgb_map + back_prev_dst_dyn_map[:,:3] * back_prev_dst_dyn_map[:,3:]).clamp(0,1)

        back_dst_cam2rf = self.get_cam2world(frame_src).clone()
        back_dst_cam2rf[:, :3, 3] += self.world2rf[curr_rf_ids] ## Move camera orientation with respect to the reference RF
        back_dst_cam2rf = back_dst_cam2rf.repeat_interleave(back_dst_ray_ids.shape[0] // frame_src.shape[0], dim=0)
        rays_o, rays_d = get_rays_lean(back_directions_dst, back_dst_cam2rf)
        rays = torch.cat([rays_o, rays_d], -1).view(-1, 6) 

        back_dst_depth_map = self.rf_forward_depth(
            self.tensorfs[curr_rf_ids],
            rays,
            is_train=True,
            N_samples=-1,
            refine=self.is_refining
        )
        
        back_dst_depth_map = back_dst_depth_map.view(-1)
        back_prev_dst_depth_map = back_prev_dst_depth_map.view(-1)
        valid_ray_mask = ( (back_prev_dst_depth_map > back_dst_depth_map * (1 - backward_depth_margin))
                            & (back_prev_dst_depth_map < back_dst_depth_map * (1 + backward_depth_margin))
                            & (back_dst_depth_map < backward_depth_thres)
                            )           
        
        return back_prev_dst_rgb_map, back_dst_rgbs, back_prev_dst_static_rgb_map, back_prev_dst_depth_map, back_prev_dst_dyn_map, back_prev_T, valid_ray_mask, back_dst_loss_weights