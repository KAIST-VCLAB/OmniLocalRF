# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2022 Anpei Chen

import numpy as np
import torch
from kornia import create_meshgrid
from torch import searchsorted

def contract(x):
    x_norm = torch.norm(x, dim=-1, keepdim=True)
    z = torch.where(x_norm <= 1, x, ((2 * x_norm - 1) / (x_norm**2 + 1e-6)) * x)
    return z

def get_ray_directions_lean(i, j, focal, center):
    '''
    get_ray_directions but returns only relevant rays
    Inputs:
        focal: (2), focal length
    Outputs:
        directions: (b, 3), the direction of the rays in camera coordinate
    '''
    i, j = i.float() + 0.5, j.float() + 0.5
    directions = torch.stack([(i - center[0]) / focal, -(j - center[1]) / focal, -torch.ones_like(i)], -1)  # (b, 3)
    return directions

def sphere2xyz(r, theta, phi):
    x = torch.cos(phi) * torch.sin(theta)
    y = torch.sin(phi)
    z = torch.cos(phi) * torch.cos(theta)
    return torch.stack([r*x, r*y, r*z], axis=-1)

def get_ray_directions_360(i, j, W, H):
    i, j = i.float() + 0.5, j.float() + 0.5
    phi = j * torch.pi / H - torch.pi / 2.
    theta = i * 2. * torch.pi / W + torch.pi
    directions = sphere2xyz(torch.ones_like(theta), theta, phi)
    return directions

def get_rays_lean(directions, c2w):
    '''
    get_rays but returns only relevant rays
    Inputs:
        directions: (B, 3) ray directions in camera coordinate
        c2w: (B, 3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (B, 3), the origin of the rays in world coordinate
        rays_d: (B, 3), the normalized direction of the rays in world coordinate
    '''
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, :3, 3]  # (B, 3)
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = torch.bmm(c2w[:, :3, :3], directions[..., None])[..., 0]  # (B, 3)

    return rays_o, rays_d

def patch_sampling(W, H, batch_size, device):    
    '''
    Sampling the rays by 2x2 patch shape
    '''
    idx = torch.randint(0, H*W, size = (1,batch_size//4), dtype=torch.int64, device=device)[0]            
    idx = idx.repeat_interleave(4)
    lower_idx = idx[1::4]        
    lower_idx[((lower_idx // W) % H)<(H-1)] += W
    right_idx = idx[2::4]
    right_idx[(right_idx % W) < (W-1)] += 1
    lower_right_idx = idx[3::4]
    lower_right_idx_cd  = ((lower_right_idx % W) < (W-1)) & (((lower_right_idx // W) % H)<(H-1))
    lower_right_idx[lower_right_idx_cd] += (W+1)        
    return idx

    
def patch_sampling_np(W, H, batch_size):    
    '''
    Sampling the rays by 2x2 patch shape
    '''
    idx = np.random.randint(0, H*W, batch_size//4, dtype=np.int64)
    idx = np.repeat(idx, 4)
    lower_idx = idx[1::4]        
    lower_idx[((lower_idx // W) % H)<(H-1)] += W
    right_idx = idx[2::4]
    right_idx[(right_idx % W) < (W-1)] += 1
    lower_right_idx = idx[3::4]
    lower_right_idx_cd  = ((lower_right_idx % W) < (W-1)) & (((lower_right_idx // W) % H)<(H-1))
    lower_right_idx[lower_right_idx_cd] += (W+1)  
    return idx