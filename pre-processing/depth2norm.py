import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

def d2n_tblr(points: torch.Tensor, 
             k: int = 3, 
             d_min: float = 1e-3, 
             d_max: float = 256.0) -> torch.Tensor:
    """ points:     3D points in camera coordinates, shape: (B, 3, H, W)
        k:          neighborhood size
            e.g.)   If k=3, 3x3 neighborhood is used. Two vectors are defined by doing (top-bottom) and (left-right) 
                    Then the normals are computed via cross-product
        d_min/max:  Range of valid depth values 
    """
    k = (k - 1) // 2

    B, _, H, W = points.size()
    points_pad = F.pad(points, (k,k,k,k), mode='constant', value=0)             # (B, 3, k+H+k, k+W+k)
    valid_pad = (points_pad[:,2:,:,:] > d_min) & (points_pad[:,2:,:,:] < d_max) # (B, 1, k+H+k, k+W+k)
    valid_pad = valid_pad.float()

    # vertical vector (top - bottom)
    vec_vert = points_pad[:, :, :H, k:k+W] - points_pad[:, :, 2*k:2*k+H, k:k+W]   # (B, 3, H, W)

    # horizontal vector (left - right)
    vec_hori = points_pad[:, :, k:k+H, :W] - points_pad[:, :, k:k+H, 2*k:2*k+W]   # (B, 3, H, W)

    # valid_mask (all five depth values - center/top/bottom/left/right should be valid)
    valid_mask = valid_pad[:, :, k:k+H,     k:k+W       ] * \
                 valid_pad[:, :, :H,        k:k+W       ] * \
                 valid_pad[:, :, 2*k:2*k+H, k:k+W       ] * \
                 valid_pad[:, :, k:k+H,     :W          ] * \
                 valid_pad[:, :, k:k+H,     2*k:2*k+W   ]
    valid_mask = valid_mask > 0.5
    
    # get cross product (B, 3, H, W)
    cross_product = - torch.linalg.cross(vec_vert, vec_hori, dim=1)
    normal = F.normalize(cross_product, p=2.0, dim=1, eps=1e-12)
   
    return normal, valid_mask

def normal_to_rgb(normal, normal_mask=None):
    """ surface normal map to RGB
        (used for visualization)

        NOTE: x, y, z are mapped to R, G, B
        NOTE: [-1, 1] are mapped to [0, 255]
    """
    if torch.is_tensor(normal):
        normal = normal.permute(0, 2, 3, 1).cpu().numpy()
        normal_mask = normal_mask.permute(0, 2, 3, 1).cpu().numpy() if normal_mask is not None else None

    normal_norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal_norm[normal_norm < 1e-12] = 1e-12
    normal = normal / normal_norm

    normal_rgb = (((normal + 1) * 0.5) * 255).astype(np.uint8)
    if normal_mask is not None:
        normal_rgb = normal_rgb * normal_mask     # (B, H, W, 3)
    return normal_rgb

def intrins_to_intrins_inv(intrins):
    """ intrins to intrins_inv

        NOTE: top-left is (0,0)
    """
    if torch.is_tensor(intrins):
        intrins_inv = torch.zeros_like(intrins)
    elif type(intrins) is np.ndarray:
        intrins_inv = np.zeros_like(intrins)
    else:
        raise Exception('intrins should be torch tensor or numpy array')

    intrins_inv[0, 0] = 1 / intrins[0, 0]
    intrins_inv[0, 2] = - intrins[0, 2] / intrins[0, 0]
    intrins_inv[1, 1] = 1 / intrins[1, 1]
    intrins_inv[1, 2] = - intrins[1, 2] / intrins[1, 1]
    intrins_inv[2, 2] = 1.0
    return intrins_inv

def get_cam_coords(intrins_inv, depth):
    """ camera coordinates from intrins_inv and depth
    
        NOTE: intrins_inv should be a torch tensor of shape (B, 3, 3)
        NOTE: depth should be a torch tensor of shape (B, 1, H, W)
        NOTE: top-left is (0,0)
    """
    assert torch.is_tensor(intrins_inv) and intrins_inv.ndim == 3
    assert torch.is_tensor(depth) and depth.ndim == 4
    assert intrins_inv.dtype == depth.dtype
    assert intrins_inv.device == depth.device
    B, _, H, W = depth.size()

    u_range = torch.arange(W, dtype=depth.dtype, device=depth.device).view(1, W).expand(H, W) # (H, W)
    v_range = torch.arange(H, dtype=depth.dtype, device=depth.device).view(H, 1).expand(H, W) # (H, W)
    ones = torch.ones(H, W, dtype=depth.dtype, device=depth.device)
    pixel_coords = torch.stack((u_range, v_range, ones), dim=0).unsqueeze(0).repeat(B,1,1,1)  # (B, 3, H, W)
    pixel_coords = pixel_coords.view(B, 3, H*W)  # (B, 3, H*W)

    cam_coords = intrins_inv.bmm(pixel_coords).view(B, 3, H, W)
    cam_coords = cam_coords * depth
    return cam_coords

def main():
    # 创建保存法向量的文件夹
    norms_folder = r'F:\Dataset\3DRealCAR\sample_data\car_01\norms'
    if not os.path.exists(norms_folder):
        os.makedirs(norms_folder)

    # 读取相机内参
    intrinsics = np.load(r'F:\Dataset\3DRealCAR\sample_data\car_01\intrinsics.npy', allow_pickle=True).item()['FRONT'].astype(np.float64)
    intrinsics = intrinsics[0]
    
    # 将内参转换为torch张量
    intrinsics = torch.tensor(intrinsics, dtype=torch.float32)
    intrins_inv = intrins_to_intrins_inv(intrinsics).unsqueeze(0)  # (1, 3, 3)
    
    # 读取深度图文件
    depths_folder = r'F:\Dataset\3DRealCAR\sample_data\car_01\depths'
    depth_files = sorted([f for f in os.listdir(depths_folder) if f.endswith('.npy')])

    for depth_file in tqdm(depth_files):
        depth_path = os.path.join(depths_folder, depth_file)
        depth = np.load(depth_path).astype(np.float32)
        depth = torch.tensor(depth).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        # 计算相机坐标
        cam_coords = get_cam_coords(intrins_inv, depth)
        
        # 计算法向量
        normal, valid_mask = d2n_tblr(cam_coords)
        normal = normal.squeeze(0)  # 从 (1, 3, H, W) 转换为 (3, H, W)

        # 保存法向量
        norm_save_path = os.path.join(norms_folder, depth_file)
        np.save(norm_save_path, normal.cpu().numpy())

if __name__ == '__main__':
    main()
