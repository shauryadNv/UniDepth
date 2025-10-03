# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys

# Ensure current directory is on path to import local packages
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)

import argparse
import torch
import tqdm
import os.path as osp
import numpy as np
import logging
from PIL import Image
import glob
import cv2
import imageio
# import open3d as o3d

from unidepth.models import UniDepthV2
from unidepth.utils.camera import Pinhole


def write_pfm(data: np.array, fpath: str, scale=1, file_identifier=b'Pf', dtype="float32"):
    """
    Writes pfm file to disk.

    Args:
        data (np.array): data to save to pfm.
        fpath (str): path to save pfm output.
        scale (int, optional): scaling factor. Defaults to 1.
        file_identifier (bytes, optional): PF=color, Pf=grayscale. Defaults to b'Pf'.
        dtype (str, optional): data type. Defaults to "float32".
    """
    # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html

    data = np.flipud(data)
    height, width = data.shape  # For 2D disparity data
    values = np.ndarray.flatten(np.asarray(data, dtype=dtype))
    endianess = data.dtype.byteorder

    if endianess == '<' or (endianess == '=' and sys.byteorder == 'little'):
        scale *= -1

    with open(fpath, 'wb') as file:
        file.write((file_identifier))
        file.write(('\n%d %d\n' % (width, height)).encode())
        file.write(('%d\n' % scale).encode())
        file.write(values)


def is_image_completed(resume_dir, image_name_no_ext):
    """Check if an image has already been processed (resume mode)."""
    if resume_dir is None:
        return False
    completion_file = osp.join(resume_dir, f"{image_name_no_ext}.txt")
    return osp.exists(completion_file)


def mark_image_completed(resume_dir, image_name_no_ext):
    """Mark an image as completed (resume mode)."""
    if resume_dir is None:
        return
    completion_file = osp.join(resume_dir, f"{image_name_no_ext}.txt")
    with open(completion_file, 'w') as f:
        f.write(f"Completed processing of image: {image_name_no_ext}\n")


def depth_to_uint8_encoding(depth, scale=1000):
    """
    Encode depth map as uint8 with 3 channels for better precision.
    
    Args:
        depth: Input depth map
        scale: Scaling factor for depth values
    """
    depth = depth * scale
    H, W = depth.shape
    out = np.zeros((H, W, 3), dtype=float)
    out[..., 0] = depth // (255 * 255)
    out[..., 1] = (depth - out[..., 0] * 255 * 255) // 255
    out[..., 2] = depth - out[..., 0] * 255 * 255 - out[..., 1] * 255

    if not (out[..., 2] <= 255).all():
        print(f"Min:{out[..., 2].min()}, max:{out[..., 2].max()}")

    return out.astype(np.uint8)


def depth_to_normals(depth: np.ndarray,
                     fx: float, fy: float, cx: float, cy: float,
                     base_valid_mask: np.ndarray = None,
                     gaussian_sigma: float = 1.0):
    """
    Convert a depth map into a surface normal map using finite differences in 3D.
    Returns normals (H,W,3) in unit vectors and a validity mask.
    
    Args:
        depth: Input depth map
        fx, fy, cx, cy: Camera intrinsics
        base_valid_mask: Optional validity mask for depth values
        gaussian_sigma: Gaussian blur sigma for depth smoothing before normal computation
    """
    H, W = depth.shape
    # Valid depth mask
    if base_valid_mask is None:
        valid = np.isfinite(depth) & (depth > 0)
    else:
        valid = base_valid_mask & np.isfinite(depth) & (depth > 0)

    # Apply Gaussian smoothing to depth before computing normals
    depth_smoothed = depth.astype(np.float64)
    if gaussian_sigma is not None and gaussian_sigma > 0:
        depth_smoothed = cv2.GaussianBlur(depth_smoothed, (0, 0), gaussian_sigma)

    # Build 3D points from smoothed depth
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    Z = depth_smoothed
    X = (us - cx) * Z / fx
    Y = (vs - cy) * Z / fy

    # Derivatives along x (axis=1) and y (axis=0)
    dX_du = np.gradient(X, axis=1)
    dY_du = np.gradient(Y, axis=1)
    dZ_du = np.gradient(Z, axis=1)
    dX_dv = np.gradient(X, axis=0)
    dY_dv = np.gradient(Y, axis=0)
    dZ_dv = np.gradient(Z, axis=0)

    # Tangent vectors
    t_u = np.stack([dX_du, dY_du, dZ_du], axis=-1)
    t_v = np.stack([dX_dv, dY_dv, dZ_dv], axis=-1)

    # Normal via cross product (orientation not critical; we normalize)
    n = np.cross(t_v, t_u)

    # Normalize
    norm = np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    n_unit = n / norm

    # Valid normals where depth valid and norm magnitude reasonable
    normals_valid = valid & np.isfinite(n_unit).all(axis=-1)

    # Fill invalid normals with zero
    n_unit[~normals_valid] = 0.0
    return n_unit, normals_valid


def normal_float_to_uint8(normal_map, valid_mask=None):
    """
    Convert surface normals to uint8.
    Maps normals from [-1,1] to [0,1] for each channel.
    """
    img = ((normal_map + 1.) / 2.0) * 255
    img = np.clip(img, 0, 255).astype(np.uint8)

    if valid_mask is not None:
        img[~valid_mask] = 0

    return img


def set_logging_format(level=logging.INFO):
    import importlib
    importlib.reload(logging)
    FORMAT = '%(message)s'
    logging.basicConfig(level=level, format=FORMAT, datefmt='%m-%d|%H:%M:%S')


def set_seed(random_seed):
    import random
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def toOpen3dCloud(points, colors=None, normals=None):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        if colors.max() > 1:
            colors = colors / 255.0
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    return cloud


def depth2xyzmap(depth: np.ndarray, K, uvs: np.ndarray = None, zmin=0.1):
    invalid_mask = (depth < zmin)
    H, W = depth.shape[:2]
    if uvs is None:
        vs, us = np.meshgrid(np.arange(0, H), np.arange(0, W), sparse=False, indexing='ij')
        vs = vs.reshape(-1)
        us = us.reshape(-1)
    else:
        uvs = uvs.round().astype(int)
        us = uvs[:, 0]
        vs = uvs[:, 1]
    zs = depth[vs, us]
    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    pts = np.stack((xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)), 1)  # (N,3)
    xyz_map = np.zeros((H, W, 3), dtype=np.float32)
    xyz_map[vs, us] = pts
    if invalid_mask.any():
        xyz_map[invalid_mask] = 0
    return xyz_map


def compute_intrinsics_from_hfov(width: int, height: int, hfov_deg: float):
    fx = width / np.tan(np.radians(hfov_deg / 2.0)) / 2.0
    fy = height / np.tan(np.radians(hfov_deg / 2.0)) / 2.0 * width / height
    cx = 0.5 * width
    cy = 0.5 * height
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    return K


@torch.no_grad()
def inference_images_depth(model, args, device=torch.device('cuda')):
    input_dir = args.input_dir

    # Determine images directory: prefer 'left' subfolder if it exists, otherwise use input_dir directly
    left_dir = osp.join(input_dir, 'left')
    images_dir = left_dir if osp.exists(left_dir) and osp.isdir(left_dir) else input_dir

    # Get all image files from images directory
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(osp.join(images_dir, ext)))
        image_paths.extend(glob.glob(osp.join(images_dir, ext.upper())))

    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")

    image_paths = sorted(image_paths)

    # Create output directories (mirror UniDepth video script)
    output_dir = args.output_dir
    output_rgb_dir = osp.join(output_dir, 'rgb')
    output_depth_dir = osp.join(output_dir, 'depth')
    output_depth_vis_dir = osp.join(output_dir, 'depth_vis') if args.save_depth_vis else None
    output_pointcloud_dir = osp.join(output_dir, 'pointcloud') if args.save_pointcloud else None
    output_normal_maps_dir = osp.join(output_dir, 'normal_maps') if not args.disable_normal_maps else None
    resume_dir = osp.join(output_dir, 'resume') if args.resume else None

    os.makedirs(output_rgb_dir, exist_ok=True)
    os.makedirs(output_depth_dir, exist_ok=True)
    if args.save_depth_vis:
        os.makedirs(output_depth_vis_dir, exist_ok=True)
    if args.save_pointcloud:
        os.makedirs(output_pointcloud_dir, exist_ok=True)
    if not args.disable_normal_maps:
        os.makedirs(output_normal_maps_dir, exist_ok=True)
    if args.resume:
        os.makedirs(resume_dir, exist_ok=True)

    hfov_deg = 60

    processed_count = 0
    resumed_count = 0

    for image_path in tqdm.tqdm(image_paths, desc="Processing images"):
        image_name = osp.basename(image_path)
        image_name_no_ext = osp.splitext(image_name)[0]

        # Read image
        try:
            img = np.array(Image.open(image_path))
        except Exception as e:
            print(f"Error reading image {image_name}: {e}")
            continue

        # Resume check
        if args.resume and is_image_completed(resume_dir, image_name_no_ext):
            print(f"Skipping already processed image: {image_name}")
            resumed_count += 1
            continue

        # Apply scaling if needed
        scale = args.scale
        assert scale <= 1, "scale must be <=1"
        if scale != 1.0:
            img = cv2.resize(img, fx=scale, fy=scale, dsize=None)

        H, W = img.shape[:2]

        # Save input image
        rgb_output_path = osp.join(output_rgb_dir, image_name)
        Image.fromarray(img).save(rgb_output_path)

        # Prepare tensor (C, H, W)
        image_tensor = torch.from_numpy(img).permute(2, 0, 1)
        if torch.cuda.is_available() and next(model.parameters()).is_cuda:
            image_tensor = image_tensor.cuda()

        # Camera intrinsics and inference
        intrinsics = compute_intrinsics_from_hfov(W, H, hfov_deg)
        camera = Pinhole(K=torch.from_numpy(intrinsics))
        prediction = model.infer(image_tensor, camera=camera)
        depth = prediction["depth"][0, 0]

        # To numpy
        depth_np = depth.detach().cpu().numpy() if isinstance(depth, torch.Tensor) else depth

        # Save depth - either as uint8 PNG or PFM
        if args.depth_as_uint8:
            depth_path = osp.join(output_depth_dir, f'{image_name_no_ext}.png')
            depth_uint8 = depth_to_uint8_encoding(depth_np)
            imageio.imwrite(depth_path, depth_uint8)
        else:
            depth_path = osp.join(output_depth_dir, f'{image_name_no_ext}.pfm')
            write_pfm(depth_np, depth_path)

        # Save depth visualization if enabled
        if args.save_depth_vis:
            depth_vis_path = osp.join(output_depth_vis_dir, f'{image_name_no_ext}.png')
            inv_depth = 1.0 / np.clip(depth_np, 1e-6, None)
            norm = (inv_depth - inv_depth.min()) / (inv_depth.max() - inv_depth.min() + 1e-12)
            depth_vis = (norm.clip(0, 1) * 255).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)[..., ::-1]
            imageio.imwrite(depth_vis_path, depth_vis)

        # Save point cloud if enabled
        if args.save_pointcloud:
            xyz_map = depth2xyzmap(depth_np, intrinsics)
            pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), img.reshape(-1, 3))

            # Filter by depth range
            keep_mask = (np.asarray(pcd.points)[:, 2] > 0) & (np.asarray(pcd.points)[:, 2] <= args.z_far)
            keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
            pcd = pcd.select_by_index(keep_ids)

            pointcloud_path = osp.join(output_pointcloud_dir, f'{image_name_no_ext}.ply')
            o3d.io.write_point_cloud(pointcloud_path, pcd)

            if args.denoise_cloud:
                cl, ind = pcd.remove_radius_outlier(nb_points=args.denoise_nb_points, radius=args.denoise_radius)
                inlier_cloud = pcd.select_by_index(ind)
                pointcloud_denoised_path = osp.join(output_pointcloud_dir, f'{image_name_no_ext}_denoised.ply')
                o3d.io.write_point_cloud(pointcloud_denoised_path, inlier_cloud)

        # Generate and save normal maps if enabled
        if not args.disable_normal_maps:
            fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
            
            # Create validity mask for normal computation
            valid_depth_mask = (depth_np > 0) & np.isfinite(depth_np)
            
            # Compute surface normals from depth
            normals, normals_valid = depth_to_normals(
                depth_np, fx, fy, cx, cy, 
                base_valid_mask=valid_depth_mask,
                gaussian_sigma=args.normal_gaussian_sigma
            )
             
            if args.normals_as_float:
                # Save normal map as NPY file (preserves full precision)
                normal_map_path = osp.join(output_normal_maps_dir, f'{image_name_no_ext}.npy')
                np.save(normal_map_path, normals)
            else:
                # Save normal map as uint8 PNG
                normal_vis_rgb = normal_float_to_uint8(normals, normals_valid)
                normal_vis_path = osp.join(output_normal_maps_dir, f'{image_name_no_ext}.png')
                imageio.imwrite(normal_vis_path, normal_vis_rgb)

        torch.cuda.empty_cache()

        # Mark completed for resume
        if args.resume:
            mark_image_completed(resume_dir, image_name_no_ext)
        processed_count += 1

    print(f"\nProcessing complete!")
    print(f"Total images found: {len(image_paths)}")
    print(f"Successfully processed: {processed_count} images")
    if args.resume:
        print(f"Skipped (already completed): {resumed_count} images")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', help='Input directory containing images or a left subfolder', type=str)
    parser.add_argument('--output_dir', help='Output directory to save results', type=str, default='unidepth_dataset/processed')
    parser.add_argument('--scale', default=1, type=float, help='Downsize the image by scale, must be <=1')
    parser.add_argument('--model', help='Model version', type=str, default='unidepth-v2-vitl14')

    # Arguments for depth visualization and point cloud generation
    parser.add_argument('--save_depth_vis', action='store_true', help='Save depth visualization as PNG')
    parser.add_argument('--save_pointcloud', action='store_true', help='Save point cloud output as PLY')
    parser.add_argument('--z_far', default=10, type=float, help='Max depth to clip in point cloud')
    parser.add_argument('--denoise_cloud', action='store_true', help='Whether to denoise the point cloud')
    parser.add_argument('--denoise_nb_points', type=int, default=30, help='Number of points to consider for radius outlier removal')
    parser.add_argument('--denoise_radius', type=float, default=0.03, help='Radius to use for outlier removal')
    
    # Arguments for uint8 depth and normal map generation
    parser.add_argument('--depth_as_uint8', action='store_true', help='Save depth map as uint8 PNG instead of PFM')
    parser.add_argument('--disable_normal_maps', action='store_true', help='Disable normal map generation (enabled by default)')
    parser.add_argument('--normal_gaussian_sigma', type=float, default=2.0, help='Gaussian blur sigma for depth smoothing before normal computation (0 to disable)')
    parser.add_argument('--normals_as_float', action='store_true', help='Save normal map as float in NPY (default is uint8 PNG)')

    # Resume support
    parser.add_argument('--resume', action='store_true', help='Enable resume mode - skip already processed images')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if args.input_dir is None:
        raise ValueError("--input_dir must be provided")

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load UniDepthV2 model (mirrors UniDepth video script)
    logging.info("Loading UniDepthV2 model...")
    model = UniDepthV2.from_pretrained(f"lpiccinelli/{args.model}")

    if torch.cuda.is_available():
        model = model.cuda()
        logging.info("Model moved to CUDA")
    else:
        logging.info("CUDA not available, using CPU")

    model.eval()

    inference_images_depth(model, args, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


if __name__ == '__main__':
    main() 

