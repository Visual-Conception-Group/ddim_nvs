import torch, random, datasets, math, fastcore.all as fc, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision.transforms.functional as TF,torch.nn.functional as F
from torch.utils.data import random_split

from torch.utils.data import DataLoader,default_collate,Dataset
from pathlib import Path
from torch import nn,tensor
from torch.nn import init
from fastcore.foundation import L
from datasets import load_dataset
from operator import itemgetter,attrgetter
from functools import partial,wraps
from torch.optim import lr_scheduler
from torch import optim
from torchvision.io import read_image,ImageReadMode
from glob import glob
import os
from PIL import Image
from itertools import zip_longest
import warnings
warnings.filterwarnings('ignore')
from torch.utils.data import Subset
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
    DDIMInverseScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    UNet2DModel
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from transformers import AutoProcessor, CLIPModel
from transformers import CLIPTextModel, CLIPTokenizer
## Initiating tokenizer and encoder.
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16)
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to("cuda")
from tqdm import tqdm 
from scipy.stats import pearsonr
import signal
import pickle
import pandas as pd
import json

from typing import Optional
import os
import torch
import numpy as np
import pandas as pd
import random
from pathlib import Path
from torch.utils.data import Dataset
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.unets.unet_2d_blocks import (
    DownBlock2D,
    AttnDownBlock2D,
    CrossAttnDownBlock2D,
    ResnetBlock2D,
    UNetMidBlock2DCrossAttn,
    UpBlock2D,
    AttnUpBlock2D,
    CrossAttnUpBlock2D
)
from diffusers.models.attention import Attention as CrossAttention, FeedForward, AdaLayerNorm
from diffusers import UNet2DConditionModel
from torch import nn
from diffusers.models.unets.unet_2d import UNet2DOutput
import types
import torch
import torch.nn as nn
from typing import Optional
import logging
import inspect
from diffusers.models.attention_processor import AttnProcessor
import inspect
from dataclasses import dataclass
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, deprecate, logging, scale_lora_layers, unscale_lora_layers
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from types import MethodType
import lpips
import torch.cuda.amp as amp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import csv
# import torcheval
from torcheval.metrics import PeakSignalNoiseRatio
from torcheval.metrics.functional import peak_signal_noise_ratio


@fc.delegates(plt.Axes.imshow)
def show_image(im, ax=None, figsize=None, title=None, noframe=True, **kwargs):
    "Show a PIL or PyTorch image on `ax`."
    if fc.hasattrs(im, ('cpu','permute','detach')):
        im = im.detach().cpu()
        if len(im.shape)==3 and im.shape[0]<5: im=im.permute(1,2,0)
    elif not isinstance(im,np.ndarray): im=np.array(im)
    if im.shape[-1]==1: im=im[...,0]
    if ax is None: _,ax = plt.subplots(figsize=figsize)
    ax.imshow(im, **kwargs)
    if title is not None: ax.set_title(title)
    ax.set_xticks([]) 
    ax.set_yticks([]) 
    if noframe: ax.axis('off')
    return ax
@fc.delegates(plt.subplots, keep=True)
def subplots(
    nrows:int=1, # Number of rows in returned axes grid
    ncols:int=1, # Number of columns in returned axes grid
    figsize:tuple=None, # Width, height in inches of the returned figure
    imsize:int=3, # Size (in inches) of images that will be displayed in the returned figure
    suptitle:str=None, # Title to be set to returned figure
    **kwargs
): # fig and axs
    "A figure and set of subplots to display images of `imsize` inches"
    if figsize is None: figsize=(ncols*imsize, nrows*imsize)
    fig,ax = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    if suptitle is not None: fig.suptitle(suptitle)
    if nrows*ncols==1: ax = np.array([ax])
    return fig,ax

# %% ../nbs/05_datasets.ipynb 45
@fc.delegates(subplots)
def get_grid(
    n:int, # Number of axes
    nrows:int=None, # Number of rows, defaulting to `int(math.sqrt(n))`
    ncols:int=None, # Number of columns, defaulting to `ceil(n/rows)`
    title:str=None, # If passed, title set to the figure
    weight:str='bold', # Title font weight
    size:int=14, # Title font size
    **kwargs,
): # fig and axs
    "Return a grid of `n` axes, `rows` by `cols`"
    if nrows: ncols = ncols or int(np.floor(n/nrows))
    elif ncols: nrows = nrows or int(np.ceil(n/ncols))
    else:
        nrows = int(math.sqrt(n))
        ncols = int(np.floor(n/nrows))
    fig,axs = subplots(nrows, ncols, **kwargs)
    for i in range(n, nrows*ncols): axs.flat[i].set_axis_off()
    if title is not None: fig.suptitle(title, weight=weight, size=size)
    return fig,axs
@fc.delegates(subplots)
def show_images(ims:list, # Images to show
                nrows:Optional[int] = None, # Number of rows in grid
                ncols:Optional[int] = None, # Number of columns in grid (auto-calculated if None)
                titles:Optional[int] = None, # Optional list of titles for each image
                **kwargs):
    "Show all images `ims` as subplots with `rows` using `titles`"
    axs = get_grid(len(ims), nrows, ncols, **kwargs)[1].flat
    for im,t,ax in zip_longest(ims, titles or [], axs): show_image(im, ax=ax, title=t)

def decode_and_show_image(latents, vae):
    # Reverse the scaling applied during encoding
    latents = latents / 0.18215
    
    # Decode the latent vectors into images
    decoded_images = vae.decode(latents).sample
    
    # Reverse the normalization from [-1, 1] to [0, 1]
    decoded_images = (decoded_images + 1.) / 2.
    
    # Convert the tensor to a NumPy array and squeeze if necessary
    decoded_images = decoded_images.squeeze().detach().cpu().numpy()
    
    # Plot the decoded image
    plt.imshow(decoded_images.transpose(1, 2, 0))
    # plt.imshow(decoded_images)
    plt.axis('off')
    plt.show()

device = "cuda:0"

def normalize_data(cam_data):
    mean = cam_data.mean(dim=0, keepdim=True)
    std = cam_data.std(dim=0, keepdim=True)
    return (cam_data - mean) / std

def read_cameras_txt(path):
    cameras = {}
    with open(path, 'r') as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            parts = line.strip().split()
            camera_id = int(parts[0])
            model = parts[1]
            width, height = int(parts[2]), int(parts[3])
            params = list(map(float, parts[4:]))
            if model == "SIMPLE_RADIAL":
                # SIMPLE_RADIAL: fx, cx, cy, k (ignore distortion for now)
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            elif model == "PINHOLE":
                fx, fy, cx, cy = params[:4]
            else:
                raise NotImplementedError(f"Unsupported camera model: {model}")
            
            cameras[camera_id] = {
                "model": model,
                "width": width,
                "height": height,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy
            }
    return cameras

def read_images_txt(path):
    images = {}
    with open(path, 'r') as f:
        lines = f.readlines()

    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if line.startswith("#") or line == "":
            idx += 1
            continue

        parts = line.split()
        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        camera_id = int(parts[8])
        image_name = parts[9]

        idx += 1
        # Read the next line containing 2D keypoints (if present)
        if idx < len(lines) and not lines[idx].strip().startswith("#"):
            points2d_line = lines[idx].strip().split()
            points2d = []
            if len(points2d_line) >= 3:  # Make sure there are enough parts
                for i in range(0, len(points2d_line), 3):
                    if i + 2 < len(points2d_line):  # Check if we have 3 values available
                        x = float(points2d_line[i])
                        y = float(points2d_line[i+1])
                        point3D_id = int(points2d_line[i+2])
                        points2d.append((x, y, point3D_id))
            idx += 1
        else:
            points2d = []
            
        images[image_id] = {
            "qvec": np.array([qw, qx, qy, qz]),
            "tvec": np.array([tx, ty, tz]),
            "camera_id": camera_id,
            "image_name": image_name,
            "points2d": points2d
        }
    return images

def qvec2rotmat(qvec):
    qw, qx, qy, qz = qvec
    R = np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,      1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,      2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy]
    ])
    return R

def adjust_intrinsics_for_resize_and_crop(K_orig, orig_width, orig_height, crop_size=512):
    """
    Adjust the intrinsics after resizing such that the shorter side becomes crop_size,
    followed by center-cropping to (crop_size, crop_size).
    """
    # Compute resize scale
    scale = crop_size / min(orig_width, orig_height)
    
    K_resized = K_orig.copy()
    K_resized[:2] *= scale
    
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)
    
    # Center-crop offsets (for the longer dimension)
    offset_x = max(0, (new_width - crop_size) / 2)
    offset_y = max(0, (new_height - crop_size) / 2)
    K_resized[0, 2] -= offset_x
    K_resized[1, 2] -= offset_y
    
    return K_resized

def downscale_intrinsics(K_crop, crop_size=512, latent_size=64):
    """
    Adjust intrinsics for VAE latent resolution (e.g., from 512 to 64).
    """
    factor = crop_size / latent_size
    return K_crop / factor

def get_ray_directions(H, W, K):
    """
    Compute ray directions in camera space for an image of size (H,W) using intrinsic matrix K.
    Returns:
        directions: (H, W, 3) array
    """
    # Create a meshgrid of pixel coordinates
    i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    # Compute normalized directions: (x - cx) / fx, (y - cy) / fy, and 1
    dirs = np.stack([(i - K[0, 2]) / K[0, 0],
                     (j - K[1, 2]) / K[1, 1],
                     np.ones_like(i)], axis=-1)
    return dirs

def get_camera_rays_world(R_w2c, t_w2c, ray_directions):
    """
    Convert ray directions from camera coordinates into world coordinates.
    Args:
        R: (3,3) rotation matrix (from camera to world)
        T: (3,) translation vector (camera center in world coordinates)
        ray_directions: (H, W, 3) directions in camera space
    Returns:
        rays_o: (H, W, 3) ray origins (all equal to T)
        rays_d: (H, W, 3) ray directions in world coordinates
    """
    H, W, _ = ray_directions.shape
    # Rotate directions: note that R should map camera-space directions to world-space
    R_c2w = R_w2c.T
    C = - R_c2w @ t_w2c
    rays_d = ray_directions @ R_c2w
    rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)
    # The origin is the same for all rays (the camera center)
    rays_o = np.broadcast_to(C.reshape(1, 1, 3), (H, W, 3))
    return rays_o, rays_d

class ImagesDS_random_regenerate_means(Dataset):
    def __init__(self, csv_path, latents_source_path, mmshape, train_split=0.9, val_split=0.05, test_split=0.05, 
                split_log_file="scene_split_log.csv", mode="train", crop_size=512, latent_size=64, 
                rays_source_path=None, compute_rays=True):
        self.csv_path = csv_path
        self.latents_source_path = Path(latents_source_path)
        self.mmshape = mmshape
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.split_log_file = split_log_file
        self.mode = mode  # train, val, or test
        self.crop_size = crop_size
        self.latent_size = latent_size
        self.compute_rays = compute_rays
        self.rays_source_path = rays_source_path
        
        # Load CSV file
        self.data = pd.read_csv(self.csv_path)
        
        # Lazy loading of memory-mapped latents
        self.source_latents = np.memmap(self.latents_source_path, dtype=np.float32, mode='r', shape=self.mmshape)
        self.source_rays = None
        
        # Group images by scene
        self.image_data = {}
        for _, row in self.data.iterrows():
            image_path = row['image_path']
            class_name = str(row['class'])
            scene_id = Path(image_path).parts[-3]  # Extract scene ID from the path
            
            if class_name not in self.image_data:
                self.image_data[class_name] = {}
            if scene_id not in self.image_data[class_name]:
                self.image_data[class_name][scene_id] = []
            
            self.image_data[class_name][scene_id].append(image_path)
        
        # Cache for cameras and images data
        self.scene_camera_data = {}
        
        self.train_pairs = []
        self.val_pairs = []
        self.test_pairs = []
        self.train_scenes = {}
        self.val_scenes = {}
        self.test_scenes = {}
        self._filter_valid_scenes()

        self.split_scenes_lexicographically()
        self.generate_pairs()
        self.save_scene_distribution()
        
        # Precompute rays if needed and not already precomputed
        if compute_rays and rays_source_path is None:
            self.rays_shape = (len(self.data), self.latent_size, self.latent_size, 6)  # origins(3) + directions(3)
            self.rays_source_path = os.path.join(os.path.dirname(self.latents_source_path), "precomputed_rays.npy")
            if not os.path.exists(self.rays_source_path):
                self.precompute_rays_to_memmap()
    
    def _filter_valid_scenes(self):
        """Remove any scene that doesn't have a sparse/0 folder with both cameras.txt and images.txt."""
        to_remove = []
        for class_name, scenes in self.image_data.items():
            for scene_id in list(scenes.keys()):
                scene_path = Path(scenes[scene_id][0]).parent.parent
                sparse0    = scene_path / "sparse" / "0"
                cams_f     = sparse0 / "cameras.txt"
                imgs_f     = sparse0 / "images.txt"
                if not (sparse0.is_dir() and cams_f.is_file() and imgs_f.is_file()):
                    to_remove.append((class_name, scene_id))

        for class_name, scene_id in to_remove:
            print(f"Skipping scene {class_name}/{scene_id}: missing sparse/0 or files")
            del self.image_data[class_name][scene_id]
            
    def split_scenes_lexicographically(self):
        """Split scenes based on lexicographic ordering of scene names for each class."""
        self.train_scenes = {}
        self.val_scenes = {}
        self.test_scenes = {}
        
        for class_name, scenes in self.image_data.items():
            # Sort scene IDs lexicographically
            scene_ids = sorted(scenes.keys())
            total_scenes = len(scene_ids)
            
            if total_scenes <= 1:
                # If there's only one scene, put it in training
                self.train_scenes[class_name] = set(scene_ids)
                self.val_scenes[class_name] = set()
                self.test_scenes[class_name] = set()
                continue
                
            # Calculate split indices
            train_end = int(total_scenes * self.train_split)
            val_end = train_end + int(total_scenes * self.val_split)
            
            # Ensure we have at least one scene in each split if possible
            if train_end == 0 and total_scenes > 0:
                train_end = 1
            if val_end == train_end and total_scenes > 1:
                val_end = train_end + 1
                
            # Split according to lexicographic ordering
            self.train_scenes[class_name] = set(scene_ids[:train_end])
            self.val_scenes[class_name] = set(scene_ids[train_end:val_end])
            self.test_scenes[class_name] = set(scene_ids[val_end:])
    
    def generate_pairs(self):
        """Generate image pairs for training, validation, and testing."""
        self.train_pairs = []
        self.val_pairs = []
        self.test_pairs = []
        
        for class_name, scenes in self.image_data.items():
            for scene_id, image_paths in scenes.items():
                image_paths = sorted(image_paths)  # Ensure proper ordering
                
                if len(image_paths) < 25:
                    num_images = len(image_paths)
                    if num_images < 10:
                        print(f"Skipping scene {class_name}/{scene_id} due to insufficient images.")
                        continue
                    
                    mid_point = num_images // 2
                    input_images = image_paths[:mid_point]
                    target_images = image_paths[mid_point:]
                    
                    input_images_filtered = []
                    target_images_filtered = []
                    for i in range(len(input_images)):
                        if i + 5 < len(target_images):
                            input_images_filtered.append(input_images[i])
                            target_images_filtered.append(target_images[i + 5])
                    
                    for input_img, target_img in zip(input_images_filtered, target_images_filtered):
                        cam_data1 = self.parse_camera_metadata(input_img)
                        cam_data2 = self.parse_camera_metadata(target_img)
                        if cam_data1 and cam_data2:
                            cam_data1, cam_data2 = self.prepare_camera_data(cam_data1, cam_data2)
                            pair = (input_img, target_img, cam_data1, cam_data2, class_name)
                            
                            # Add to appropriate split based on scene ID
                            if scene_id in self.train_scenes[class_name]:
                                self.train_pairs.append(pair)
                            elif scene_id in self.val_scenes[class_name]:
                                self.val_pairs.append(pair)
                            elif scene_id in self.test_scenes[class_name]:
                                self.test_pairs.append(pair)
                    continue
                
                input_images = image_paths[:10]
                target_images = image_paths[14:25]
                
                shuffled_targets = random.sample(target_images, len(input_images))

                for input_img, target_img in zip(input_images, shuffled_targets):
                    cam_data1 = self.parse_camera_metadata(input_img)
                    cam_data2 = self.parse_camera_metadata(target_img)
                    if cam_data1 and cam_data2:
                        cam_data1, cam_data2 = self.prepare_camera_data(cam_data1, cam_data2)
                        pair = (input_img, target_img, cam_data1, cam_data2, class_name)
                        
                        # Add to appropriate split based on scene ID
                        if scene_id in self.train_scenes[class_name]:
                            self.train_pairs.append(pair)
                        elif scene_id in self.val_scenes[class_name]:
                            self.val_pairs.append(pair)
                        elif scene_id in self.test_scenes[class_name]:
                            self.test_pairs.append(pair)
    
    def save_scene_distribution(self):
        """Save scene distribution to a CSV file."""
        with open(self.split_log_file, 'w') as f:
            f.write("class,scene_id,split\n")
            
            for class_name in self.image_data.keys():
                # Log train scenes
                for scene_id in self.train_scenes.get(class_name, []):
                    f.write(f"{class_name},{scene_id},train\n")
                
                # Log val scenes
                for scene_id in self.val_scenes.get(class_name, []):
                    f.write(f"{class_name},{scene_id},val\n")
                
                # Log test scenes
                for scene_id in self.test_scenes.get(class_name, []):
                    f.write(f"{class_name},{scene_id},test\n")
        
        # Also save a summary file
        with open(self.split_log_file.replace('.csv', '_summary.csv'), 'w') as f:
            f.write("class,total_scenes,train_scenes,val_scenes,test_scenes\n")
            for class_name in self.image_data.keys():
                total_scenes = len(self.image_data[class_name])
                train_scenes = len(self.train_scenes.get(class_name, []))
                val_scenes = len(self.val_scenes.get(class_name, []))
                test_scenes = len(self.test_scenes.get(class_name, []))
                f.write(f"{class_name},{total_scenes},{train_scenes},{val_scenes},{test_scenes}\n")
    
    def parse_camera_metadata(self, image_path):
        scene_path = Path(image_path).parent.parent  # Move up to scene level
        metadata_path = scene_path / "sparse/0/camera_metadata.txt"
        if not metadata_path.exists():
            return None
        
        camera_data = {}
        with open(metadata_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) > 9:
                    image_name = parts[-1]
                    camera_data[image_name] = parts[1:-1]
        
        return camera_data.get(Path(image_path).name)
    
    def prepare_camera_data(self, cam_data1, cam_data2):
        cam_data1 = torch.tensor([float(s) for s in cam_data1], dtype=torch.float32)
        cam_data2 = torch.tensor([float(s) for s in cam_data2], dtype=torch.float32)
        cam_id1 = cam_data1[-1]
        cam_id2 = cam_data2[-1]
        cam_data1 = normalize_data(cam_data1[:-1])
        cam_data2 = normalize_data(cam_data2[:-1])
        cam_data1 = torch.cat([cam_data1, cam_id1.unsqueeze(0)])
        cam_data2 = torch.cat([cam_data2, cam_id2.unsqueeze(0)])
        return cam_data1, cam_data2
    
    def precompute_rays_to_memmap(self):
        """Precompute all rays and store in a memory-mapped file"""
        print("Starting ray precomputation...")
        start_time = time.time()
        
        # Create unique set of all image paths used in pairs
        all_images = set()
        for pairs in [self.train_pairs, self.val_pairs, self.test_pairs]:
            for input_img, target_img, *_ in pairs:
                all_images.add(input_img)
                all_images.add(target_img)
        
        # Create memory-mapped file for rays
        rays_file = self.rays_source_path
        rays_shape = self.rays_shape
        
        # Create the memory-mapped file
        rays_array = np.memmap(rays_file, dtype=np.float32, mode='w+', shape=rays_shape)
        
        # Track processed images and failures
        processed_count = 0
        failed_images = []
        
        # Process all images
        for img_path in all_images:
            # Get index in the original dataframe
            img_idx = self.data.index[self.data['image_path'] == img_path].tolist()
            if not img_idx:
                failed_images.append(img_path)
                continue
            img_idx = img_idx[0]
            
            # Compute rays
            origins, dirs = self.get_camera_rays_for_image(img_path)
            
            if origins is not None and dirs is not None:
                # Combine origins and directions into a single array
                rays = np.concatenate([origins.numpy(), dirs.numpy()], axis=-1)
                # Store in the memory-mapped file
                rays_array[img_idx] = rays
                processed_count += 1
            else:
                failed_images.append(img_path)
            
            # Clear cache and show progress periodically
            if processed_count % 100 == 0:
                rays_array.flush()
                self.scene_camera_data.clear()
                elapsed = time.time() - start_time
                print(f"Processed {processed_count}/{len(all_images)} images ({processed_count/len(all_images)*100:.1f}%) in {elapsed:.1f}s")
        
        # Final flush
        rays_array.flush()
        elapsed = time.time() - start_time
        print(f"Ray precomputation complete. Processed {processed_count} images in {elapsed:.1f}s")
        print(f"Failed to process {len(failed_images)} images")
        print(f"Rays saved to {rays_file}")
        
        # Write failed images to a log file
        if failed_images:
            with open("failed_ray_images.txt", "w") as f:
                for img in failed_images:
                    f.write(f"{img}\n")
    
    def get_camera_rays_for_image(self, image_path):
        """Get camera ray origins and directions for a given image using the provided functions."""
        scene_path = Path(image_path).parent.parent
        sparse_path = scene_path / "sparse/0"
        cameras_path = sparse_path / "cameras.txt"
        images_path = sparse_path / "images.txt"
        
        # Create a unique key for this scene
        scene_key = str(scene_path)
        
        # Cache camera and image data for each scene
        if scene_key not in self.scene_camera_data:
            if not cameras_path.exists() or not images_path.exists():
                return None, None
                
            cameras = read_cameras_txt(str(cameras_path))
            images = read_images_txt(str(images_path))
            self.scene_camera_data[scene_key] = (cameras, images)
        else:
            cameras, images = self.scene_camera_data[scene_key]
        
        # Find the image info by image name
        image_name = Path(image_path).name
        image_info = None
        for img_id, img_data in images.items():
            if img_data["image_name"] == image_name:
                image_info = img_data
                break
        
        if image_info is None:
            return None, None
        
        # Get camera info
        camera_id = image_info["camera_id"]
        camera = cameras.get(camera_id)
        
        if camera is None:
            return None, None
            
        # Create the camera intrinsic matrix
        K = np.array([
            [camera["fx"], 0, camera["cx"]],
            [0, camera["fy"], camera["cy"]],
            [0, 0, 1]
        ])
        
        # Get original image dimensions
        orig_width, orig_height = camera["width"], camera["height"]
        
        # Adjust intrinsics for preprocessing (resize and crop)
        K_adjusted = adjust_intrinsics_for_resize_and_crop(K, orig_width, orig_height, self.crop_size)
        
        # Downscale for latent resolution if needed
        K_latent = downscale_intrinsics(K_adjusted, self.crop_size, self.latent_size)
        
        # Convert quaternion to rotation matrix
        R_w2c = qvec2rotmat(image_info["qvec"])
        t_w2c = image_info["tvec"]
        
        # Calculate ray directions in camera space
        ray_directions = get_ray_directions(self.latent_size, self.latent_size, K_latent)
        
        # Get camera rays in world coordinates
        origins, directions = get_camera_rays_world(R_w2c, t_w2c, ray_directions)
        
        # Convert to torch tensors
        origins = torch.from_numpy(origins).float()
        directions = torch.from_numpy(directions).float()
        
        return origins, directions
    
    def __len__(self):
        """Return the length based on mode"""
        if self.mode == "train":
            return len(self.train_pairs)
        elif self.mode == "val":
            return len(self.val_pairs)
        elif self.mode == "test":
            return len(self.test_pairs)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
    
    def __getitem__(self, idx):
        """Return dataset pairs with camera rays"""
        if self.mode == "train":
            input_img_path, target_img_path, cam_data1, cam_data2, class_name = self.train_pairs[idx]
        elif self.mode == "val":
            input_img_path, target_img_path, cam_data1, cam_data2, class_name = self.val_pairs[idx]
        elif self.mode == "test":
            input_img_path, target_img_path, cam_data1, cam_data2, class_name = self.test_pairs[idx]
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        
        # Lazy load memory-mapped latents
        # if self.source_latents is None:
        #     self.source_latents = np.memmap(
        #         self.latents_source_path, 
        #         dtype=np.float32, 
        #         mode='r', 
        #         shape=self.mmshape
        #     )
        
        # Lazy load memory-mapped rays if available
        if self.compute_rays and self.source_rays is None and self.rays_source_path is not None:
            try:
                self.source_rays = np.memmap(
                    self.rays_source_path, 
                    dtype=np.float32, 
                    mode='r', 
                    shape=self.rays_shape
                )
            except Exception as e:
                print(f"Error loading rays file: {e}")
                self.source_rays = None
        
        # Find latents
        def find_latent(img_path):
            latent_idx = self.data.index[self.data['image_path'] == img_path].tolist()
            if not latent_idx:
                raise ValueError(f"Image path {img_path} not found in dataset.")
            return torch.from_numpy(self.source_latents[latent_idx[0]]).float()
        
        input_latent = find_latent(input_img_path)
        target_latent = find_latent(target_img_path)
        
        # Get rays
        input_rays = None
        target_rays = None
        
        if self.compute_rays:
            if self.source_rays is not None:
                # Get rays from memory-mapped file
                def find_rays(img_path):
                    ray_idx = self.data.index[self.data['image_path'] == img_path].tolist()
                    if not ray_idx:
                        return None
                    return torch.from_numpy(self.source_rays[ray_idx[0]]).float()
                
                input_rays = find_rays(input_img_path)
                target_rays = find_rays(target_img_path)
            # else:
            #     # Fall back to computing rays on-the-fly
            #     input_origins, input_dirs = self.get_camera_rays_for_image(input_img_path)
            #     target_origins, target_dirs = self.get_camera_rays_for_image(target_img_path)
                
            #     if input_origins is not None and input_dirs is not None:
            #         input_rays = torch.cat([input_origins, input_dirs], dim=-1)
                
            #     if target_origins is not None and target_dirs is not None:
            #         target_rays = torch.cat([target_origins, target_dirs], dim=-1)
        
        return input_img_path, target_img_path, input_latent, target_latent, cam_data1, cam_data2,class_name, input_rays, target_rays 


csv_file = '/home/sehajs/mvimgnet_3_classes/dataset_image_data.csv'
latents_source_path = "/home/sehajs/mvimgnet_3_classes/all_images_means_512.npmm"
mmshape = (98558,4,64,64) 
images_ds_means = ImagesDS_random_regenerate_means(
        csv_path=csv_file,
        latents_source_path=latents_source_path,
        mmshape=mmshape,
        mode="train",
        crop_size=512,
        latent_size=64,
        compute_rays=True  # This will trigger precomputation if needed
        )
inverted_latent_camera_dl = DataLoader(images_ds_means, batch_size=32, shuffle=True, num_workers=4)
images_ds_means_val = ImagesDS_random_regenerate_means(csv_file,latents_source_path,mmshape,mode='val')
inverted_latent_camera_dl_val = DataLoader(images_ds_means_val, batch_size=32, shuffle=False, num_workers=4)

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.unets.unet_2d_blocks import (
    DownBlock2D,
    AttnDownBlock2D,
    CrossAttnDownBlock2D,
    ResnetBlock2D,
    UNetMidBlock2DCrossAttn,
    UpBlock2D,
    AttnUpBlock2D,
    CrossAttnUpBlock2D
)
from diffusers.models.attention import Attention as CrossAttention, FeedForward, AdaLayerNorm
from diffusers import UNet2DConditionModel
from torch import nn
from diffusers.models.unets.unet_2d import UNet2DOutput
import types
import torch
import torch.nn as nn
from typing import Optional
import logging
import inspect
from diffusers.models.attention_processor import AttnProcessor
import inspect
from dataclasses import dataclass
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, deprecate, logging, scale_lora_layers, unscale_lora_layers
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from types import MethodType

class CameraClassConditionedResnetBlock2DDownsample(ResnetBlock2D):
    """
    Custom ResnetBlock2D that incorporates camera parameters and class embeddings
    """
    def __init__(
        self,
        *args,
        num_classes=3,
        camera_params_dim=8,  # Dimension of raw camera parameters
        camera_emb_size=32,    # Size of camera embedding
        class_emb_size=32,    # Size of class embedding
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        # Create embedding networks for camera parameters
        self.camera_emb = nn.Linear(camera_params_dim, camera_emb_size)
        self.class_emb = nn.Embedding(num_classes, class_emb_size)
        # Projection layer to incorporate camera and class embeddings
        # This will project (time_emb_dim + camera_emb_size + class_emb_size) to time_emb_dim
        self.combined_emb_proj = nn.Linear(
            kwargs.get("temb_channels", 512) + camera_emb_size + class_emb_size, 
            kwargs.get("temb_channels", 512)
        )

    def forward(
        self, 
        input_tensor, 
        temb,
        camera_params_input=None,
        class_labels=None,
        *args, 
        **kwargs
    ):
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The scale argument is deprecated and will be ignored."
            logger.warning(deprecation_message)

        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)

        # Process temporal embedding as usual
        if temb is not None and self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            combined_features = [temb]    
            # Process camera parameters if provided
            # Process camera parameters if provided
            if camera_params_input is not None:
                camera_features = self.camera_emb(camera_params_input)
                combined_features.append(camera_features)
                # logger.info("using camera input params in custom down block resnet")
                
            # Process class labels if provided
            if class_labels is not None:
                class_emb = self.class_emb(class_labels.long())
                combined_features.append(class_emb)
                # logger.info("using camera class labels in custom down block resnet")
                
            # Combine all available embeddings
            if len(combined_features) > 1:
                combined_emb = torch.cat(combined_features, dim=1)
                # Project to original temporal embedding dimension
                temb = self.combined_emb_proj(combined_emb)
                # logger.info("using combined params in custom down block resnet")
            
            # Apply temporal embedding
            temb = self.time_emb_proj(temb)[:, :, None, None]

        if self.time_embedding_norm == "default":
            if temb is not None:
                hidden_states = hidden_states + temb
            hidden_states = self.norm2(hidden_states)
        elif self.time_embedding_norm == "scale_shift":
            if temb is None:
                raise ValueError(
                    f"temb should not be None when time_embedding_norm is {self.time_embedding_norm}"
                )
            time_scale, time_shift = torch.chunk(temb, 2, dim=1)
            hidden_states = self.norm2(hidden_states)
            hidden_states = hidden_states * (1 + time_scale) + time_shift
        else:
            hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        # logger.info("Using custom dowblock resnet")
        return output_tensor

class CameraClassConditionedResnetBlock2DUpsample(ResnetBlock2D):
    """
    Custom ResnetBlock2D that incorporates camera parameters and class embeddings
    """
    def __init__(
        self,
        *args,
        num_classes=3,
        camera_params_dim=8,  # Dimension of raw camera parameters
        camera_emb_size=32,    # Size of camera embedding
        class_emb_size=32,    # Size of class embedding
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        # Create embedding networks for camera parameters
        self.camera_emb = nn.Linear(camera_params_dim, camera_emb_size)
        self.class_emb = nn.Embedding(num_classes, class_emb_size)
        # Projection layer to incorporate camera and class embeddings
        # This will project (time_emb_dim + camera_emb_size + class_emb_size) to time_emb_dim
        self.combined_emb_proj = nn.Linear(
            kwargs.get("temb_channels", 512) + camera_emb_size + class_emb_size, 
            kwargs.get("temb_channels", 512)
        )

    def forward(
        self, 
        input_tensor, 
        temb,
        camera_params_target=None,
        class_labels=None,
        *args, 
        **kwargs
    ):
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The scale argument is deprecated and will be ignored."
            logger.warning(deprecation_message)

        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)

        # Process temporal embedding as usual
        if temb is not None and self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            combined_features = [temb]    
            # Process camera parameters if provided
            # Process camera parameters if provided
            if camera_params_target is not None:
                camera_features = self.camera_emb(camera_params_target)
                combined_features.append(camera_features)
                # logger.info("using camera target params in custom up block resnet")
                
            # Process class labels if provided
            if class_labels is not None:
                class_emb = self.class_emb(class_labels.long())
                combined_features.append(class_emb)
                # logger.info("using class labels in custom up block resnet")
                
            # Combine all available embeddings
            if len(combined_features) > 1:
                combined_emb = torch.cat(combined_features, dim=1)
                # Project to original temporal embedding dimension
                temb = self.combined_emb_proj(combined_emb)
                # logger.info("using combined in custom up block resnet")
            
            # Apply temporal embedding
            temb = self.time_emb_proj(temb)[:, :, None, None]

        if self.time_embedding_norm == "default":
            if temb is not None:
                hidden_states = hidden_states + temb
            hidden_states = self.norm2(hidden_states)
        elif self.time_embedding_norm == "scale_shift":
            if temb is None:
                raise ValueError(
                    f"temb should not be None when time_embedding_norm is {self.time_embedding_norm}"
                )
            time_scale, time_shift = torch.chunk(temb, 2, dim=1)
            hidden_states = self.norm2(hidden_states)
            hidden_states = hidden_states * (1 + time_scale) + time_shift
        else:
            hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        # logger.info("Using custom up block resnet")
        return output_tensor
class CameraClassConditionedResnetBlock2DMid(ResnetBlock2D):
    """
    Custom ResnetBlock2D that incorporates camera parameters and class embeddings
    """
    def __init__(
        self,
        *args,
        num_classes=3,
        camera_params_dim=8,  # Dimension of raw camera parameters
        camera_emb_size=32,    # Size of camera embedding
        class_emb_size=32,    # Size of class embedding
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        # Create embedding networks for camera parameters
        self.camera_emb = nn.Linear(camera_params_dim, camera_emb_size)
        self.class_emb = nn.Embedding(num_classes, class_emb_size)
        # Projection layer to incorporate camera and class embeddings
        # This will project (time_emb_dim + camera_emb_size + class_emb_size) to time_emb_dim
        self.combined_emb_proj = nn.Linear(
            kwargs.get("temb_channels", 512) + (camera_emb_size*2) + class_emb_size, 
            kwargs.get("temb_channels", 512)
        )

    def forward(
        self, 
        input_tensor, 
        temb,
        camera_params_input=None,
        camera_params_target=None,
        class_labels=None,
        *args, 
        **kwargs
    ):
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The scale argument is deprecated and will be ignored."
            # logger.warning(deprecation_message)

        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)

        # Process temporal embedding as usual
        if temb is not None and self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            combined_features = [temb]    
            # Process camera parameters if provided
            # Process camera parameters if provided
            if camera_params_target is not None:
                camera_features = self.camera_emb(camera_params_target)
                combined_features.append(camera_features)
                # logger.info("using camera target params in custom mid block resnet")

            if camera_params_input is not None:
                camera_features = self.camera_emb(camera_params_input)
                combined_features.append(camera_features)
                # logger.info("using camera input params in custom mid block resnet")

                
            # Process class labels if provided
            if class_labels is not None:
                class_emb = self.class_emb(class_labels.long())
                combined_features.append(class_emb)
                # logger.info("using class labels in custom mid block resnet")
                
            # Combine all available embeddings
            if len(combined_features) > 1:
                combined_emb = torch.cat(combined_features, dim=1)
                # Project to original temporal embedding dimension
                temb = self.combined_emb_proj(combined_emb)
                # logger.info("using combined params in custom mid block resnet")
            
            # Apply temporal embedding
            temb = self.time_emb_proj(temb)[:, :, None, None]

        if self.time_embedding_norm == "default":
            if temb is not None:
                hidden_states = hidden_states + temb
            hidden_states = self.norm2(hidden_states)
        elif self.time_embedding_norm == "scale_shift":
            if temb is None:
                raise ValueError(
                    f"temb should not be None when time_embedding_norm is {self.time_embedding_norm}"
                )
            time_scale, time_shift = torch.chunk(temb, 2, dim=1)
            hidden_states = self.norm2(hidden_states)
            hidden_states = hidden_states * (1 + time_scale) + time_shift
        else:
            hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        
        return output_tensor

@dataclass
class UNet2DConditionOutput(BaseOutput):
    """
    The output of [`UNet2DConditionModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.Tensor = None
class UNet2DConditionWithCamClass(UNet2DConditionModel):
    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        camera_params_input=None,
        camera_params_target=None,
        class_labels=None,
        # class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        r"""
        The [`UNet2DConditionModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch, channel, height, width)`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
            class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            timestep_cond: (`torch.Tensor`, *optional*, defaults to `None`):
                Conditional embeddings for timestep. If provided, the embeddings will be summed with the samples passed
                through the `self.time_embedding` layer to obtain the timestep embeddings.
            attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            added_cond_kwargs: (`dict`, *optional*):
                A kwargs dictionary containing additional embeddings that if specified are added to the embeddings that
                are passed along to the UNet blocks.
            down_block_additional_residuals: (`tuple` of `torch.Tensor`, *optional*):
                A tuple of tensors that if specified are added to the residuals of down unet blocks.
            mid_block_additional_residual: (`torch.Tensor`, *optional*):
                A tensor that if specified is added to the residual of the middle unet block.
            down_intrablock_additional_residuals (`tuple` of `torch.Tensor`, *optional*):
                additional residuals to be added within UNet down blocks, for example from T2I-Adapter side model(s)
            encoder_attention_mask (`torch.Tensor`):
                A cross-attention mask of shape `(batch, sequence_length)` is applied to `encoder_hidden_states`. If
                `True` the mask is kept, otherwise if `False` it is discarded. Mask will be converted into a bias,
                which adds large negative values to the attention scores corresponding to "discard" tokens.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] is returned,
                otherwise a `tuple` is returned where the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                # Forward upsample size to force interpolation output size.
                forward_upsample_size = True
                break

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        t_emb = self.get_time_embed(sample=sample, timestep=timestep)
        emb = self.time_embedding(t_emb, timestep_cond)

        # class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
        # if class_emb is not None:
        #     if self.config.class_embeddings_concat:
        #         emb = torch.cat([emb, class_emb], dim=-1)
        #     else:
        #         emb = emb + class_emb

        aug_emb = self.get_aug_embed(
            emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )
        if self.config.addition_embed_type == "image_hint":
            aug_emb, hint = aug_emb
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        encoder_hidden_states = self.process_encoder_hidden_states(
            encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )

        # 2. pre-process
        sample = self.conv_in(sample)

        # 2.5 GLIGEN position net
        if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            gligen_args = cross_attention_kwargs.pop("gligen")
            cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

        # 3. down
        # we're popping the `scale` instead of getting it because otherwise `scale` will be propagated
        # to the internal blocks and will raise deprecation warnings. this will be confusing for our users.
        if cross_attention_kwargs is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            lora_scale = cross_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)

        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        # using new arg down_intrablock_additional_residuals for T2I-Adapters, to distinguish from controlnets
        is_adapter = down_intrablock_additional_residuals is not None
        # maintain backward compatibility for legacy usage, where
        #       T2I-Adapter and ControlNet both use down_block_additional_residuals arg
        #       but can only use one or the other
        if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
            deprecate(
                "T2I should not use down_block_additional_residuals",
                "1.3.0",
                "Passing intrablock residual connections with `down_block_additional_residuals` is deprecated \
                       and will be removed in diffusers 1.3.0.  `down_block_additional_residuals` should only be used \
                       for ControlNet. Please make sure use `down_intrablock_additional_residuals` instead. ",
                standard_warn=False,
            )
            down_intrablock_additional_residuals = down_block_additional_residuals
            is_adapter = True

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                # For t2i-adapter CrossAttnDownBlock2D
                additional_residuals = {}
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)

                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    camera_params_input=camera_params_input,
                    class_labels=class_labels,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb,camera_params_input=camera_params_input,class_labels=class_labels)
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    sample += down_intrablock_additional_residuals.pop(0)

            down_block_res_samples += res_samples

        if is_controlnet:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    camera_params_input=camera_params_input,
                    camera_params_target=camera_params_target,
                    class_labels=class_labels,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = self.mid_block(sample, emb,camera_params_input=camera_params_input,camera_params_target=camera_params_target,class_labels=class_labels)

            # To support T2I-Adapter-XL
            if (
                is_adapter
                and len(down_intrablock_additional_residuals) > 0
                and sample.shape == down_intrablock_additional_residuals[0].shape
            ):
                sample += down_intrablock_additional_residuals.pop(0)

        if is_controlnet:
            sample = sample + mid_block_additional_residual

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    camera_params_target=camera_params_target,
                    class_labels=class_labels,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    camera_params_target=camera_params_target,
                    class_labels=class_labels,
                    upsample_size=upsample_size,
                )

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)

class CustomCrossAttnProcessor:
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        target_view_ray_data=None,
        input_view_ray_data=None,
        input_latents=None,
        **kwargs
    ):
        """
        Custom cross-attention processor that handles ray and latent data
        """
        # Debug message to confirm this processor is being used
        # logger.info("✅ CustomCrossAttnProcessor is being called!")
        
        B, L, _ = hidden_states.shape
        
        # Debug the input shapes
        # logger.info(f"Hidden states shape: {hidden_states.shape}")
        
        # Check if params are tuples (happens when passed through cross_attention_kwargs)
        # and extract the tensor if needed
        if isinstance(target_view_ray_data, tuple) and len(target_view_ray_data) > 0:
            target_view_ray_data = target_view_ray_data[0]
            # logger.info("Unpacked target_view_ray_data from tuple")
            
        if isinstance(input_view_ray_data, tuple) and len(input_view_ray_data) > 0:
            input_view_ray_data = input_view_ray_data[0]
            # logger.info("Unpacked input_view_ray_data from tuple")
            
        if isinstance(input_latents, tuple) and len(input_latents) > 0:
            input_latents = input_latents[0]
            # logger.info("Unpacked input_latents from tuple")
        
        # Debug the custom data availability
        # if target_view_ray_data is not None:
        #     # logger.info(f"Using target_view_ray_data with shape {target_view_ray_data.shape}")
        # if input_view_ray_data is not None:
        #     # logger.info(f"Using input_view_ray_data with shape {input_view_ray_data.shape}")
        # if input_latents is not None:
        #     # logger.info(f"Using input_latents with shape {input_latents.shape}")
        
        # === Q: concat learnable features + target view rays
        if target_view_ray_data is not None:
            tv = target_view_ray_data.view(B, L, -1)
            q_in = torch.cat([hidden_states, tv], dim=-1)
            if not hasattr(attn, "custom_q"):
                attn.custom_q = nn.Linear(q_in.shape[-1], attn.inner_dim, bias=False).to(q_in.device)
                # logger.info("Created custom_q linear layer")
            # logger.info(f"✅ custom q shape after linear projection {attn.custom_q.weight.shape}")
            q = attn.custom_q(q_in)
            # logger.info(f"✅  q shape after passed as keys to attention {q.shape}")
            # logger.info("✅ Using CUSTOM Q path")
        else:
            q = attn.to_q(hidden_states)
            logger.info("Using default Q path")
        
        # === K: concat input latents + input view rays
        if input_latents is not None and input_view_ray_data is not None:
            il = input_latents.view(B, L, -1)
            iv = input_view_ray_data.view(B, L, -1)
            k_in = torch.cat([il, iv], dim=-1)
            if not hasattr(attn, "custom_k"):
                attn.custom_k = nn.Linear(k_in.shape[-1], attn.inner_dim, bias=False).to(k_in.device)
                # logger.info("Created custom_k linear layer")
            # logger.info(f"✅ custom k shape after linear projection {attn.custom_k.weight.shape}")
            k = attn.custom_k(k_in)
            # logger.info(f"✅  k shape after passed as keys to attention {k.shape}")
            # logger.info("✅ Using CUSTOM K path")
        else:
            k = attn.to_k(encoder_hidden_states or hidden_states)
            logger.info("Using default K path")
        
        # === V: from input_latents only
        if input_latents is not None:
            vl = input_latents.view(B, L, -1)
            if not hasattr(attn, "custom_v"):
                attn.custom_v = nn.Linear(vl.shape[-1], attn.inner_dim, bias=False).to(vl.device)
                # logger.info("Created custom_v linear layer")
            # logger.info(f"✅ custom v shape after linear projection {attn.custom_v.weight.shape}")
            v = attn.custom_v(vl)
            # logger.info(f"✅  v shape after passed as keys to attention {v.shape}")
            # logger.info("✅ Using CUSTOM V path")
        else:
            v = attn.to_v(encoder_hidden_states or hidden_states)
            logger.info("Using default V path")
        
        # Standard attention steps
        q = attn.head_to_batch_dim(q)
        k = attn.head_to_batch_dim(k)
        v = attn.head_to_batch_dim(v)
        
        scores = torch.matmul(q, k.transpose(-1, -2)) * attn.scale
        probs = scores.softmax(dim=-1)
        out = torch.matmul(probs, v)
        
        out = attn.batch_to_head_dim(out)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        
        return out

def create_and_patch_unet(
    in_channels=4,
    out_channels=4,
    sample_size=512,
    block_out_channels=(128, 256, 512,512),
    layers_per_block=2,
    cross_attention_dim=768,
    attention_head_dim=64,
    dropout=0.0,
):
    # 1) Build the base model
    # from diffusers import UNet2DConditionModel
    # from diffusers.models.attention import CrossAttention
    # import types
    
    config = {
        "sample_size": sample_size,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "center_input_sample": False,
        "down_block_types": [
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ],
        "mid_block_type": "UNetMidBlock2DCrossAttn",
        "up_block_types": [
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ],
        "block_out_channels": block_out_channels,
        "layers_per_block": layers_per_block,
        "cross_attention_dim": cross_attention_dim,
        "attention_head_dim": attention_head_dim,
        "dropout": dropout,
    }
    model = UNet2DConditionWithCamClass.from_config(config)
    
    # Keep track of patched modules
    patched_modules = []
    
    # Create and set our custom attention processor
    custom_processor = CustomCrossAttnProcessor()
    
    # Register processor for all relevant cross-attention blocks
    for name, module in model.named_modules():
        if (
            isinstance(module, CrossAttention)
            and ("mid_block" in name or "up_blocks.0" in name or "up_blocks.1" in name)
            and name.endswith("attn2")
        ):
            # Use the direct processor assignment
            module.processor = custom_processor
            patched_modules.append(name)
            print(f"Patched processor for {name}")
    
    print(f"Total patched modules: {len(patched_modules)}")
    
    # For older diffusers versions, we might also need to patch the UNet2DConditionModel's 
    # _set_attention_processor method to prevent it from resetting our processors
    def custom_set_attn_processor(self, processor):
        # Only set processors for modules that are not in our patched list
        count = 0
        for name, module in self.named_modules():
            if isinstance(module, CrossAttention) and name not in patched_modules:
                module.processor = processor
                count += 1
        print(f"Set default processor for {count} non-patched attention modules")
        return self
    
    # Patch the _set_attention_processor method
    model._set_attention_processor = types.MethodType(custom_set_attn_processor, model)
    
    return model
def replace_resnet_with_custom(
    block: ResnetBlock2D,
    num_classes: int,
    camera_params_dim: int,
    camera_emb_size: int,
    class_emb_size: int,
) -> CameraClassConditionedResnetBlock2DDownsample:
    # 1) pull all of the original ResnetBlock2D's init args from its attrs
    temb_ch = None
    if block.time_emb_proj is not None:
        # time_emb_proj is an nn.Linear(temb_ch, out_ch) or Conv1d
        temb_ch = block.time_emb_proj.in_features

    init_kwargs = {
        "in_channels":    block.in_channels,
        "out_channels":   block.out_channels,
        "temb_channels":  temb_ch,
        "dropout":        block.dropout.p if isinstance(block.dropout, nn.Dropout) else block.dropout,
        "up":       block.upsample,
        "down":     block.downsample,
        "num_classes":        num_classes,
        "camera_params_dim":  camera_params_dim,
        "camera_emb_size":    camera_emb_size,
        "class_emb_size":     class_emb_size,
    }

    # 2) instantiate your subclass
    new_block = CameraClassConditionedResnetBlock2DDownsample(**init_kwargs)

    # 3) copy over the old weights (strict=False lets us ignore the two new embedding layers)
    new_block.load_state_dict(block.state_dict(), strict=False)
    return new_block
def replace_resnet_with_custom_up(
    block: ResnetBlock2D,
    num_classes: int,
    camera_params_dim: int,
    camera_emb_size: int,
    class_emb_size: int,
) -> CameraClassConditionedResnetBlock2DUpsample:
    # 1) pull all of the original ResnetBlock2D's init args from its attrs
    temb_ch = None
    if block.time_emb_proj is not None:
        # time_emb_proj is an nn.Linear(temb_ch, out_ch) or Conv1d
        temb_ch = block.time_emb_proj.in_features

    init_kwargs = {
        "in_channels":    block.in_channels,
        "out_channels":   block.out_channels,
        "temb_channels":  temb_ch,
        "dropout":        block.dropout.p if isinstance(block.dropout, nn.Dropout) else block.dropout,
        "up":       block.upsample,
        "down":     block.downsample,
        "num_classes":        num_classes,
        "camera_params_dim":  camera_params_dim,
        "camera_emb_size":    camera_emb_size,
        "class_emb_size":     class_emb_size,
    }

    # 2) instantiate your subclass
    new_block = CameraClassConditionedResnetBlock2DUpsample(**init_kwargs)

    # 3) copy over the old weights (strict=False lets us ignore the two new embedding layers)
    new_block.load_state_dict(block.state_dict(), strict=False)
    return new_block

def replace_resnet_with_custom_mid(
    block: ResnetBlock2D,
    num_classes: int,
    camera_params_dim: int,
    camera_emb_size: int,
    class_emb_size: int,
) -> CameraClassConditionedResnetBlock2DMid:
    # 1) pull all of the original ResnetBlock2D's init args from its attrs
    temb_ch = None
    if block.time_emb_proj is not None:
        # time_emb_proj is an nn.Linear(temb_ch, out_ch) or Conv1d
        temb_ch = block.time_emb_proj.in_features

    init_kwargs = {
        "in_channels":    block.in_channels,
        "out_channels":   block.out_channels,
        "temb_channels":  temb_ch,
        "dropout":        block.dropout.p if isinstance(block.dropout, nn.Dropout) else block.dropout,
        "up":       block.upsample,
        "down":     block.downsample,
        "num_classes":        num_classes,
        "camera_params_dim":  camera_params_dim,
        "camera_emb_size":    camera_emb_size,
        "class_emb_size":     class_emb_size,
    }

    # 2) instantiate your subclass
    new_block = CameraClassConditionedResnetBlock2DMid(**init_kwargs)

    # 3) copy over the old weights (strict=False lets us ignore the two new embedding layers)
    new_block.load_state_dict(block.state_dict(), strict=False)
    return new_block
def inject_custom_down_blocks(
    unet: UNet2DConditionModel,
    num_classes=3,
    camera_params_dim=8,
    camera_emb_size=32,
    class_emb_size=32,
):
    # from diffusers.models.unet_2d_condition import ResnetBlock2D

    for down in unet.down_blocks:                    # each is a DownBlock2D or AttnDownBlock2D
        for i, orig_resnet in enumerate(down.resnets):
            if isinstance(orig_resnet, ResnetBlock2D):
                down.resnets[i] = replace_resnet_with_custom(
                    orig_resnet,
                    num_classes,
                    camera_params_dim,
                    camera_emb_size,
                    class_emb_size,
                )
def inject_custom_up_blocks(
    unet: UNet2DConditionModel,
    num_classes=3,
    camera_params_dim=8,
    camera_emb_size=32,
    class_emb_size=32,
):
    # from diffusers.models.unet_2d_condition import ResnetBlock2D

    for up in unet.up_blocks:                    # each is a DownBlock2D or AttnDownBlock2D
        for i, orig_resnet in enumerate(up.resnets):
            if isinstance(orig_resnet, ResnetBlock2D):
                up.resnets[i] = replace_resnet_with_custom_up(
                    orig_resnet,
                    num_classes,
                    camera_params_dim,
                    camera_emb_size,
                    class_emb_size,
                )
def inject_custom_mid_block(
    unet: UNet2DConditionModel,
    num_classes=3,
    camera_params_dim=8,
    camera_emb_size=32,
    class_emb_size=32,
):
    # from diffusers.models.unet_2d_condition import ResnetBlock2D

    # for mid in unet.mid_block:                    # each is a DownBlock2D or AttnDownBlock2D
        for i, orig_resnet in enumerate(unet.mid_block.resnets):
            if isinstance(orig_resnet, ResnetBlock2D):
                unet.mid_block.resnets[i] = replace_resnet_with_custom_mid(
                    orig_resnet,
                    num_classes,
                    camera_params_dim,
                    camera_emb_size,
                    class_emb_size,
                )


model = create_and_patch_unet()
inject_custom_down_blocks(model)
inject_custom_up_blocks(model)
inject_custom_mid_block(model)
total_params = sum(p.numel() for p in model.parameters()) / 1e6

print(f"Total number of parameters in TransferUnet: {total_params:.2f}M")

def patched_forward(self, hidden_states, temb, encoder_hidden_states=None,
                    camera_params_input=None, class_labels=None):
        output_states = ()

        for resnet in self.resnets:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb,camera_params_input=camera_params_input,class_labels=class_labels)
            else:
                hidden_states = resnet(hidden_states, temb,camera_params_input=camera_params_input,class_labels=class_labels)

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states

def patched_forward_attention_downsample(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        camera_params_input=None, 
        class_labels=None,
        upsample_size: Optional[int] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        if cross_attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        output_states = ()

        for resnet, attn in zip(self.resnets, self.attentions):
            # if torch.is_grad_enabled() and self.gradient_checkpointing:
            #     hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb,camera_params_input=camera_params, class_labels=class_labels)
            #     hidden_states = attn(hidden_states, **cross_attention_kwargs)
            #     output_states = output_states + (hidden_states,)
            # else:
                hidden_states = resnet(hidden_states, temb,camera_params_input=camera_params_input, class_labels=class_labels)
                hidden_states = attn(hidden_states, **cross_attention_kwargs)
                output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                if self.downsample_type == "resnet":
                    hidden_states = downsampler(hidden_states, temb=temb,camera_params_input=camera_params_input, class_labels=class_labels)
                else:
                    hidden_states = downsampler(hidden_states,camera_params_input=camera_params_input, class_labels=class_labels)

            output_states += (hidden_states,)

        return hidden_states, output_states

def patch_forward_cross_attn_up_block_(
    self,
    hidden_states: torch.Tensor,
    res_hidden_states_tuple: Tuple[torch.Tensor, ...],
    temb: Optional[torch.Tensor] = None,
    camera_params_input=None,
    camera_params_target=None, 
    class_labels=None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    upsample_size: Optional[int] = None,
    attention_mask: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if cross_attention_kwargs is not None:
        if cross_attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

    is_freeu_enabled = (
        getattr(self, "s1", None)
        and getattr(self, "s2", None)
        and getattr(self, "b1", None)
        and getattr(self, "b2", None)
    )

    for resnet, attn in zip(self.resnets, self.attentions):
        # pop res hidden states
        res_hidden_states = res_hidden_states_tuple[-1]
        res_hidden_states_tuple = res_hidden_states_tuple[:-1]

        # FreeU: Only operate on the first two stages
        if is_freeu_enabled:
            hidden_states, res_hidden_states = apply_freeu(
                self.resolution_idx,
                hidden_states,
                res_hidden_states,
                s1=self.s1,
                s2=self.s2,
                b1=self.b1,
                b2=self.b2,
            )

        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb,camera_params_target=camera_params_target, class_labels=class_labels)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]
        else:
            hidden_states = resnet(hidden_states, temb,camera_params_target=camera_params_target, class_labels=class_labels)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]

    if self.upsamplers is not None:
        for upsampler in self.upsamplers:
            hidden_states = upsampler(hidden_states, upsample_size)

    return hidden_states

def patch_forward_up_block_2d(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        temb: Optional[torch.Tensor] = None,
        camera_params_target=None, 
        camera_params_input=None,
        class_labels=None,
        upsample_size: Optional[int] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        is_freeu_enabled = (
            getattr(self, "s1", None)
            and getattr(self, "s2", None)
            and getattr(self, "b1", None)
            and getattr(self, "b2", None)
        )

        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            # FreeU: Only operate on the first two stages
            if is_freeu_enabled:
                hidden_states, res_hidden_states = apply_freeu(
                    self.resolution_idx,
                    hidden_states,
                    res_hidden_states,
                    s1=self.s1,
                    s2=self.s2,
                    b1=self.b1,
                    b2=self.b2,
                )

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb,camera_params_target=camera_params_target, class_labels=class_labels)
            else:
                hidden_states = resnet(hidden_states, temb,camera_params_target=camera_params_target, class_labels=class_labels)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states

def patch_forward_mid_block(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        camera_params_target=None, 
        camera_params_input=None,
        class_labels=None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb,camera_params_target=camera_params_target, camera_params_input=camera_params_input,class_labels=class_labels)
            else:
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
                hidden_states = resnet(hidden_states, temb,camera_params_target=camera_params_target, camera_params_input=camera_params_input,class_labels=class_labels)

        return hidden_states
# Apply to every down-block in your model:
# for down in model.down_blocks:
#     down.forward = MethodType(patched_forward, down)
for down in model.down_blocks:
    if isinstance(down, DownBlock2D):
        down.forward = MethodType(patched_forward, down)
    elif isinstance(down, AttnDownBlock2D):
        down.forward = MethodType(patched_forward_attention_downsample, down)
    else:
        # (you can warn or handle others if you like)
        print(f"⚠️  Unexpected block type: {type(down)}")

for up in model.up_blocks:
    if isinstance(up, UpBlock2D):
        up.forward = MethodType(patch_forward_up_block_2d, up)
    elif isinstance(up, CrossAttnUpBlock2D):
        up.forward = MethodType(patch_forward_cross_attn_up_block_, up)
    else:
        # (you can warn or handle others if you like)
        print(f"⚠️  Unexpected block type: {type(up)}")

# for mid in model.mid_block:
if isinstance(model.mid_block, UNetMidBlock2DCrossAttn):
    model.mid_block.forward = MethodType(patch_forward_mid_block,model.mid_block)
# elif isinstance(up, CrossAttnUpBlock2D):
#     up.forward = MethodType(patch_forward_cross_attn_up_block_, up)
else:
    # (you can warn or handle others if you like)
    print(f"⚠️  Unexpected block type: {type(up)}")
TransferUnet = model
TransferUnet.to("cuda")
TransferUnet.float()

def decode_and_show_image(latents, vae):
    # Reverse the scaling applied during encoding
    latents = latents / 0.18215
    
    # Decode the latent vectors into images
    decoded_images = vae.decode(latents).sample
    
    # Reverse the normalization from [-1, 1] to [0, 1]
    decoded_images = (decoded_images + 1.) / 2.
    
    # Convert the tensor to a NumPy array and squeeze if necessary
    decoded_images = decoded_images.squeeze().detach().cpu().numpy()
    
    # Plot the decoded image
    plt.imshow(decoded_images.transpose(1, 2, 0))
    # plt.imshow(decoded_images)
    plt.axis('off')
    plt.show()

device = "cuda:0"
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to("cuda").requires_grad_(False)

@torch.no_grad()
def invert(
    start_latents,
    prompt="",
    guidance_scale=3.5,
    num_inference_steps=50,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
    device="cuda:0",
    batch_size=32
):

    torch.cuda.empty_cache()  # Free GPU memory before starting

    # Dynamically adjust batch size based on available GPU memory
    batch_size = min(batch_size, start_latents.shape[0])  # Ensure batch_size <= total samples

    text_embeddings = pipe._encode_prompt(
        [""] * batch_size, device, num_images_per_prompt, do_classifier_free_guidance, [""] * batch_size
    )

    # latents = start_latents.clone().to(device, dtype=torch.float16)  # Move to GPU & use float16
    latents = start_latents.clone().to(device)
    intermediate_latents = []
    intermediate_means = []
    intermediate_variance = []

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = reversed(pipe.scheduler.timesteps)

    stream = torch.cuda.Stream()  # Create a CUDA stream for parallel execution

    with torch.cuda.stream(stream):  # Run operations asynchronously on the stream
        for i in range(1, num_inference_steps - 1):  # Skip the last step
            t = timesteps[i]

            with amp.autocast():  # Enable mixed precision
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
                noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
               

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                current_t = max(0, t.item() - (1000 // num_inference_steps))
                next_t = t
                alpha_t = pipe.scheduler.alphas_cumprod[current_t]
                alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]

                mean = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt())
                variance = (1 - alpha_t_next).sqrt() * noise_pred
                latents = mean + variance
                # if i == 30:
                intermediate_latents.append(latents.clone())  # Avoid modifying in-place
                intermediate_means.append(mean.clone())
                intermediate_variance.append(variance.clone())

    torch.cuda.synchronize()  # Ensure all CUDA operations are complete

    return intermediate_latents, intermediate_means, intermediate_variance

@torch.no_grad()
def sample(
    prompt,
    start_step=0,
    start_latents=None,
    guidance_scale=3.5,
    num_inference_steps=30,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
    device=device,
    keep_last_k=10
):
    
    # Encode prompt
    text_embeddings = pipe._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )
    noise_preds_list = []
    latents_list = []
    predicted_x0_list = []
    direction_towards_xt = []
    # intermediate_pils = []
    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Create a random starting point if we don't have one already
    if start_latents is None:
        start_latents = torch.randn(1, 4, 64, 64, device=device)
        start_latents *= pipe.scheduler.init_noise_sigma

    latents = start_latents.clone()

    # for i in tqdm(range(start_step, num_inference_steps)):
    for i in range(start_step, num_inference_steps):

        t = pipe.scheduler.timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Normally we'd rely on the scheduler to handle the update step:
        # latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # Instead, let's do it ourselves:
        prev_t = max(1, t.item() - (1000 // num_inference_steps))  # t-1
        alpha_t = pipe.scheduler.alphas_cumprod[t.item()]
        alpha_t_prev = pipe.scheduler.alphas_cumprod[prev_t]
        predicted_x0 = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        direction_pointing_to_xt = (1 - alpha_t_prev).sqrt() * noise_pred
        latents = alpha_t_prev.sqrt() * predicted_x0 + direction_pointing_to_xt
        noise_preds_list.append(noise_pred)
        latents_list.append(latents)
        predicted_x0_list.append(predicted_x0)
        direction_towards_xt.append(direction_pointing_to_xt)
        # if i >= num_inference_steps - keep_last_k:
        #     np_img = pipe.decode_latents(latents)
        #     pil_img = pipe.numpy_to_pil(np_img)
        #     intermediate_pils.append(pil_img)

    # Post-processing
    images = pipe.decode_latents(latents)
    images = pipe.numpy_to_pil(images)

    return images,noise_preds_list, latents_list,predicted_x0_list, direction_towards_xt
    # return intermediate_pils,noise_preds_list, latents_list,predicted_x0_list, direction_towards_xt
from PIL import Image, ImageDraw, ImageFont
def merge_images(result, result_direct_fusion, target, save_path):
    # Ensure all images have the same height
    width, height = result.size
    merged_width = width * 3

    # Create a new blank image
    merged_image = Image.new("RGB", (merged_width, height + 30), (255, 255, 255))  # Extra space for text

    # Paste images side by side
    merged_image.paste(result, (0, 30))
    merged_image.paste(result_direct_fusion, (width, 30))
    merged_image.paste(target, (width * 2, 30))

    # Draw text labels
    draw = ImageDraw.Draw(merged_image)
    font = ImageFont.load_default()  # Use default font, or specify a TTF file

    draw.text((width // 2 - 20, 5), "Result", fill="black", font=font)
    draw.text((width + width // 2 - 40, 5), "Direct Fusion", fill="black", font=font)
    draw.text((2 * width + width // 2 - 20, 5), "Target", fill="black", font=font)

    # Save the merged image
    merged_image.save(save_path)
def find_closest_timestep(latent_inp):
    # Load the diffusion model
    # pipeline = StableDiffusionPipeline.from_pretrained(model_name)
    scheduler = pipe.scheduler

    # Ensure the noisy latent is a tensor
    if not isinstance(latent_inp, torch.Tensor):
        latent_inp = torch.tensor(latent_inp)

    # Calculate the empirical variance of the noisy latent
    empirical_variance = latent_inp.var().item()

    # Calculate theoretical variance for each timestep
    beta_schedule = scheduler.betas.numpy()
    alphas = 1.0 - beta_schedule
    alphas_cumprod = np.cumprod(alphas)  # \(\bar{\alpha}_t\)

    theoretical_variances = 1.0 - alphas_cumprod

    # Find the timestep \( t \) where the theoretical variance is closest to the empirical variance
    closest_timestep = np.argmin(np.abs(theoretical_variances - empirical_variance))

    return closest_timestep, theoretical_variances[closest_timestep], empirical_variance
def to_imgs(img_path):
        # This function reads an image, applies any transformations, and returns it
        img = read_image(img_path, mode=ImageReadMode.RGB) / 255 
        transform = T.Compose([
            T.Resize((512)),
             T.CenterCrop(512)
        ])
        img = transform(img)
        img_ndarray = img.permute(1, 2, 0).numpy()  # (H, W, C)
        return img_ndarray
def save_consolidated_image_six_methods(input_image, target_image, all_results, index, cls, class_prompt, save_dir, epochs,epoch):
    """
    Create and save a consolidated image showing input, target, and all six result types from each checkpoint
    """
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    
    # Define image size and spacing
    img_width, img_height = target_image.size
    padding = 10
    title_height = 30
    score_height = 25  # Height for score text
    method_label_width = 120  # Width for method labels
    method_labels = [
                        "Standard", 
                        "Direct Fusion", 
                        "With Prompt", 
                        "More Variance", 
                        "More Var + Prompt",
                        "Direct Fusion + Prompt",
                        # "More Var Start 17",
                        # "More Var More Guidance",
                        "More Var 15",
                        # "Direct Fusion 10" # New method
                    ]
    # Calculate dimensions for the consolidated image
    epoch_columns = len(epochs)
    total_width = method_label_width + 2 * (img_width + padding) + epoch_columns * (img_width + padding) + padding
    # Each of the 6 methods gets its own row
    total_height = len(method_labels) * (img_height + padding + score_height) + 2 * padding + title_height

    
    # Create a new blank image
    consolidated_img = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(consolidated_img)
    
    # Try to get a font, use default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        small_font = ImageFont.truetype("arial.ttf", 12)
    except IOError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Add title
    title = f"Results for Index {index}, Class {cls} ({class_prompt})"
    draw.text((padding, padding), title, fill=(0, 0, 0), font=font)
    
    # Define method labels

    
    # Add column headers
    x_offset_start = method_label_width
    y_offset = padding + title_height
    
    # Input and Target column headers
    draw.text((x_offset_start + (img_width // 2), y_offset - 15), "Input", fill=(0, 0, 0), font=small_font)
    draw.text((x_offset_start + img_width + padding + (img_width // 2), y_offset - 15), "Target", fill=(0, 0, 0), font=small_font)
    
    # epoch column headers
    x_offset_ckpt = x_offset_start + 2 * (img_width + padding)
    for i, epoch in enumerate(epochs):
        draw.text((x_offset_ckpt + i * (img_width + padding) + (img_width // 2), y_offset - 15), 
                  f"{epoch}", fill=(0, 0, 0), font=small_font)
    
    # Process each method (rows)
    for row, method in enumerate(method_labels):
        row_y_offset = y_offset + row * (img_height + padding + score_height)
        
        # Draw method label
        draw.text((padding, row_y_offset + (img_height // 2)), method, fill=(0, 0, 0), font=small_font)
        
        # Place input image
        x_offset = x_offset_start
        consolidated_img.paste(input_image, (x_offset, row_y_offset))
        
        # Place target image
        x_offset += img_width + padding
        consolidated_img.paste(target_image, (x_offset, row_y_offset))
        
        # Place checkpoint results for this method
        x_offset = x_offset_start + 2 * (img_width + padding)
        
        for i, epoch in enumerate(epochs):
            if epoch in all_results:
                x_pos = x_offset + i * (img_width + padding)
                
                # Select the appropriate image and scores based on the current method
                if method == "Standard":
                    result_img = all_results[epoch]["image"]
                    scores = all_results[epoch]["scores"]
                elif method == "Direct Fusion":
                    result_img = all_results[epoch]["direct_fusion"]
                    scores = all_results[epoch]["scores_direct_fusion"]
                elif method == "With Prompt":
                    result_img = all_results[epoch]["with_prompt"]
                    scores = all_results[epoch]["scores_with_prompt"]
                elif method == "More Variance":
                    result_img = all_results[epoch]["more_var"]
                    scores = all_results[epoch]["scores_more_var"]
                elif method == "More Var + Prompt":
                    result_img = all_results[epoch]["with_prompt_more_var"]
                    scores = all_results[epoch]["scores_with_prompt_more_var"]
                elif method ==  "Direct Fusion + Prompt":
                    result_img = all_results[epoch]["direct_fusion_with_prompt"]
                    scores = all_results[epoch]["scores_direct_fusion_with_prompt"]
                # elif method == "More Var Start 17":
                #     result_img = all_results[epoch]["more_var_start_17"]
                #     scores = all_results[epoch]["scores_more_var_start_17"]
                # elif method == "More Var More Guidance":
                #     result_img = all_results[epoch]["more_var_more_guidance"]
                #     scores = all_results[epoch]["scores_more_var_more_guidance"]
                else:
                    result_img = all_results[epoch]["more_var_15"]
                    scores = all_results[epoch]["scores_more_var_15"]
                # else : #direct fusion 10
                #     result_img = all_results[epoch]["direct_fusion_10"]
                #     scores = all_results[epoch]["scores_direct_fusion_10"]
                    
                    
                    
                
                # Paste the image
                consolidated_img.paste(result_img, (x_pos, row_y_offset))
                
                # Add scores
                alex_score = scores["AlexNet"]
                vgg_score = scores["VGG"]
                vgg_color = (0, 128, 0) if vgg_score < 0.60 else (255, 0, 0)
                
                draw.text((x_pos, row_y_offset + img_height + 5), 
                        f"A:{alex_score:.3f} V:{vgg_score:.3f}", fill=vgg_color, font=small_font)
    
    # Save the consolidated image
    save_path = os.path.join(save_dir, f"consolidated_{index}_{cls}_{epoch}.jpg")
    consolidated_img.save(save_path)
    print(f"Saved consolidated results image with all six methods to {save_path}")
loss_fn_alex = lpips.LPIPS(net='alex')
loss_fn_vgg = lpips.LPIPS(net='vgg')
def load_captions(csv_file_path):
    """
    Reads the CSV and returns a dict: { image_path: caption }
    """
    captions = {}
    with open(csv_file_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # row["image_path"] → "/home/.../001.jpg"
            # row["caption"]    → "a wooden chair sitting..."
            captions[row["image_path"]] = row["caption"]
    return captions


caption_map = load_captions("/home/sehajs/mvimgnet_3_classes/image_captions.csv")
def get_caption_for(path, caption_map):
    return caption_map.get(path, "")


import logging
# import torch.multiprocessing as mp
# mp.set_start_method("spawn", force=True)
# mp.set_start_method("fork", force=True)
# Get the logger used by diffusers
logging.getLogger("diffusers.models.attention_processor").setLevel(logging.ERROR)
def create_class_mapping(class_list):
    """
    Creates a dictionary mapping each class to a unique index.
    
    Args:
        class_list: List of class labels as strings
        
    Returns:
        Dictionary with class labels as keys and indices as values
    """
    return {class_label: idx for idx, class_label in enumerate(sorted(class_list))}

def flatten_hw_input_latents(x):
    B, C, H, W = x.shape
    L = H * W
    return x.reshape(B, C, L).permute(0, 2, 1).contiguous()

def flatten_hw_for_rays(x_hw3):
    """
    Turn x_hw3 of shape (H, W, C) or (B, H, W, C)
    into (B, L=H*W, C).
    """
    # 1) ensure a batch dim
    if x_hw3.ndim == 3:           # (H, W, C)
        x = x_hw3.unsqueeze(0)    # → (1, H, W, C)
    else:                         # already (B, H, W, C)
        x = x_hw3

    B, H, W, C = x.shape
    L = H * W

    # 2) move channels to front
    x = x.permute(0, 3, 1, 2)     # → (B, C, H, W)

    # 3) flatten spatial dims, then move channels to last
    x = x.reshape(B, C, L)        # → (B, C, L)
    x = x.permute(0, 2, 1).contiguous()  # → (B, L, C)

    return x

# Your classes
classes = ['6','8','36']
save_dir = "/home/sehajs/mvimgnet_3_classes/results"
aggregated_results_path = os.path.join(save_dir, "lpips_scores_aggregated.json")
# Create the mapping
class_to_idx = create_class_mapping(classes)
num_epochs = 2500
tmax = num_epochs * len(inverted_latent_camera_dl)
use_amp = True
train_losses = []
val_losses = []
optimizer = torch.optim.AdamW(TransferUnet.parameters(), lr=1e-5)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-5, total_steps=tmax)
val_losses = []
    
loss_df = pd.DataFrame(columns=['epoch', 'train_loss', 'train_mse_loss', 'val_loss','val_mse_loss'])
def plot_losses(train_losses, val_losses, save_path="loss_plot.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', color='blue')
    
    # Check if validation losses are provided and plot them
    if val_losses is not None:
        plt.plot(val_losses, label='Validation Loss', color='orange')
        
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Save the plot to a file
    plt.savefig(save_path)
    plt.close()  # Close the plot to avoid displaying in notebooks

# Signal handler to catch interruptions (Ctrl + C)
def signal_handler(sig, frame):
    print("\nTraining interrupted! Plotting the losses...")
    plot_losses(train_losses, val_losses)
    exit(0)

for epoch in range(num_epochs):
    # Training mode
    TransferUnet.train()
    train_loss = 0.0
    # distribution_loss_epoch = 0.0
    mse_loss_epoch = 0
    images_ds_means.generate_pairs()

    # Initialize the dataloader
    inverted_latent_camera_dl = DataLoader(images_ds_means, batch_size=32, shuffle=True, num_workers=4)
    print("dataloader created")
    # Initialize the progress bar for training
    with tqdm(enumerate(inverted_latent_camera_dl), total=len(inverted_latent_camera_dl), desc=f"Epoch {epoch+1}/{num_epochs} - Training", unit="batch") as tepoch:
        for step, batch in tepoch:
            xb_path,yb_path,xb, yb,cam_data1,cam_data2,class_data,input_rays,target_rays = batch
            batch_size = len(batch[0])
            yb,cam_data1,cam_data2 = yb.cuda(),cam_data1.cuda(),cam_data2.cuda()
            prompt = [""] * batch_size
            if step == 0:  # Check if it's the first batch
                print(f"First batch paths:\nInput: {xb_path[0]}\nTarget: {yb_path[0]}")
            xb_for_cross_attn = flatten_hw_input_latents(xb)
            xb,xb_for_cross_attn = xb.cuda(),xb_for_cross_attn.cuda()
            input_rays = flatten_hw_for_rays(input_rays)
            target_rays = flatten_hw_for_rays(target_rays)
            input_rays,target_rays = input_rays.cuda(),target_rays.cuda()
            class_labels = torch.tensor([class_to_idx[c] for c in class_data], dtype=torch.long).to("cuda")
            # print("input latents for cross attn, target and input rays shape", xb_for_cross_attn.shape,input_rays.shape,target_rays.shape)
            cross_attention_kwargs = {
                                        "target_view_ray_data": target_rays,
                                        "input_view_ray_data": input_rays,
                                        "input_latents": xb_for_cross_attn
                                    }
            # Using CLIP model to get embeddings
            # text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            # with torch.no_grad(): 
            #     text_embeddings = text_encoder(
            #         text_input.input_ids.to("cuda")
            #     )[0]
            # text_embeddings = text_embeddings.half()

            transfer_unet_input = xb
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                # transfer_pred = TransferUnet(transfer_unet_input, 0,camera_data_input=cam_data1,camera_data_target=cam_data2,class_labels=class_labels).sample
                transfer_pred = TransferUnet(sample=transfer_unet_input,timestep=0,encoder_hidden_states=None,camera_params_input=cam_data1,camera_params_target=cam_data2,class_labels=class_labels,cross_attention_kwargs=cross_attention_kwargs,return_dict=True).sample
                target = yb
                # mmd_loss = MMD(transfer_pred, target,kernel='multiscale')
                mse_loss = (F.mse_loss(transfer_pred.float(), target.float(), reduction="mean"))
                loss = mse_loss 
                # loss = mse_loss
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Update learning rate scheduler
            lr_scheduler.step()

            # Accumulate the training loss
            train_loss += loss.item()
            # distribution_loss_epoch += mmd_loss/len(inverted_latent_camera_dl)
            mse_loss_epoch += mse_loss / len(inverted_latent_camera_dl)
            tepoch.set_postfix(loss=loss.item())
    train_loss_epoch = train_loss / len(inverted_latent_camera_dl)
    train_losses.append(train_loss_epoch)
    if (epoch + 1) % 50 == 0:
                    save_path = f"/home/sehajs/mvimgnet_3_classes/ckpt/transfer_unet_{epoch+1}"
                    torch.save({'epoch': epoch + 1,'model_state_dict': TransferUnet.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'scheduler': lr_scheduler.state_dict(),'loss' : train_loss_epoch}, save_path)
                            
    
    print(f"Epoch {epoch+1}, Training Loss: {train_loss_epoch:.4f},MSE Loss:{mse_loss_epoch:.4f}")

    ## Validation set run
    TransferUnet.eval()
    images_ds_means_val.generate_pairs()
    inverted_latent_camera_dl_val = DataLoader(images_ds_means_val, batch_size=32, shuffle=False, num_workers=4)
    val_loss = 0.0
    mse_loss_epoch_val = 0.0

    with torch.no_grad():
    # Initialize the progress bar for training
        with tqdm(enumerate(inverted_latent_camera_dl_val), total=len(inverted_latent_camera_dl_val), desc=f"Epoch {epoch+1}/{num_epochs} - Validation", unit="batch") as tepoch:
            for step, batch in tepoch:
                xb_path,yb_path,xb, yb,cam_data1,cam_data2,class_data,input_rays,target_rays = batch
                batch_size = len(batch[0])
                yb,cam_data1,cam_data2 = yb.cuda(),cam_data1.cuda(),cam_data2.cuda()
                prompt = [""] * batch_size
                if step == 0:  # Check if it's the first batch
                    print(f"First batch paths:\nInput: {xb_path[0]}\nTarget: {yb_path[0]}")
                
                xb_for_cross_attn = flatten_hw_input_latents(xb)
                xb,xb_for_cross_attn = xb.cuda(),xb_for_cross_attn.cuda()
                input_rays = flatten_hw_for_rays(input_rays)
                target_rays = flatten_hw_for_rays(target_rays)
                input_rays,target_rays = input_rays.cuda(),target_rays.cuda()
                class_labels = torch.tensor([class_to_idx[c] for c in class_data], dtype=torch.long).to("cuda")
                cross_attention_kwargs = {
                                            "target_view_ray_data": target_rays,
                                            "input_view_ray_data": input_rays,
                                            "input_latents": xb_for_cross_attn
                                        }
                
                
    
                transfer_unet_input = xb
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    # transfer_pred = TransferUnet(transfer_unet_input, 0,camera_data_input=cam_data1,camera_data_target=cam_data2,class_labels=class_labels).sample
                    transfer_pred = TransferUnet(sample=transfer_unet_input,timestep=0,encoder_hidden_states=None,camera_params_input=cam_data1,camera_params_target=cam_data2,class_labels=class_labels,cross_attention_kwargs=cross_attention_kwargs,return_dict=True).sample
                    target = yb
                    # mmd_loss = MMD(transfer_pred, target,kernel='multiscale')
                    mse_loss_val = (F.mse_loss(transfer_pred.float(), target.float(), reduction="mean"))
                    loss_val = mse_loss_val 
                    # loss = mse_loss

                if(epoch+1) % 50 == 0:
                    # if step==0:
                            epochs = [epoch]
                            if batch_size == 32:
                                indices = [3, 8, 12, 14, 19, 25, 28, 31]
                            else:
                                num_indices = 8
                                indices = random.sample(range(batch_size), min(batch_size, num_indices)) 
                            results = []
                            for index in indices:
                                    global_idx = step * batch_size + index
                                    result = {
                                                "epoch": epoch,
                                                "index": global_idx,
                                                "class": class_labels[index].item(),
                                                "input_path": xb_path[index],
                                                "target_path": yb_path[index],
                                                
                                                # LPIPS
                                                "lpips_scores": {},
                                                "lpips_scores_direct_fusion": {},
                                                "lpips_scores_with_prompt": {},
                                                "lpips_scores_more_var": {},
                                                "lpips_scores_with_prompt_more_var": {},
                                                "lpips_scores_direct_fusion_with_prompt": {},
                                                # "lpips_scores_more_var_start_17": {},
                                                # "lpips_score_with_prompt_more_var_more_guidance": {},
                                                "lpips_score_more_var_15": {},
                                                # "lpips_score_direct_fusion_10": {},

                                                # PSNR
                                                "psnr_scores": {},
                                                "psnr_scores_direct_fusion": {},
                                                "psnr_scores_with_prompt": {},
                                                "psnr_scores_more_var": {},
                                                "psnr_scores_with_prompt_more_var": {},
                                                "psnr_scores_direct_fusion_with_prompt": {},
                                                # "psnr_scores_more_var_start_17": {},
                                                # "psnr_score_with_prompt_more_var_more_guidance": {},
                                                "psnr_score_more_var_15": {},
                                                # "psnr_score_direct_fusion_10": {}
                                            }
                                
                                    # For storing all results to create consolidated image
                                    all_results = {}
                                    target_image = None
                                    has_good_score = False  # Flag to track if any checkpoint has VGG score < 0.60
                                    # class_prompt = "chair"
                                    class_prompt = get_caption_for(xb_path[index], caption_map)
                                    print(class_prompt)
                                    input_image = to_imgs(xb_path[index])
                                    print("input image type",type(input_image))
                                    input_image_prompt = ""
                                    with torch.no_grad():
                                        latent = pipe.vae.encode(T.functional.to_tensor(input_image).unsqueeze(0).to(device) * 2 - 1)
                                        l = 0.18215 * latent.latent_dist.sample()
                                        inverted_latents = invert(l, input_image_prompt, num_inference_steps=50)
                                    
                                    noise_pred_transfer_unet = transfer_pred[index].float()
                                    
                                    # 1. Original result
                                    alpha_t_next = pipe.scheduler.alphas_cumprod[800]
                                    latent_input_v2 = noise_pred_transfer_unet + (1 - alpha_t_next).sqrt() * inverted_latents[2][-20]
                                    
                                    closest_timestep, _, _ = find_closest_timestep(latent_input_v2)
                                    text_embeddings = pipe._encode_prompt("", device, 1, True, "")
            
                                    latents = latent_input_v2.clone()
                                    latent_model_input = torch.cat([latents] * 2)
                                    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, closest_timestep)
                                    
                                    with torch.no_grad():
                                        noise_pred = pipe.unet(latent_model_input, closest_timestep, encoder_hidden_states=text_embeddings).sample
                                    
                                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                    noise_pred = noise_pred_uncond + 3.5 * (noise_pred_text - noise_pred_uncond)
            
                                    alpha_t_next = pipe.scheduler.alphas_cumprod[700]
                                    latent_input = noise_pred_transfer_unet + (1 - alpha_t_next).sqrt() * noise_pred
                                    result_img = sample("", start_latents=latent_input, start_step=20, num_inference_steps=50)[0][0]
                                    
                                    # 2. Direct fusion result
                                    alpha_t_next_direct_fusion = pipe.scheduler.alphas_cumprod[700]
                                    latent_input_direct_fusion = noise_pred_transfer_unet + (1 - alpha_t_next_direct_fusion).sqrt() * inverted_latents[0][-1]
                                    result_direct_fusion = sample("", start_latents=latent_input_direct_fusion, start_step=18, num_inference_steps=50)[0][0]
                                    
                                    # 3. Result with class prompt
                                    result_with_prompt = sample(class_prompt, start_latents=latent_input, start_step=20, num_inference_steps=50)[0][0]
                                    
                                    # 4. More variance result (using [-2] instead of [-20])
                                    alpha_t_next_more_var = pipe.scheduler.alphas_cumprod[800]
                                    latent_input_v2_more_var = noise_pred_transfer_unet + (1 - alpha_t_next_more_var).sqrt() * inverted_latents[2][-2]
                                    
                                    closest_timestep_more_var, _, _ = find_closest_timestep(latent_input_v2_more_var)
                                    latents_more_var = latent_input_v2_more_var.clone()
                                    latent_model_input_more_var = torch.cat([latents_more_var] * 2)
                                    latent_model_input_more_var = pipe.scheduler.scale_model_input(latent_model_input_more_var, closest_timestep_more_var)
                                    
                                    with torch.no_grad():
                                        noise_pred_more_var = pipe.unet(latent_model_input_more_var, closest_timestep_more_var, encoder_hidden_states=text_embeddings).sample
                                    
                                    noise_pred_uncond_more_var, noise_pred_text_more_var = noise_pred_more_var.chunk(2)
                                    noise_pred_more_var = noise_pred_uncond_more_var + 3.5 * (noise_pred_text_more_var - noise_pred_uncond_more_var)
                                    
                                    alpha_t_next_more_var = pipe.scheduler.alphas_cumprod[700]
                                    latent_input_more_var = noise_pred_transfer_unet + (1 - alpha_t_next_more_var).sqrt() * noise_pred_more_var
                                    result_img_more_var = sample("", start_latents=latent_input_more_var, start_step=20, num_inference_steps=50)[0][0]
                                    
                                    # 5. Result with class prompt on variance added as -2
                                    result_with_prompt_more_var = sample(class_prompt, start_latents=latent_input_more_var, start_step=20, num_inference_steps=50)[0][0]

                                    
                                    # 8. More variance result (using [-15] instead of [-20])
                                    alpha_t_next_more_var = pipe.scheduler.alphas_cumprod[800]
                                    latent_input_v2_more_var = noise_pred_transfer_unet + (1 - alpha_t_next_more_var).sqrt() * inverted_latents[2][-15]
                                    
                                    closest_timestep_more_var, _, _ = find_closest_timestep(latent_input_v2_more_var)
                                    latents_more_var = latent_input_v2_more_var.clone()
                                    latent_model_input_more_var = torch.cat([latents_more_var] * 2)
                                    latent_model_input_more_var = pipe.scheduler.scale_model_input(latent_model_input_more_var, closest_timestep_more_var)
                                    
                                    with torch.no_grad():
                                        noise_pred_more_var = pipe.unet(latent_model_input_more_var, closest_timestep_more_var, encoder_hidden_states=text_embeddings).sample
                                    
                                    noise_pred_uncond_more_var, noise_pred_text_more_var = noise_pred_more_var.chunk(2)
                                    noise_pred_more_var = noise_pred_uncond_more_var + 3.5 * (noise_pred_text_more_var - noise_pred_uncond_more_var)
                                    
                                    alpha_t_next_more_var = pipe.scheduler.alphas_cumprod[700]
                                    latent_input_more_var = noise_pred_transfer_unet + (1 - alpha_t_next_more_var).sqrt() * noise_pred_more_var
                                    result_img_more_var_15 = sample(class_prompt, start_latents=latent_input_more_var, start_step=19, num_inference_steps=50)[0][0]
                                
                                    # 9. NEW: Direct fusion + prompt - use same latent as direct fusion but sample with prompt
                                    result_direct_fusion_with_prompt = sample(class_prompt, start_latents=latent_input_direct_fusion, start_step=20, num_inference_steps=50)[0][0]
                                    
                                    # 10. Direct fusion result with 10th last noise latent
                                    # alpha_t_next_direct_fusion = pipe.scheduler.alphas_cumprod[700]
                                    # latent_input_direct_fusion = noise_pred_transfer_unet + (1 - alpha_t_next_direct_fusion).sqrt() * inverted_latents[0][-10]
                                    # result_direct_fusion_10 = sample(class_prompt, start_latents=latent_input_direct_fusion, start_step=20, num_inference_steps=50)[0][0]
                                    
                                    if target_image is None:
                                        target_image = to_imgs(yb_path[index])
                                    # Create metrics for all results
                                    transform = T.Compose([T.ToTensor(), T.Resize((256, 256))])
                                    result_metric = transform(result_img).unsqueeze(0)
                                    result_direct_fusion_metric = transform(result_direct_fusion).unsqueeze(0)
                                    result_with_prompt_metric = transform(result_with_prompt).unsqueeze(0)
                                    result_more_var_metric = transform(result_img_more_var).unsqueeze(0)
                                    result_with_prompt_more_var_metric = transform(result_with_prompt_more_var).unsqueeze(0)
                                    #new
                                    # result_with_more_var_start_17_metric = transform(result_img_more_var_start_17).unsqueeze(0)
                                    # result_with_prompt_more_var_more_guidance_metric = transform(result_with_prompt_more_var_more_guidance).unsqueeze(0)
                                    result_img_more_var_15_metric = transform(result_img_more_var_15).unsqueeze(0)
                                    # result_direct_fusion_10_metric = transform(result_direct_fusion_10).unsqueeze(0)
                                    result_direct_fusion_with_prompt_metric = transform(result_direct_fusion_with_prompt).unsqueeze(0)  # New metric
                                    target_metric = transform(target_image).unsqueeze(0)
                                    
                                    # Calculate LPIPS for original result
                                    lpips_alex = loss_fn_alex(result_metric, target_metric).item()
                                    lpips_vgg = loss_fn_vgg(result_metric, target_metric).item()
                                    
                                    # Calculate LPIPS for direct fusion result
                                    lpips_alex_direct_fusion = loss_fn_alex(result_direct_fusion_metric, target_metric).item()
                                    lpips_vgg_direct_fusion = loss_fn_vgg(result_direct_fusion_metric, target_metric).item()
                                    
                                    # Calculate LPIPS for result with prompt
                                    lpips_alex_with_prompt = loss_fn_alex(result_with_prompt_metric, target_metric).item()
                                    lpips_vgg_with_prompt = loss_fn_vgg(result_with_prompt_metric, target_metric).item()
                                    
                                    # Calculate LPIPS for more variance result
                                    lpips_alex_more_var = loss_fn_alex(result_more_var_metric, target_metric).item()
                                    lpips_vgg_more_var = loss_fn_vgg(result_more_var_metric, target_metric).item()
                                    
                                    # Calculate LPIPS for result with prompt more variance
                                    lpips_alex_with_prompt_more_var = loss_fn_alex(result_with_prompt_more_var_metric, target_metric).item()
                                    lpips_vgg_with_prompt_more_var = loss_fn_vgg(result_with_prompt_more_var_metric, target_metric).item()
                                    
                                    # Calculate LPIPS for direct fusion with prompt
                                    lpips_alex_direct_fusion_with_prompt = loss_fn_alex(result_direct_fusion_with_prompt_metric, target_metric).item()
                                    lpips_vgg_direct_fusion_with_prompt = loss_fn_vgg(result_direct_fusion_with_prompt_metric, target_metric).item()

                                    # # Calculate LPIPS for more var start 17 (NEW)
                                    # lpips_alex_more_var_start_17 = loss_fn_alex(result_with_more_var_start_17_metric, target_metric).item()
                                    # lpips_vgg_more_var_start_17 = loss_fn_vgg(result_with_more_var_start_17_metric, target_metric).item()

                                    # # Calculate LPIPS for more var and more guidance
                                    # lpips_alex_more_var_more_guidance = loss_fn_alex(result_with_prompt_more_var_more_guidance_metric, target_metric).item()
                                    # lpips_vgg_more_var_more_guidance = loss_fn_vgg(result_with_prompt_more_var_more_guidance_metric, target_metric).item()

                                    # Calculate LPIPS for more var and more guidance
                                    lpips_alex_more_var_15 = loss_fn_alex(result_img_more_var_15_metric, target_metric).item()
                                    lpips_vgg_more_var_15 = loss_fn_vgg(result_img_more_var_15_metric, target_metric).item()

                                    # # Calculate LPIPS for more var and more guidance
                                    # lpips_alex_direct_fusion_10 = loss_fn_alex(result_direct_fusion_10_metric, target_metric).item()
                                    # lpips_vgg_direct_fusion_10 = loss_fn_vgg(result_direct_fusion_10_metric, target_metric).item()
                                    
                                    # psnr = PeakSignalNoiseRatio()
                                    psnr = peak_signal_noise_ratio
                                    # PSNR for original result
                                    psnr_original = psnr(result_metric, target_metric).item()
                                    # PSNR for direct fusion result
                                    psnr_direct_fusion = psnr(result_direct_fusion_metric, target_metric).item()
                                    # PSNR for result with prompt
                                    psnr_with_prompt = psnr(result_with_prompt_metric, target_metric).item()
                                    # PSNR for more variance result
                                    psnr_more_var = psnr(result_more_var_metric, target_metric).item()
                                    # PSNR for result with prompt more variance
                                    psnr_with_prompt_more_var = psnr(result_with_prompt_more_var_metric, target_metric).item()
                                    # PSNR for direct fusion with prompt
                                    psnr_direct_fusion_with_prompt = psnr(result_direct_fusion_with_prompt_metric, target_metric).item()
                                    # PSNR for more var start 17 (NEW)
                                    # psnr_more_var_start_17 = psnr(result_with_more_var_start_17_metric, target_metric).item()
                                    # PSNR for more var and more guidance
                                    # psnr_more_var_more_guidance = psnr(result_with_prompt_more_var_more_guidance_metric, target_metric).item()
                                    # PSNR for more var 15
                                    psnr_more_var_15 = psnr(result_img_more_var_15_metric, target_metric).item()
                                    # PSNR for direct fusion 10
                                    # psnr_direct_fusion_10 = psnr(result_direct_fusion_10_metric, target_metric).item()
                                    
                                    # Check if this checkpoint has a good VGG score on any method
                                    if (lpips_vgg < 0.60 or lpips_vgg_direct_fusion < 0.60 or 
                                        lpips_vgg_with_prompt < 0.60 or lpips_vgg_more_var < 0.60 or 
                                        lpips_vgg_with_prompt_more_var < 0.60 or lpips_vgg_direct_fusion_with_prompt < 0.60  or
                                        lpips_alex_more_var_15 < 0.60):  # Added new check
                                        has_good_score = True
                                    
                                    # Save all LPIPS scores
                                    result["lpips_scores"][epoch] = {"AlexNet": lpips_alex, "VGG": lpips_vgg}
                                    result["lpips_scores_direct_fusion"][epoch] = {"AlexNet": lpips_alex_direct_fusion, "VGG": lpips_vgg_direct_fusion}
                                    result["lpips_scores_with_prompt"][epoch] = {"AlexNet": lpips_alex_with_prompt, "VGG": lpips_vgg_with_prompt}
                                    result["lpips_scores_more_var"][epoch] = {"AlexNet": lpips_alex_more_var, "VGG": lpips_vgg_more_var}
                                    result["lpips_scores_with_prompt_more_var"][epoch] = {"AlexNet": lpips_alex_with_prompt_more_var, "VGG": lpips_vgg_with_prompt_more_var}
                                    result["lpips_scores_direct_fusion_with_prompt"][epoch] = {"AlexNet": lpips_alex_direct_fusion_with_prompt, "VGG": lpips_vgg_direct_fusion_with_prompt} 
                                    # result["lpips_scores_more_var_start_17"][epoch] = {"AlexNet": lpips_alex_more_var_start_17, "VGG": lpips_vgg_more_var_start_17}
                                    # result["lpips_score_with_prompt_more_var_more_guidance"][epoch] = {"AlexNet": lpips_alex_more_var_more_guidance, "VGG": lpips_vgg_more_var_more_guidance} 
                                    result["lpips_score_more_var_15"][epoch] = {"AlexNet": lpips_alex_more_var_15, "VGG": lpips_vgg_more_var_15}
                                    # result["lpips_score_direct_fusion_10"][epoch] = {"AlexNet": lpips_alex_direct_fusion_10, "VGG": lpips_vgg_direct_fusion_10} 
                                    result["psnr_scores"][epoch] = psnr_original
                                    result["psnr_scores_direct_fusion"][epoch] = psnr_direct_fusion
                                    result["psnr_scores_with_prompt"][epoch] = psnr_with_prompt
                                    result["psnr_scores_more_var"][epoch] = psnr_more_var
                                    result["psnr_scores_with_prompt_more_var"][epoch] = psnr_with_prompt_more_var
                                    result["psnr_scores_direct_fusion_with_prompt"][epoch] = psnr_direct_fusion_with_prompt
                                    # result["psnr_scores_more_var_start_17"][epoch] = psnr_more_var_start_17
                                    # result["psnr_score_with_prompt_more_var_more_guidance"][epoch] = psnr_more_var_more_guidance
                                    result["psnr_score_more_var_15"][epoch] = psnr_more_var_15
                                    # result["psnr_score_direct_fusion_10"][epoch] = psnr_direct_fusion_10
            
                                    # Store all result images
                                    all_results[epoch] = {
                                        "image": result_img,
                                        "direct_fusion": result_direct_fusion,
                                        "with_prompt": result_with_prompt,
                                        "more_var": result_img_more_var,
                                        "with_prompt_more_var": result_with_prompt_more_var,
                                        "direct_fusion_with_prompt": result_direct_fusion_with_prompt,
                                        # "more_var_start_17": result_img_more_var_start_17,
                                        # "more_var_more_guidance" : result_with_prompt_more_var_more_guidance,
                                        "more_var_15" : result_img_more_var_15,
                                        # "direct_fusion_10" : result_direct_fusion_10,  # Added new result
                                        "scores": {"AlexNet": lpips_alex, "VGG": lpips_vgg},
                                        "scores_direct_fusion": {"AlexNet": lpips_alex_direct_fusion, "VGG": lpips_vgg_direct_fusion},
                                        "scores_with_prompt": {"AlexNet": lpips_alex_with_prompt, "VGG": lpips_vgg_with_prompt},
                                        "scores_more_var": {"AlexNet": lpips_alex_more_var, "VGG": lpips_vgg_more_var},
                                        "scores_with_prompt_more_var": {"AlexNet": lpips_alex_with_prompt_more_var, "VGG": lpips_vgg_with_prompt_more_var},
                                        "scores_direct_fusion_with_prompt": {"AlexNet": lpips_alex_direct_fusion_with_prompt, "VGG": lpips_vgg_direct_fusion_with_prompt},
                                        # "scores_more_var_start_17" : {"AlexNet": lpips_alex_more_var_start_17, "VGG" : lpips_vgg_more_var_start_17 },
                                        # "scores_more_var_more_guidance" : {"AlexNet":lpips_alex_more_var_more_guidance, "VGG" : lpips_vgg_more_var_more_guidance},
                                        "scores_more_var_15" : {"AlexNet": lpips_alex_more_var_15, "VGG" : lpips_vgg_more_var_15},
                                        # "scores_direct_fusion_10":{"AlexNet" : lpips_alex_direct_fusion_10, "VGG" : lpips_vgg_direct_fusion_10}  # Added new scores
                                    }
                                    print(result)
                                    print(all_results)
                                    target_image = (target_image * 255).clip(0, 255).astype(np.uint8)
                                    target_pil_image = Image.fromarray(target_image)
                                    input_image = (input_image * 255).clip(0, 255).astype(np.uint8)
                                    input_pil_image = Image.fromarray(input_image)
                                    if os.path.exists(aggregated_results_path):
                                        with open(aggregated_results_path, "r") as f:
                                            results = json.load(f)
                                    results.append(result)
                                    with open(aggregated_results_path, "w") as f:
                                            json.dump(results, f, indent=4)
                                    save_consolidated_image_six_methods(input_pil_image, target_pil_image, all_results, global_idx, class_labels[index].item(), class_prompt, save_dir, epochs,epoch=epoch)
            
                    
                    
                    
                
    
                # Accumulate the training loss
                val_loss += loss_val.item()
                # distribution_loss_epoch += mmd_loss/len(inverted_latent_camera_dl)
                mse_loss_epoch_val += mse_loss_val / len(inverted_latent_camera_dl_val)
                tepoch.set_postfix(loss=loss_val.item())
    val_loss_epoch = val_loss / len(inverted_latent_camera_dl_val)
    val_losses.append(val_loss_epoch)
    if (epoch + 1) % 20 == 0:
                    # save_path = f"ckpt/smaller_model_60m_mse_camera_param_512_var_only/transfer_unet_{epoch+1}"
                    # torch.save({'epoch': epoch + 1,'model_state_dict': TransferUnet.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'scheduler': lr_scheduler.state_dict(),'loss' : train_loss_epoch}, save_path)
                    plot_losses(train_losses, val_losses, save_path="/home/sehajs/mvimgnet_3_classes/ckpt/loss_plot.png")
                    new_row = pd.DataFrame([{
                                    'epoch': epoch,
                                    'train_loss': train_loss_epoch,
                                    'train_mse_loss': mse_loss_epoch.item(),
                                    'val_loss' : val_loss_epoch,
                                    'val_mse_loss' : mse_loss_epoch_val.item()
                                }])
        
                    # Concatenate the new row with the existing DataFrame
                    loss_df = pd.concat([loss_df, new_row], ignore_index=True)
                    
                    # Save the DataFrame to CSV (overwrite each time)
                    loss_df.to_csv("/home/sehajs/mvimgnet_3_classes/ckpt/losses.csv", index=False)            
    
    print(f"Epoch {epoch+1}, Val Loss: {val_loss_epoch:.4f},MSE Loss:{mse_loss_epoch_val:.4f}")
    # print(f"Epoch {epoch+1},MSE Loss:{mse_loss_epoch_val:.4f}")