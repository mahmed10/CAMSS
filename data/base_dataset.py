import os
import random
import time
import numpy as np
import torch
from PIL import Image
import cv2

from torchvision import transforms, datasets
from torchvision.transforms import functional as F

def _getvocpallete(num_colors):
    return [0, 0, 0] * num_colors


# ----------------------
# Augmentation Classes
# ----------------------
class Rotate:
    def __init__(self, angle):
        self.angle = random.randint(-angle, angle)

    def __call__(self, img):
        return F.rotate(img, angle=self.angle)
    
class Shear:
    def __init__(self, shear=10, scale=(1.0, 1.0)):
        self.shear = random.uniform(-shear, shear)
        self.scale = random.uniform(scale[0], scale[1])

    def __call__(self, img):
        return F.affine(img, angle=0, translate=(0, 0), scale=self.scale, shear=[self.shear, self.shear])
    
class Skew:
    def __init__(self, magnitude=0.2):
        self.xshift = random.uniform(-magnitude, magnitude)
        self.yshift = random.uniform(-magnitude, magnitude)

    def __call__(self, img):
        width, height = img.size
        x_shift = int(self.xshift * width)
        y_shift = int(self.yshift * height)
        return img.transform(img.size, Image.AFFINE, (1, 0, x_shift, 0, 1, y_shift))
    
class Crop:
    def __init__(self, min_crop=0.8, max_crop=0.9):
        self.crop_scale = random.uniform(min_crop, max_crop)
        self.seed = time.time()

    def __call__(self, img):
        width, height = img.size        
        crop_width = int(self.crop_scale * width)
        crop_height = int(self.crop_scale * height)
        
        random.seed(self.seed)
        left = random.randint(0, width - crop_width)
        top = random.randint(0, height - crop_height)
        return F.crop(img, top, left, crop_height, crop_width).resize((width, height))
    

class GaussianNoise:
    def __init__(self, mean=0, std=(10,20)):
        self.mean = mean
        self.std = random.uniform(std[0], std[1])

    def __call__(self, img):
        img = np.array(img)

        noise = np.random.normal(self.mean, self.std, img.shape)
        img = img + noise
        img = np.clip(img, 0, 255).astype(np.uint8)
        return Image.fromarray(img)
    

class SaltAndPepperNoise:
    def __init__(self, min_prob=0.01, max_prob=0.05):
        self.salt_prob = random.uniform(min_prob, max_prob)
        self.pepper_prob = random.uniform(min_prob, max_prob)

    def __call__(self, img):
        img_array = np.array(img)

        salt_mask = np.random.rand(*img_array.shape[:2]) < self.salt_prob
        pepper_mask = np.random.rand(*img_array.shape[:2]) < self.pepper_prob
        img_array[salt_mask] = 255
        img_array[pepper_mask] = 0
        return Image.fromarray(img_array.astype(np.uint8))
    
class MotionBlur:
    def __init__(self, min_size=3, max_size=21):
        self.kernel_size = random.randint(min_size, max_size)

    def __call__(self, img):
        img_array = np.array(img)

        kernel = np.zeros((self.kernel_size, self.kernel_size))
        kernel[int((self.kernel_size - 1) / 2), :] = np.ones(self.kernel_size)
        kernel = kernel / self.kernel_size
        blurred = cv2.filter2D(img_array, -1, kernel)
        return Image.fromarray(blurred.astype(np.uint8))
    
class HideAndSeekNoise:
    def __init__(self, min_size=90, max_size=190):
        self.patch_size = random.randint(min_size, max_size)
        self.seed = time.time()

    def __call__(self, img):
        img_array = np.array(img)
        height, width, _ = img_array.shape
        
        random.seed(self.seed)
        top = random.randint(0, height - self.patch_size)
        left = random.randint(0, width - self.patch_size)
        img_array[top:top + self.patch_size, left:left + self.patch_size] = [0, 0, 0]
        return Image.fromarray(img_array)
    


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, path_list, transform = None, data_set = 'val', seed=None,
                img_size=768, interpolation=Image.BILINEAR, color_pallete = 'city'):
        """
        :param path_list: Path to file listing image paths.
        :param transform: Additional torchvision transforms.
        :param data_set: 'train' or other mode.
        :param seed: Seed for shuffling.
        :param img_size: Resize dimensions.
        :param interpolation: Interpolation method for resizing.
        """
        self.transform = transform
        self.data_set = data_set
        self.color_pallete = color_pallete

        with open(path_list, "r") as file:
            self.imgs = file.readlines()

        if seed:
            random.seed(seed)
            random.shuffle(self.imgs)

        self.masks = [img_path for img_path in self.imgs]
        self.learning_map = None
        
        self.aug_weights = [0.4, 0.3, 0.3, 0.2, 0.2, 0.05, 0.05, 0.02, 0.02]
        if img_size:
            self.transform_resize = transforms.Resize((img_size, img_size), interpolation=Image.BILINEAR)

    def convert_label(self, label, inverse=False):
        temp = label.copy()
        converted_label = np.zeros_like(label)
        for k, v in self.learning_map.items():
            converted_label[temp == k] = v
        return converted_label

    def get_color_pallete(self, npimg, dataset='city'):
        out_img = Image.fromarray(npimg.astype('uint8')).convert('P')
        if dataset == 'city':
            cityspallete = [
                0, 0, 0,
                128, 64, 128,
                244, 35, 232,
                70, 70, 70,
                102, 102, 156,
                190, 153, 153,
                153, 153, 153,
                250, 170, 30,
                220, 220, 0,
                107, 142, 35,
                152, 251, 152,
                0, 130, 180,
                220, 20, 60,
                255, 0, 0,
                0, 0, 142,
                0, 0, 70,
                0, 60, 100,
                0, 80, 100,
                0, 0, 230,
                119, 11, 32,
            ]
            out_img.putpalette(cityspallete)
        else:
            vocpallete = _getvocpallete(256)
            out_img.putpalette(vocpallete)
        return out_img.convert("RGB")

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index].rstrip(), self.masks[index].rstrip()

        # Load and resize the image
        img = Image.open(img_path).convert('RGB')
        img = self.transform_resize(img)
        
        # Load, convert, and resize the mask
        mask = Image.open(mask_path)
        mask = np.array(mask)
        mask = self.convert_label(mask)
        mask = mask.astype(np.uint8)
        mask = self.get_color_pallete(mask, self.color_pallete)
        mask = self.transform_resize(mask)
        
        # Augmentation stage
        augmentation_num = random.choices(range(9), weights=self.aug_weights, k=1)[0] if self.data_set == 'train' else 0
        if augmentation_num > 0:
            augmentation_set = [
                transforms.RandomHorizontalFlip(p=1),  # Flip horizontally
                transforms.RandomVerticalFlip(p=1),  # Flip vertically
                Crop(min_crop=0.6, max_crop=0.9),  # Random crop
                Rotate(angle=90),  # Rotate
                Shear(shear=10, scale=(0.8, 1.2)),  # Shear
                Skew(magnitude=0.2),  # Skew
                HideAndSeekNoise(min_size=90, max_size=210), #Hide and seek noise
                GaussianNoise(mean=0, std=(5,20)), # Gaussian noise (only for image) / std 10-20
                SaltAndPepperNoise(min_prob=0.01, max_prob=0.03), #Salt and pepper noise (only for image) 
                transforms.GaussianBlur(kernel_size=3, sigma=(0.2, 1)),  # Gaussian blur (only for image)
                MotionBlur(min_size=3, max_size=15), # Motion blur (only for image)
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Color jitter (only for image)
            ]
            random.shuffle(augmentation_set)
            augmentation_set = augmentation_set[:augmentation_num]
            for aug in augmentation_set:
                if isinstance(aug, (transforms.GaussianBlur, transforms.ColorJitter, GaussianNoise, SaltAndPepperNoise, MotionBlur)):
                    img = aug(img)
                else:
                    img = aug(img)
                    mask = aug(mask)

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        return img, mask, img_path

    def __len__(self):
        return len(self.imgs)