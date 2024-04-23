import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def black_level_subtraction(image, black_level=2048):
    """ Subtract black level and clamp to zero if negative. """
    array = np.array(image).astype(np.int32)  # Ensure we have enough bit depth
    array = array - black_level
    array[array < 0] = 0
    return Image.fromarray(array.astype(np.uint16))  # Convert back to 16-bit

def saturation_detection(image, threshold=50):
    """ Set saturated pixels to zero. """
    array = np.array(image).astype(np.int32)
    max_val = array.max() - threshold
    mask = np.any(array >= max_val, axis=-1)
    array[mask] = 0
    return Image.fromarray(array.astype(np.uint16))

def color_target_masking(image, rect_size=(700, 1000)):
    """ Mask out a rectangle in the lower right corner. """
    array = np.array(image).astype(np.int32)
    h, w, _ = array.shape
    array[h-rect_size[0]:, w-rect_size[1]:, :] = 0
    return Image.fromarray(array.astype(np.uint16))

class CubePPDataset(Dataset):
    def __init__(self, directory, gt_path, transform=None):
        """
        Initialize dataset.
        
        Args:
            directory (string): Path to the directory containing images.
            gt_path (string): Path to the ground-truth CSV file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.directory = directory
        self.transform = transform
        self.filenames = [f for f in os.listdir(directory) if f.endswith('.png')]
        self.ground_truth = pd.read_csv(gt_path)

    def __len__(self):
        """ Return the total number of images in the dataset. """
        return len(self.filenames)

    def __getitem__(self, idx):
        """ Retrieve an image and its label by index. """
        img_name = os.path.join(self.directory, self.filenames[idx])
        img = Image.open(img_name).convert('RGB')  # Convert to RGB to handle as 3 channel image
        
        # Apply preprocessing steps
        img = black_level_subtraction(img)
        img = saturation_detection(img)
        img = color_target_masking(img)
        
        # Apply other transformations
        if self.transform:
            img = self.transform(img)

        # Get ground-truth data
        ground_truth = self.ground_truth.iloc[idx]

        return img, ground_truth
