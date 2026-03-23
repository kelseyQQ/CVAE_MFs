import torch
import os
from torch.utils.data import Dataset
import numpy as np
    
class MultiFolderMinkowskiDataset6(Dataset):
    def __init__(self, folder_paths):
        """
        Custom dataset to load images and their corresponding Minkowski functionals from multiple folders.
        
        Args:
            folder_paths (list): List of folder paths to include in the dataset.
        """
        self.folder_paths = folder_paths  # Store folder paths
        self.samples = []
        self.folder_indices = []  # Track which folder each sample comes from
        
        for folder_idx, folder in enumerate(folder_paths):
            # Iterate over .dat files in the folder
            for filename in os.listdir(folder):
                if filename.endswith(".dat"):
                    filepath = os.path.join(folder, filename)
                    image, mfs = self._parse_file(filepath)
                    self.samples.append((image, mfs))
                    self.folder_indices.append(folder_idx)

    def _parse_file(self, filepath):
        """
        Parse a single .dat file to extract the binary image and Minkowski functionals.

        Expected file format:
        - Lines 0~39: Binary image (40x40, "True"/"False")
        - Lines 40~42: Local Minkowski Functionals (3 lines, 1 or more values)
        - Line 43: Global Minkowski Functionals (1 line with 3 values)
        - Line 44: Sigma values (1 line with 3 values)
        """
        with open(filepath, "r") as f:
            lines = f.read().strip().split("\n")

        # 1. Parse image (40x40)
        image_data = []
        for line in lines[:40]:
            row = [1.0 if val == "True" else 0.0 for val in line.split()]
            image_data.append(row)
        image_array = np.array(image_data, dtype=np.float32)
        image_tensor = torch.tensor(image_array).unsqueeze(0)  # [1, 40, 40]

        # 2. Parse Minkowski functionals
        # Lines 40~42: Local MFs
        local_mfs = []
        for i in range(40, 43):
            local_mfs.extend([float(val) for val in lines[i].split()])

        # Line 43: Global MFs
        global_mfs = [float(val) for val in lines[43].split()]

        # Line 44: Sigma
        sigma = [float(val) for val in lines[45].split()]

        # Combine all into one tensor [9]
        all_mfs = torch.tensor(local_mfs + global_mfs + sigma, dtype=torch.float32)  # [9]

        return image_tensor, all_mfs


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, all_mfs = self.samples[idx]  # all_mfs.shape = [9]

        local_mfs = all_mfs[:3]
        global_mfs = all_mfs[3:6]
        sigma = all_mfs[6:9]

        return image, local_mfs, global_mfs, sigma
