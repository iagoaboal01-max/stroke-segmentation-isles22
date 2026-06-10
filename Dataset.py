import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd
import nibabel as nib
import numpy as np


class Dataset_v1(Dataset):
    def __init__(self, csv_file, image_size, base_path, transform=True, name="unnamed dataset"):
        self.data = pd.read_csv(csv_file)
        self.do_transform = transform
        self.size = image_size
        self.base_path = base_path
        self.name = name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_vol  = nib.load(self.base_path + row['img_path']).get_fdata()
        mask_vol = nib.load(self.base_path + row['mask_path']).get_fdata()

        img_vol  = np.transpose(img_vol,  (2, 0, 1))   # (H,W,D) → (D,H,W)
        mask_vol = np.transpose(mask_vol, (2, 0, 1))

        if self.do_transform:
            img_vol  = self.transformation(img_vol,  'image')
            mask_vol = self.transformation(mask_vol, 'mask')

        img_vol  = torch.from_numpy(img_vol).unsqueeze(0)    # (1,D,H,W)
        mask_vol = torch.from_numpy(mask_vol).unsqueeze(0)   # (1,D,H,W)

        # FIX: return idx so predict() can unpack (images, masks, idx) regardless
        #      of which dataset class is used — was missing here, only present in
        #      Dataset_combined.
        return img_vol, mask_vol, idx

    def transformation(self, img: np.ndarray, img_type: str) -> np.ndarray:
        img = img.astype(np.float32)
        img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)

        mode = "nearest" if img_type == "mask" else "trilinear"

        if len(self.size) == 3:
            target_size = self.size
        elif len(self.size) == 2:
            # Keep the original depth (axis 0), resize H and W only.
            # img.shape is (D,H,W) → img.shape[0] = D.
            target_size = (img.shape[0], self.size[0], self.size[1])
        else:
            raise ValueError("image_size must be a tuple of length 2 or 3 for 3D volumes")

        kwargs = {"size": target_size, "mode": mode}
        if mode != "nearest":
            kwargs["align_corners"] = False

        img_tensor = F.interpolate(img_tensor, **kwargs)
        img = img_tensor.squeeze(0).squeeze(0).numpy()  # back to (D,H,W)

        if img_type == "image":
            img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-7)
            return img.astype(np.float32)
        else:
            return img.astype(np.int64)


class Dataset_combined(Dataset):

    def __init__(self, csv_file, image_size, base_path,
                 modalities=['img_path'], transform=True, name="unnamed dataset"):
        self.data = pd.read_csv(csv_file)
        self.do_transform = transform
        self.size = image_size
        self.base_path = base_path
        self.name = name
        self.modalities = modalities  # list of column names in the CSV

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        channels = []
        for col in self.modalities:
            img_vol = nib.load(self.base_path + row[col]).get_fdata()
            img_vol = np.transpose(img_vol, (2, 0, 1))
            if self.do_transform:
                img_vol = self.transformation(img_vol, 'image')
            channels.append(img_vol)

        if len(channels) == 1:
            img_vol = torch.from_numpy(channels[0]).unsqueeze(0)    # (1,D,H,W)
        else:
            img_vol = torch.from_numpy(np.stack(channels, axis=0))  # (C,D,H,W)

        mask_vol = nib.load(self.base_path + row['mask_path']).get_fdata()
        mask_vol = np.transpose(mask_vol, (2, 0, 1))
        if self.do_transform:
            mask_vol = self.transformation(mask_vol, 'mask')
        mask_vol = torch.from_numpy(mask_vol).unsqueeze(0)  # (1,D,H,W)

        return img_vol, mask_vol, idx

    def transformation(self, img: np.ndarray, img_type: str) -> np.ndarray:
        img = img.astype(np.float32)
        img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)

        mode = "nearest" if img_type == "mask" else "trilinear"

        # -----------------------------
        # TARGET SIZE LOGIC
        # -----------------------------
        if len(self.size) == 3:
            target_d, target_h, target_w = self.size
            current_d, current_h, current_w = img.shape

            # 1. Resize XY ONLY (keep depth unchanged)
            xy_resized = F.interpolate(
                img_tensor,
                size=(current_d, target_h, target_w),
                mode="nearest" if img_type == "mask" else "trilinear",
                align_corners=False if img_type != "mask" else None
            )

            # 2. PAD / CROP depth
            if current_d < target_d:
                pad = target_d - current_d
                pad_tensor = torch.zeros(
                    (1, 1, pad, target_h, target_w),
                    dtype=xy_resized.dtype
                )
                img_tensor = torch.cat([xy_resized, pad_tensor], dim=2)

            elif current_d > target_d:
                img_tensor = xy_resized[:, :, :target_d, :, :]

            else:
                img_tensor = xy_resized

        elif len(self.size) == 2:
            target_size = (img.shape[0], self.size[0], self.size[1])
            kwargs = {"size": target_size, "mode": mode}
            if mode != "nearest":
                kwargs["align_corners"] = False

            img_tensor = F.interpolate(img_tensor, **kwargs)

        else:
            raise ValueError("image_size must be tuple of length 2 or 3")

        # -----------------------------
        # BACK TO NUMPY
        # -----------------------------
        img = img_tensor.squeeze(0).squeeze(0).numpy()

        # -----------------------------
        # NORMALIZATION / MASK CAST
        # -----------------------------
        if img_type == "image":
            img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-7)
            return img.astype(np.float32)
        else:
            return img.astype(np.int64)
