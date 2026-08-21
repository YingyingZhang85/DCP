import os
import random
import pandas as pd
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler
import scipy.ndimage as ndimage

class DTI_Sequence_Dataset(Dataset):
    def __init__(self, df, data_root, modalities=['FA', 'MD', 'RD', 'AD'], mode='train'):
        self.data_root = data_root
        self.modalities = modalities
        self.mode = mode
        self.is_train = (mode == 'train') 
        self.df = df.copy()
        df = self.df
        print(f"[{mode.upper()}] Loading MULTIMODAL dataset: {self.modalities}...")

        # Extract valid rows with values 1, 2, 3
        df = df[df['DIAGNOSIS'].isin([1, 2, 3])].copy()
        
        # Use DIAGNOSIS directly as PyTorch Label, no extra mapping needed
        df['Label'] = df['DIAGNOSIS'].astype(int)
        
        df = df.dropna(subset=['Label', 'Subject (PTID)', 'Acq Date', 'Image ID'])

        # Use YYYYMMDD integer directly as time axis, preserving sort order without date library conversion
        df['Time_Point'] = df['Acq Date'].astype(float)

        self.subjects = []
        self.subject_records = {}

        for subj, group in df.groupby('Subject (PTID)'):
            # Sort by acquisition date to ensure monotonic time progression
            records = group.sort_values('Time_Point').to_dict('records')
            valid_records = [r for r in records if self._get_image_paths(r) is not None]
            
            if len(valid_records) > 0:
                self.subjects.append(subj)
                self.subject_records[subj] = valid_records
                
        # Assign internal numeric IDs for subject-specific matrix masks
        self.subj_to_id = {subj: i for i, subj in enumerate(self.subjects)}
        
        # Record baseline labels (V0 diagnosis) for balanced sampling
        self.subject_baseline_labels = {}
        for subj, records in self.subject_records.items():
            self.subject_baseline_labels[subj] = records[0]['Label']
            
        print(f"[{mode.upper()}] Init complete: Extracted {len(self.subjects)} longitudinal subjects.")

        self.record_paths = {}  # Dict structure: subj -> [path_v0, path_v1, ...]
        
        # Iterate over all longitudinal records for all subjects
        for subj, records in self.subject_records.items():
            visit_paths = []
            for r in records:
                paths = self._get_image_paths(r)
                
                # Fail-Fast: ensure all 4 modalities exist for current visit
                for p in paths:
                    if not os.path.exists(p):
                        raise FileNotFoundError(f"Cannot find preprocessed clean file -> \n{p}")
                
                visit_paths.append(paths)
            
            self.record_paths[subj] = visit_paths
        
        print(f"Successfully cached {len(self.record_paths)} subjects' 4D multimodal paths!")
    
    def _get_image_paths(self, record):
        """O(1) pure in-memory string concatenation, no disk search"""
        subj = record['Subject (PTID)']
        img_id = record['Image ID']
        folder_name = f"{subj}_{img_id}"
        
        paths = []
        for mod in self.modalities:
            if mod == 'PET':
                file_name = f"res_AES_{mod}_MNI_Z.nii.gz"
            else:
                file_name = f"res_AES_{subj}_{img_id}_{mod}_reg.nii.gz"
            paths.append(os.path.join(self.data_root, folder_name, file_name))
        return paths

    def _load_nifti_raw(self, path):
        """Read and normalize NIfTI file"""
        target_shape = (91, 109, 91)
        if path is None: return None
        try:
            img_obj = nib.load(path)
            data = img_obj.get_fdata(dtype=np.float32)
            
            # Clean redundant dimensions
            if data.ndim == 4 and data.shape[-1] == 1: data = data.squeeze(-1)
            if data.ndim != 3: return None 

            tensor = torch.from_numpy(data).unsqueeze(0)
            
            # Force resize to standard MNI dimensions
            if tensor.shape[1:] != target_shape:
                tensor = F.interpolate(
                    tensor.unsqueeze(0), 
                    size=target_shape, 
                    mode='trilinear', 
                    align_corners=False
                ).squeeze(0)
            
            return tensor

        except Exception as e:
            print(f"\nLoad Error: {e}")
            return None

    # Load and stack multimodal data
    def _load_multimodal(self, paths):
        """Read 4 individual files and concatenate into (4, D, H, W) tensor"""
        tensors = []
        for p in paths:
            t = self._load_nifti_raw(p)
            if t is None: return None
            tensors.append(t)
        
        # Concatenate 4 tensors of (1, D, H, W) into (4, D, H, W)
        return torch.cat(tensors, dim=0)

    def _augment(self, tensor):
        """Joint multimodal data augmentation. Input: Tensor (4, D, H, W)"""
        # Convert to numpy for geometric transformations (D, H, W)
        data = tensor.numpy()
        
        # --- 1. Small angle rotation (Simulate positioning errors) ---
        # Rotate between -5 and +5 degrees in the Axial plane
        if random.random() > 0.5:
            angle = random.uniform(-5, 5)
            # Note: data shape is (C, D, H, W), so spatial axes are 2 and 3 (H and W)
            data = ndimage.rotate(data, angle, axes=(2, 3), reshape=False, order=1, mode='constant', cval=0.0)

        # Use FA channel (index 0) to compute global mask
        mask = np.abs(data[0]) > 1e-5

        if mask.sum() > 0:
            # 4. Channel-adaptive noise injection: inject magnitude-matched noise for each modality
            for c in range(data.shape[0]):
                c_voxels = data[c, mask]
                
                if random.random() > 0.5:
                    # Dynamically compute proportional offset to prevent 0.03 absolute offset from corrupting MD (MD total value ~0.001)
                    c_max = np.abs(c_voxels).max()
                    shift = random.uniform(-0.03, 0.03) * c_max
                    data[c, mask] += shift

                if random.random() > 0.5:
                    # Dynamically generate noise based on channel standard deviation
                    c_std = np.std(c_voxels)
                    noise = np.random.normal(0, 0.05 * c_std, size=c_voxels.shape)
                    data[c, mask] += noise
            
        return torch.from_numpy(data.copy()).float()

    def _normalize(self, tensor):
        """5. Independent channel normalization (Channel-wise Z-score). Input: Tensor (4, D, H, W)"""
        # Shared mask computed from FA channel
        # Explicitly use FA channel (index 0) to generate 3D mask
        mask = torch.abs(tensor[0]) > 1e-5 

        if mask.sum() > 0:
            # Perform residual clipping and Z-score for FA, MD, RD, AD separately
            for c in range(tensor.shape[0]):
                c_voxels = tensor[c, mask]
                
                p_low = torch.quantile(c_voxels, 0.02)
                p_high = torch.quantile(c_voxels, 0.98)
                
                tensor[c, mask] = torch.clamp(tensor[c, mask], min=p_low.item(), max=p_high.item())
                
                brain_mean = tensor[c, mask].mean()
                brain_std = tensor[c, mask].std()
                tensor[c, mask] = (tensor[c, mask] - brain_mean) / (brain_std + 1e-8)
                
                # Force-clean background again to eliminate edge blur from resampling or augmentation
                tensor[c, ~mask] = 0.0
            
        return tensor

    def get_visit_count(self, idx):
        return len(self.subject_records[self.subjects[idx]])

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subj = self.subjects[idx]
        records = self.subject_records[subj]
        subj_id = self.subj_to_id[subj]
        paths = self.record_paths[subj]

        imgs, times, labels, dates = [], [], [], []
        for r, p_list in zip(records, paths):
            # p_list contains [FA_path, MD_path, RD_path, AD_path]
            img_tensor = self._load_multimodal(p_list)

            if img_tensor is not None:
                if self.is_train:
                    img_tensor = self._augment(img_tensor)

                img_tensor = self._normalize(img_tensor)

                imgs.append(img_tensor)
                times.append(r['Time_Point'])
                labels.append(r['Label'])
                dates.append(str(int(r['Acq Date'])))
        return {
            'imgs': torch.stack(imgs) if len(imgs) > 0 else torch.empty(0),
            'times': torch.tensor(times, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long),
            'subj_ids': torch.full((len(imgs),), subj_id, dtype=torch.long),
            'dates': dates
        }


class DynamicImageSampler(Sampler):
    """
    Balanced sampler based on baseline diagnosis buckets.
    Uses an oversampling strategy for minority classes to ensure image-level parity.
    """
    def __init__(self, dataset, max_images=96, shuffle=True):
        self.dataset = dataset
        self.max_images = max_images
        self.shuffle = shuffle
        
        self.patient_indices = list(range(len(dataset)))
        self.visit_counts = [dataset.get_visit_count(i) for i in self.patient_indices]
        
        unique_labels = set(self.dataset.subject_baseline_labels.values())
        self.buckets = {label: [] for label in unique_labels}
        for idx in self.patient_indices:
            subj_name = self.dataset.subjects[idx]
            label = self.dataset.subject_baseline_labels[subj_name]
            self.buckets[label].append(idx)
            
        self.active_labels = [k for k, v in self.buckets.items() if len(v) > 0]

        self.balanced_total_imgs = sum(self.visit_counts)
        
    def __iter__(self):
        working_buckets = {}
        for label in self.active_labels:
            working_buckets[label] = self.buckets[label].copy()
            if self.shuffle:
                random.shuffle(working_buckets[label])
                
        batch = []
        current_img_counts = {label: 0 for label in self.active_labels}
        yielded_imgs = 0
        
        # Core fix: target balanced_total_imgs
        while yielded_imgs < self.balanced_total_imgs:
            # Greedy strategy: Pick the class with the least images in current batch
            min_label = min(current_img_counts, key=current_img_counts.get)
            
            # Oversampling: Refill bucket if empty
            if len(working_buckets[min_label]) == 0:
                working_buckets[min_label] = self.buckets[min_label].copy()
                if self.shuffle:
                    random.shuffle(working_buckets[min_label])

            idx = working_buckets[min_label].pop()
            v_count = self.visit_counts[idx]
            total_current = sum(current_img_counts.values())
            
            # Memory safety: check if adding this subject exceeds max_images
            if total_current + v_count > self.max_images and len(batch) > 0:
                yield batch
                yielded_imgs += total_current
                batch = [idx]
                current_img_counts = {label: 0 for label in self.active_labels}
                current_img_counts[min_label] += v_count
            else:
                batch.append(idx)
                current_img_counts[min_label] += v_count
                
        if len(batch) > 0:
            yield batch

    def __len__(self):
        # Number of batches for DataLoader, based on balanced total
        return self.balanced_total_imgs // self.max_images + 1

def dynamic_visit_collate(batch):
    """Concatenates variable-length longitudinal visits into flat batches"""
    all_imgs, all_times, all_labels, all_subj_ids, all_dates = [], [], [], [], []
    for item in batch:
        imgs = item['imgs']
        if imgs.shape[0] == 0: continue
        all_imgs.append(imgs)
        all_times.append(item['times'])
        all_labels.append(item['labels'])
        all_subj_ids.append(item['subj_ids'])
        all_dates.extend(item['dates'])
    return {
        'imgs': torch.cat(all_imgs, dim=0),           # (N, 4, 91, 109, 91)
        'times': torch.cat(all_times, dim=0),         # (N,)
        'labels': torch.cat(all_labels, dim=0),       # (N,)
        'subj_ids': torch.cat(all_subj_ids, dim=0),   # (N,)
        'dates': all_dates
    }