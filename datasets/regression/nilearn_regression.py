import os
import pandas as pd
import numpy as np
import nibabel as nib
from tqdm import tqdm
from joblib import Parallel, delayed

from nilearn.datasets import load_mni152_template, load_mni152_brain_mask
from nilearn.image import resample_img
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Standalone save function to avoid memory leaks during Joblib parallel serialization
def save_single_image(voxel_row, rec, mask, affine, target_dir):
    vol = np.zeros(mask.shape, dtype=np.float32)
    vol[mask] = voxel_row
    new_img = nib.Nifti1Image(vol, affine)
    
    new_folder_path = os.path.join(target_dir, rec['folder'])
    os.makedirs(new_folder_path, exist_ok=True)
    
    new_file_name = f"res_AES_{rec['file']}"
    new_img.to_filename(os.path.join(new_folder_path, new_file_name))
    return True

def process_and_residualize():
    print("Starting normative modeling voxel-level residualization pipeline (pure NumPy accelerated version)...")

    # ==========================================
    # 1. Path and parameter configuration
    # ==========================================
    CSV_PATH = os.path.join(PROJECT_ROOT, 'datasets', 'csv', '5_folds', 'dataset_5_folds_DTI.csv')
    SOURCE_DIR = os.environ.get('DTI_SOURCE_DIR', os.path.join(PROJECT_ROOT, 'data', 'Processed_DTI'))
    TARGET_DIR = os.environ.get('DTI_TARGET_DIR', os.path.join(PROJECT_ROOT, 'data', 'dti_residuals'))
    MODALITIES = ['FA', 'MD', 'RD', 'AD'] 
    os.makedirs(TARGET_DIR, exist_ok=True)

    # ==========================================
    # 2. Read CSV and extract covariates
    # ==========================================
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=['Subject (PTID)', 'Image ID', 'Age', 'PTEDUCAT', 'Sex', 'Group', 'DIAGNOSIS'])

    valid_subjects = []
    valid_confounds = []
    valid_records = []
    is_cn_mask = []

    print("Scanning files and extracting confounding factors...")
    for _, row in df.iterrows():
        subj, img_id, sex_val = row['Subject (PTID)'], row['Image ID'], row['Sex']
        
        folder_name = f"{subj}_{img_id}"

        all_exist = all(os.path.exists(os.path.join(SOURCE_DIR, folder_name, f"{subj}_{img_id}_{m}_reg.nii.gz")) for m in MODALITIES)
        
        if all_exist:
            valid_subjects.append({'subj': subj, 'img_id': img_id, 'folder': folder_name})
            sex_numeric = 1.0 if str(sex_val).strip().upper().startswith('M') else 0.0
            valid_confounds.append([float(row['Age']), float(row['PTEDUCAT']), sex_numeric])
            is_cn_mask.append(str(row['Group']).strip() == 'CN' and int(row['DIAGNOSIS']) == 1)

    N = len(valid_subjects)
    if N == 0:
        raise ValueError("No valid files found!")
    
    confounds = np.array(valid_confounds, dtype=np.float32)
    is_cn_mask = np.array(is_cn_mask)
    print(f"Successfully found {N} valid images, including {is_cn_mask.sum()} CN group images.")

    # ==========================================
    # Global Mask (MNI standard mask + FA physical filtering)
    # ==========================================
    print("Generating global strict mask using MNI official mask and FA images...")
    mni_template = load_mni152_template(resolution=2)
    target_affine = mni_template.affine
    target_shape = (91, 109, 91)

    # Load MNI official brain mask
    mni_mask_img = load_mni152_brain_mask(resolution=2)

    # Resample official 99x117x95 mask to our 91x109x91 space
    resampled_mask_img = resample_img(
        mni_mask_img, 
        target_affine=target_affine, 
        target_shape=target_shape, 
        interpolation='nearest'  # Critical: Mask is binary, must use nearest to avoid fractional boundaries!
    )

    official_mask = resampled_mask_img.get_fdata() > 0
    
    sum_data = np.zeros(target_shape, dtype=np.float32)
    for item in tqdm(valid_subjects, desc="Reading FA for global mask"):
        fa_file = os.path.join(SOURCE_DIR, item['folder'], f"{item['subj']}_{item['img_id']}_FA_reg.nii.gz")
        img = nib.load(fa_file)
        img_2mm = resample_img(img, target_affine=target_affine, target_shape=target_shape, interpolation='continuous')
        sum_data += img_2mm.get_fdata(dtype=np.float32)
        
    mean_data = sum_data / N
    # Adaptive threshold filtering, excluding ventricular water molecules
    brain_mask = official_mask & (mean_data > 0.05)

    num_voxels = brain_mask.sum()
    print(f"Total brain valid voxel count: {num_voxels}")
    del sum_data, mean_data

    # ==========================================
    # Process each modality in sequence (steps 3, 4, 5, 6)
    # ==========================================
    for MODALITY in MODALITIES:
        print("\n" + "="*50)
        print(f"Starting modality processing: {MODALITY}")
        print("="*50)

        # 3. MNI 2mm downsampling and memory loading
        print(f"Loading {MODALITY} and resampling images to MNI 2mm template...")
        loaded_imgs_2mm = []
        valid_records = []
        for item in tqdm(valid_subjects, desc=f"Resampling{MODALITY}"):
            file_name = f"{item['subj']}_{item['img_id']}_{MODALITY}_reg.nii.gz"
            file_path = os.path.join(SOURCE_DIR, item['folder'], file_name)
            valid_records.append({'folder': item['folder'], 'file': file_name})
            
            img = nib.load(file_path)
            img_2mm = resample_img(
                img, 
                target_affine=target_affine, 
                target_shape=target_shape, 
                interpolation='continuous'
            )
            loaded_imgs_2mm.append(img_2mm)

    
        # 4. Manually extract voxels (using the brain_mask directly)
        print(f"Extracting {MODALITY} 1D voxel matrix...")
        X_all_voxels = np.zeros((N, num_voxels), dtype=np.float32)
        for i, img in enumerate(tqdm(loaded_imgs_2mm, desc=f"Extracting {MODALITY} voxels")):
            X_all_voxels[i] = img.get_fdata(dtype=np.float32)[brain_mask]
            
        del loaded_imgs_2mm # Extraction complete, clear large 3D object cache

        # ==========================================
        # 5. Normative Modeling
        # ==========================================
        print(f"Fitting {MODALITY} healthy aging curve (CN group only)...")
        X_cn_voxels = X_all_voxels[is_cn_mask]
        confounds_cn = confounds[is_cn_mask]

        reg = LinearRegression()
        reg.fit(confounds_cn, X_cn_voxels)

        print(f"Computing {MODALITY} full pathological residuals and restoring physical units...")
        X_expected_all = reg.predict(confounds)
        X_residual_all = X_all_voxels - X_expected_all

        # Restore FA absolute units
        CN_mean_brain = X_cn_voxels.mean(axis=0)
        X_residual_all = X_residual_all + CN_mean_brain

        # 6. Restore NumPy to 3D NIfTI and save with multi-threading
        print(f"Restoring {MODALITY} 1D matrix to 3D structure and writing to disk...")
        Parallel(n_jobs=16, prefer="threads")(
            delayed(save_single_image)(
                X_residual_all[i], 
                valid_records[i], 
                brain_mask, 
                target_affine, 
                TARGET_DIR
            ) for i in tqdm(range(N), desc=f"Saving{MODALITY}")
        )
        
        # Clear current modality memory to prevent overflow when processing next modality
        del X_all_voxels, X_expected_all, X_residual_all, X_cn_voxels

    print("\n" + "="*50)
    print("All processing complete!")
    print(f"Pathological residual data saved to: {TARGET_DIR}")
    print("="*50)

if __name__ == "__main__":
    process_and_residualize()