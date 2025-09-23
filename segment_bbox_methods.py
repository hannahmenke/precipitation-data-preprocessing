import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.filters import threshold_otsu
from skimage import morphology, measure
from sklearn.cluster import KMeans
import cv2
from skimage.segmentation import watershed
from skimage.util import img_as_float
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_holes
from skimage.measure import label, regionprops
from matplotlib.colors import ListedColormap

# --- User parameters ---
EXCEL_PATH = 'data_for_classification/3mM-5_finalData.xlsx'
H5_PATH = 'image/2/2025_0628/3mM-0628-5-mix/3mM-0628-5-mix_filtered_normalized_timeseries.h5'
TIMESTEP = 18  # Last index for 19 images
ROW_IDX = 2845    # Use the third row (Python index 2)
OFFSET = 1000      # Offset for x and y
BORDER = 0         # Border around bounding box

# --- Load bounding box from Excel ---
df = pd.read_excel(EXCEL_PATH)
bbox = df.iloc[ROW_IDX]['BoundingBox']
if isinstance(bbox, str):
    import ast
    bbox = ast.literal_eval(bbox)
x, y, w, h = bbox
w, h = int(round(w)), int(round(h))

# --- Load image from HDF5 ---
with h5py.File(H5_PATH, 'r') as f:
    img = f['images'][TIMESTEP]
    img_h, img_w = img.shape

# --- Convert x, y from 1-based (MATLAB) to 0-based (Python), then add offset ---
x0 = int(round(x)) - 1 + OFFSET
y0 = int(round(y)) - 1 + OFFSET

# --- Add border for zoomed out crop ---
crop_x = max(x0 - BORDER, 0)
crop_y = max(y0 - BORDER, 0)
crop_w = min(w + 2 * BORDER, img_w - crop_x)
crop_h = min(h + 2 * BORDER, img_h - crop_y)

def safe_crop(img, x, y, w, h):
    x = max(0, min(x, img.shape[1] - w))
    y = max(0, min(y, img.shape[0] - h))
    return img[y:y+h, x:x+w]

crop = safe_crop(img, crop_x, crop_y, crop_w, crop_h)

# --- Check for NaN or Inf in crop ---
nan_count = np.isnan(crop).sum()
inf_count = np.isinf(crop).sum()
print(f"NaNs in crop: {nan_count}, Infs in crop: {inf_count}")
if nan_count > 0 or inf_count > 0:
    median_val = np.nanmedian(crop[np.isfinite(crop)])
    crop = np.where(np.isfinite(crop), crop, median_val)

# --- Border-based background detection for k-means phases ---
border_width = 10
border_mask = np.zeros_like(crop, dtype=bool)
border_mask[:border_width, :] = True
border_mask[-border_width:, :] = True
border_mask[:, :border_width] = True
border_mask[:, -border_width:] = True
border_pixels = crop[border_mask]
border_mean = border_pixels.mean()
border_std = border_pixels.std()

# 3-phase k-means
pixels = crop.reshape(-1, 1).astype(np.float32)
kmeans3 = KMeans(n_clusters=3, n_init=10, random_state=0)
labels_kmeans3 = kmeans3.fit_predict(pixels)
mask_kmeans3 = labels_kmeans3.reshape(crop.shape)
means = [crop[mask_kmeans3 == i].mean() for i in range(3)]
sorted_idx = np.argsort(means)
phase_mask = np.zeros_like(mask_kmeans3)
for new_label, old_label in enumerate(sorted_idx):
    phase_mask[mask_kmeans3 == old_label] = new_label
phase_means = [crop[phase_mask == i].mean() for i in range(3)]
background_phase = np.argmin([abs(m - border_mean) for m in phase_means])
object_mask_border = phase_mask != background_phase
object_mask_border_filled = remove_small_holes(object_mask_border, area_threshold=64)
# --- Find largest object in the filled border-based mask (face connectivity only) ---
labeled_border = label(object_mask_border_filled, connectivity=1)
regions_border = regionprops(labeled_border)
if regions_border:
    largest_region_border = max(regions_border, key=lambda r: r.area)
    largest_mask_border_filled = (labeled_border == largest_region_border.label)
else:
    largest_mask_border_filled = np.zeros_like(object_mask_border_filled)

# --- Plot only the background-based steps ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(crop, cmap='gray')
axes[0].set_title('Original Crop')
axes[0].axis('off')
axes[1].imshow(object_mask_border_filled, cmap='gray')
axes[1].set_title('Not Background (Border-based, Filled)')
axes[1].axis('off')
axes[2].imshow(largest_mask_border_filled, cmap='gray')
axes[2].set_title('Largest Object (Border-based, Filled)')
axes[2].axis('off')
plt.tight_layout()
plt.savefig('segmentation_kmeans_background_based.png', dpi=200)
plt.show()

print('Background-based k-means segmentation steps saved to segmentation_kmeans_background_based.png') 