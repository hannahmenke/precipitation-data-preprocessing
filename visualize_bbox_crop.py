import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

# --- User parameters ---
EXCEL_PATH = 'data_for_classification/3mM-5_finalData.xlsx'
H5_PATH = 'image/2/2025_0628/3mM-0628-5-mix/3mM-0628-5-mix_filtered_normalized_timeseries.h5'
TIMESTEP = 18  # Last index for 19 images
ROW_IDX = 200    # Use the third row (Python index 2)
OUTPUT_PATH = 'crop_topleft_with_full_zoomed.png'
OFFSET = 1000      # Offset for x and y
BORDER = 2         # Border around bounding box

# --- Load bounding box from Excel ---
df = pd.read_excel(EXCEL_PATH)
bbox = df.iloc[ROW_IDX]['BoundingBox']
if isinstance(bbox, str):
    # Convert string to list if needed
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

# --- Full image: exclude 1000-pixel edge ---
full_x0 = OFFSET
full_y0 = OFFSET
full_img = img[full_y0:img_h - OFFSET, full_x0:img_w - OFFSET]

# --- Adjust rectangle coordinates for full_img ---
rect_x = crop_x - full_x0
rect_y = crop_y - full_y0

# --- Plot full image with bounding box and the crop ---
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Full image with rectangle
axes[0].imshow(full_img, cmap='gray')
rect = Rectangle((rect_x, rect_y), crop_w, crop_h, linewidth=2, edgecolor='red', facecolor='none', label='Bounding Box + Border')
axes[0].add_patch(rect)
axes[0].set_title('Full Image (center region, last timestep)')
axes[0].legend(loc='upper right')
axes[0].axis('off')

# Cropped image
axes[1].imshow(crop, cmap='gray')
axes[1].set_title('Cropped Region (Top-Left Origin, Zoomed Out)')
axes[1].axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_PATH, bbox_inches='tight', dpi=200)
plt.show()
print(f'Figure saved to {OUTPUT_PATH}') 