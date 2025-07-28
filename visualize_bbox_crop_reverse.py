import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

# --- User parameters ---
EXCEL_PATH = 'data_for_classification/3mM-5_finalData.xlsx'
H5_PATH = 'image/2/2025_0628/3mM-0628-5-mix/3mM-0628-5-mix_filtered_normalized_timeseries.h5'
ROW_IDX = 200    # Use the third row (Python index 2)
OUTPUT_PATH = 'crop_reverse_timesteps.png'
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

# --- Load images from HDF5 ---
with h5py.File(H5_PATH, 'r') as f:
    images = f['images'][:]
    img_h, img_w = images[0].shape
    total_timesteps = len(images)

# Get last timestep
last_timestep = total_timesteps - 1

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

# --- Full image parameters: exclude 1000-pixel edge ---
full_x0 = OFFSET
full_y0 = OFFSET

# --- Adjust rectangle coordinates for full_img ---
rect_x = crop_x - full_x0
rect_y = crop_y - full_y0

# --- Create subplot grid for reverse timesteps only ---
num_cols = min(total_timesteps, 6)  # Limit columns to avoid too wide plot
num_rows = (total_timesteps + num_cols - 1) // num_cols

fig, axes = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 3 * num_rows))
if num_rows == 1:
    axes = axes.reshape(1, -1)

# Flatten axes for easier indexing
axes_flat = axes.flatten()

# Plot crops in reverse timestep order
for i, timestep in enumerate(range(last_timestep, -1, -1)):
    if i >= len(axes_flat):
        break
        
    img = images[timestep]
    crop = safe_crop(img, crop_x, crop_y, crop_w, crop_h)
    
    axes_flat[i].imshow(crop, cmap='gray')
    axes_flat[i].set_title(f'Crop (t={timestep})')
    axes_flat[i].axis('off')

# Hide unused subplots
for i in range(total_timesteps, len(axes_flat)):
    axes_flat[i].axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_PATH, bbox_inches='tight', dpi=200)
plt.show()
print(f'Figure saved to {OUTPUT_PATH}')