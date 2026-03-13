from PIL import Image
import numpy as np

# Load level3.png
src = 'blueprint/image.png'
out = 'blueprint/level2_mask.png'
img = Image.open(src).convert('L')
arr = np.array(img)

# Threshold: everything not black becomes white (pattern), black stays black (background)
mask = np.where(arr > 10, 255, 0).astype(np.uint8)
mask_img = Image.fromarray(mask, mode='L')
mask_img.save(out)
print(f"Saved mask to {out}")
