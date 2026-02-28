"""
Convert level1-5 PNG patterns to binary masks for real-time sewing detection
"""

import cv2
import numpy as np
import os

def convert_to_binary_mask(input_path, output_path, threshold=127):
    """
    Convert a PNG image to a binary mask.
    
    Args:
        input_path: Path to the input PNG file
        output_path: Path to save the binary mask
        threshold: Threshold value for binarization (0-255)
    """
    # Read the image
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        print(f"Error: Could not read {input_path}")
        return False
    
    print(f"Processing {input_path}...")
    print(f"  Original shape: {img.shape}")
    
    # Handle different image formats
    if len(img.shape) == 3:
        if img.shape[2] == 4:  # RGBA
            # For RGBA images, use the alpha channel to detect transparent areas
            # and RGB content to detect the actual pattern lines
            alpha = img[:, :, 3]
            bgr = img[:, :, :3]
            
            # Convert RGB to grayscale
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            
            # Create binary mask:
            # Pattern lines (pixels with significant alpha) = white (255)
            # Background (transparent areas) = black (0)
            
            # Use alpha channel as the primary mask (non-transparent areas)
            # Also consider content - keep darker pixels
            alpha_mask = alpha > 10  # Anything not fully transparent
            content_mask = gray < 250  # Not pure white
            
            # Combine both conditions
            binary_mask = np.zeros_like(gray)
            binary_mask[alpha_mask & content_mask] = 255
            
        else:  # RGB/BGR
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    else:  # Already grayscale
        gray = img
        _, binary_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Optional: Clean up the mask with morphological operations
    # Remove small noise
    kernel = np.ones((2, 2), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Fill small gaps in lines
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Save the binary mask
    cv2.imwrite(output_path, binary_mask)
    print(f"  Saved binary mask to {output_path}")
    print(f"  Mask shape: {binary_mask.shape}")
    print(f"  Non-zero pixels: {np.count_nonzero(binary_mask)}/{binary_mask.size} ({100*np.count_nonzero(binary_mask)/binary_mask.size:.2f}%)")
    
    return True

def main():
    blueprint_folder = 'blueprint'
    
    # Check if blueprint folder exists
    if not os.path.exists(blueprint_folder):
        print(f"Error: {blueprint_folder} folder not found!")
        return
    
    print("Converting level1-5 PNG files to binary masks...")
    print("=" * 60)
    
    # Convert each level
    for level in range(1, 6):
        input_file = os.path.join(blueprint_folder, f'level{level}.png')
        output_file = os.path.join(blueprint_folder, f'level{level}_mask.png')
        
        if not os.path.exists(input_file):
            print(f"Warning: {input_file} not found, skipping...")
            continue
        
        success = convert_to_binary_mask(input_file, output_file, threshold=127)
        
        if success:
            print(f"✓ Level {level} converted successfully")
        else:
            print(f"✗ Level {level} conversion failed")
        
        print("-" * 60)
    
    print("\nConversion complete!")
    print(f"Binary masks saved in {blueprint_folder}/ folder as level*_mask.png")

if __name__ == "__main__":
    main()
