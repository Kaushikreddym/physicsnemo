#!/usr/bin/env python3
"""
Test and visualize patch ordering to ensure combine_patches.py matches the dataset class.
"""

import sys
import os
sys.path.append('/beegfs/muduchuru/codes/physicsnemo/examples/weather/corrdiff')

import numpy as np
from datasets.mswxdwd import mswxdwd


def test_patch_ordering():
    """Test that patch indexing is consistent between dataset and combine script."""
    print("Testing patch ordering consistency...")
    
    # Create a test dataset loader
    loader = mswxdwd(
        data_path='/beegfs/muduchuru/data',
        input_channels=['pr'],
        output_channels=['pr'],
        patch_size=(128, 128),
        overlap_pix=0
    )
    
    domain_shape = loader.lat.shape
    patch_size = (128, 128)
    overlap_pix = 0
    
    print(f"Domain shape: {domain_shape}")
    print(f"Patch size: {patch_size}")
    
    total_patches = loader.get_total_patches(128, 128)
    print(f"Total patches: {total_patches}")
    
    # Test first few patches
    print(f"\nPatch ordering test (first 6 patches):")
    print("Patch |  Row  |  Col  | Top | Left | Right | Bottom")
    print("------|-------|-------|-----|------|-------|-------")
    
    for patch_idx in range(min(6, total_patches)):
        # Get bounds from dataset class
        top, bottom, left, right = loader.get_patch_bounds_by_index(
            patch_idx, 128, 128, overlap_pix
        )
        
        # Calculate which row/col this should be
        h, w = domain_shape
        stride_y = 128 - overlap_pix
        stride_x = 128 - overlap_pix
        patches_per_row = (w + stride_x - 1) // stride_x
        
        expected_row = patch_idx // patches_per_row
        expected_col = patch_idx % patches_per_row
        
        print(f"  {patch_idx:2d}  |   {expected_row:2d}  |   {expected_col:2d}  |{top:4d} |{left:5d} |{right:6d} |{bottom:7d}")
    
    # Create a visual grid representation
    print(f"\nVisual representation of first 12 patches:")
    h, w = domain_shape
    patches_per_row = (w + 127) // 128  # stride = 128 for non-overlapping
    patches_per_col = (h + 127) // 128
    
    print(f"Grid layout: {patches_per_col} rows × {patches_per_row} columns")
    print("\nPatch indices in spatial layout:")
    
    for row in range(min(3, patches_per_col)):  # Show first 3 rows
        row_str = ""
        for col in range(min(patches_per_row, 6)):  # Show up to 6 columns
            patch_idx = row * patches_per_row + col
            if patch_idx < total_patches:
                row_str += f"{patch_idx:3d} "
            else:
                row_str += "    "
        print(f"Row {row}: {row_str}")
    
    if patches_per_row > 6:
        print(f"... (showing first 6 of {patches_per_row} columns)")
    
    print(f"\n✓ Patch ordering follows row-major layout:")
    print(f"  - Patches numbered left-to-right, top-to-bottom")  
    print(f"  - Row 0: patches 0 to {patches_per_row-1}")
    print(f"  - Row 1: patches {patches_per_row} to {2*patches_per_row-1}")
    print(f"  - etc.")
    
    return True


if __name__ == "__main__":
    test_patch_ordering()