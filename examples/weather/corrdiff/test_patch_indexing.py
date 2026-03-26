#!/usr/bin/env python3
"""
Test the patch indexing functionality to ensure it works correctly
before running the full generation pipeline.
"""

import sys
import os
sys.path.append('/beegfs/muduchuru/codes/physicsnemo/examples/weather/corrdiff')

from datasets.mswxdwd import mswxdwd

def test_patch_indexing():
    """Test patch indexing functionality."""
    print("Testing patch indexing functionality...")
    
    # Create dataset loader
    try:
        loader = mswxdwd(
            data_path='/beegfs/muduchuru/data',
            input_channels=['pr', 'tas'],
            output_channels=['pr', 'tas'], 
            static_channels=['elevation', 'lsm', 'dwd_mask', 'pos_embed'],
            stats_dwd='/beegfs/muduchuru/data/hyras_daily/hyras_stats.json',
            stats_mswx='/beegfs/muduchuru/data/mswx/mswx_stats.json',
            patch_size=(128, 128),
            patch_index=0,  # Test with first patch
            overlap_pix=0,
            train=True
        )
        print("✓ Dataset loader created successfully")
    except Exception as e:
        print(f"✗ Failed to create dataset loader: {e}")
        return False
    
    # Get total patches
    try:
        total_patches = loader.get_total_patches(128, 128)
        print(f"✓ Total patches: {total_patches}")
    except Exception as e:
        print(f"✗ Failed to get total patches: {e}")
        return False
    
    # Test loading a few patches
    test_patches = min(3, total_patches)
    print(f"\nTesting first {test_patches} patches:")
    
    for patch_idx in range(test_patches):
        try:
            # Create loader for this patch
            patch_loader = mswxdwd(
                data_path='/beegfs/muduchuru/data',
                input_channels=['pr', 'tas'],
                output_channels=['pr', 'tas'],
                static_channels=['elevation', 'lsm', 'dwd_mask', 'pos_embed'],
                stats_dwd='/beegfs/muduchuru/data/hyras_daily/hyras_stats.json',
                stats_mswx='/beegfs/muduchuru/data/mswx/mswx_stats.json',
                patch_size=(128, 128),
                patch_index=patch_idx,
                overlap_pix=0,
                train=True
            )
            
            # Get patch data
            data = patch_loader[0]  # First time sample
            
            # Get patch metadata
            center = patch_loader.get_patch_center_latlon(patch_idx, 128, 128)
            bounds = patch_loader.get_patch_bounds_by_index(patch_idx, 128, 128)
            
            print(f"  Patch {patch_idx}:")
            print(f"    Data shapes: {data[0].shape}, {data[1].shape}")
            print(f"    Center: {center}")
            print(f"    Bounds: {bounds}")
            print(f"    ✓ Success")
            
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            return False
    
    print(f"\n✓ All tests passed! Patch indexing is working correctly.")
    print(f"Ready to run full generation with {total_patches} patches.")
    return True

if __name__ == "__main__":
    success = test_patch_indexing()
    sys.exit(0 if success else 1)