#!/usr/bin/env python3
"""
Combine patch-wise NetCDF files into a spatially coherent full domain file.

This script takes individual patch files and reconstructs them into a complete
spatial domain based on their patch indices.

Optimized with parallel I/O using ThreadPoolExecutor.
"""

import argparse
import glob
import sys
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from scipy.ndimage import gaussian_filter

# Try to import the dataset class for perfect consistency
try:
    sys.path.append(os.path.dirname(__file__))
    from datasets.mswxdwd import mswxdwd
    DATASET_AVAILABLE = True
except ImportError:
    DATASET_AVAILABLE = False
    print("Warning: Could not import dataset class. Using fallback logic.")


def get_patch_bounds_from_dataset_class(patch_index, patch_size=(128, 128), 
                                       domain_shape=(867, 642), overlap_pix=0):
    """
    Get patch bounds using the actual dataset class for perfect consistency.
    
    This is the most reliable method as it uses the exact same code.
    """
    if not DATASET_AVAILABLE:
        return get_patch_spatial_bounds_from_dataset(patch_index, patch_size, domain_shape, overlap_pix)
    
    # Create a dummy dataset instance just to use its method
    try:
        # Create a minimal mock object that mimics the dataset
        class DummyLoader:
            def __init__(self, domain_shape):
                self.lat = np.zeros(domain_shape)
                self.lon = np.zeros(domain_shape)
            
            def get_patch_bounds_by_index(self, patch_index: int, ph: int, pw: int, 
                                          overlap_pix: int = 0):
                """Exact same logic as mswxdwd dataset."""
                h, w = self.lat.shape
                
                # Calculate grid dimensions
                stride_y = ph - overlap_pix
                stride_x = pw - overlap_pix
                
                patches_per_row = (w + stride_x - 1) // stride_x
                patches_per_col = (h + stride_y - 1) // stride_y
                
                # Convert linear index to 2D grid coordinates
                patch_row = patch_index // patches_per_row
                patch_col = patch_index % patches_per_row
                
                # Calculate patch bounds
                top = patch_row * stride_y
                left = patch_col * stride_x
                
                # Ensure patch doesn't exceed image bounds
                top = min(top, h - ph)
                left = min(left, w - pw)
                
                bottom = top + ph
                right = left + pw
                
                return top, bottom, left, right
        
        dummy_loader = DummyLoader(domain_shape)
        
        # Use the exact same method
        top, bottom, left, right = dummy_loader.get_patch_bounds_by_index(
            patch_index, patch_size[0], patch_size[1], overlap_pix
        )
        
        return top, bottom, left, right
        
    except Exception as e:
        print(f"Warning: Could not use dataset class method: {e}")
        return get_patch_spatial_bounds_from_dataset(patch_index, patch_size, domain_shape, overlap_pix)


def get_patch_spatial_bounds_from_dataset(patch_index, patch_size, domain_shape, overlap_pix=0):
    """
    Calculate spatial bounds using the EXACT same logic as the dataset class.
    
    This ensures consistency between patch generation and combination.
    
    Parameters
    ----------
    patch_index : int
        Linear index of the patch (0-based)
    patch_size : tuple
        (height, width) of each patch
    domain_shape : tuple
        (height, width) of full domain
    overlap_pix : int
        Number of overlapping pixels between patches
        
    Returns
    -------
    tuple
        (y_start, y_end, x_start, x_end) bounds in full domain
    """
    ph, pw = patch_size
    h, w = domain_shape
    
    # Calculate grid dimensions - SAME as dataset class
    stride_y = ph - overlap_pix
    stride_x = pw - overlap_pix
    
    patches_per_row = (w + stride_x - 1) // stride_x  # Ceiling division
    patches_per_col = (h + stride_y - 1) // stride_y
    
    # Convert linear index to 2D grid coordinates - SAME as dataset class
    patch_row = patch_index // patches_per_row
    patch_col = patch_index % patches_per_row
    
    # Calculate patch bounds - SAME as dataset class
    top = patch_row * stride_y
    left = patch_col * stride_x
    
    # Ensure patch doesn't exceed domain bounds - SAME as dataset class
    top = min(top, h - ph)
    left = min(left, w - pw)
    
    bottom = top + ph
    right = left + pw
    
    return top, bottom, left, right


def get_patch_spatial_bounds(patch_index, total_patches_x, total_patches_y, 
                           patch_size, domain_shape, overlap_pix=0):
    """
    DEPRECATED: Use get_patch_spatial_bounds_from_dataset for consistency.
    
    Calculate spatial bounds for a patch given its index.
    
    Parameters
    ----------
    patch_index : int
        Linear index of the patch (0-based)
    total_patches_x : int
        Number of patches in x direction
    total_patches_y : int  
        Number of patches in y direction
    patch_size : tuple
        (height, width) of each patch
    domain_shape : tuple
        (height, width) of full domain
    overlap_pix : int
        Number of overlapping pixels between patches
        
    Returns
    -------
    tuple
        (y_start, y_end, x_start, x_end) bounds in full domain
    """
    # Use the new consistent function
    return get_patch_spatial_bounds_from_dataset(patch_index, patch_size, domain_shape, overlap_pix)


def create_gaussian_blend_weights(patch_size, overlap_pix, sigma=None):
    """
    Create gaussian blend weights for a patch.
    
    The weights are 1.0 in the center and smoothly taper to 0 at the edges
    in the overlap regions using a gaussian falloff.
    
    Parameters
    ----------
    patch_size : tuple
        (height, width) of each patch
    overlap_pix : int
        Number of overlapping pixels between patches
    sigma : float, optional
        Sigma for gaussian falloff. If None, defaults to overlap_pix/3
        
    Returns
    -------
    np.ndarray
        2D array of blend weights with shape patch_size
    """
    if overlap_pix == 0:
        # No overlap, return uniform weights
        return np.ones(patch_size, dtype=np.float32)
    
    ph, pw = patch_size
    
    # Default sigma: make gaussian falloff complete within overlap region
    if sigma is None:
        sigma = overlap_pix / 3.0
    
    # Create 1D weight profiles that taper at edges
    def create_1d_weights(size, overlap):
        weights = np.ones(size, dtype=np.float32)
        if overlap > 0:
            # Create gaussian taper at edges
            taper = np.linspace(0, 1, overlap)
            # Apply gaussian-like falloff: exp(-((1-x)/sigma)^2)
            taper = np.exp(-((1 - taper) / (sigma / overlap))**2)
            
            # Apply taper to both edges
            weights[:overlap] *= taper
            weights[-overlap:] *= taper[::-1]
        return weights
    
    # Create 2D weights by outer product of 1D profiles
    weights_y = create_1d_weights(ph, overlap_pix)
    weights_x = create_1d_weights(pw, overlap_pix)
    weights_2d = np.outer(weights_y, weights_x)
    
    return weights_2d


def combine_patches_spatially(patch_files, output_file, domain_shape=(867, 642), 
                             patch_size=(128, 128), overlap_pix=0, n_workers=8):
    """
    Combine patch files into a spatially coherent full domain file using parallel processing.
    
    Parameters
    ----------
    patch_files : list
        List of patch NetCDF file paths
    output_file : str
        Output file path for combined NetCDF
    domain_shape : tuple
        (height, width) of full domain
    patch_size : tuple
        (height, width) of each patch
    overlap_pix : int
        Number of overlapping pixels between patches
    n_workers : int
        Number of parallel workers for processing patches
    """
    import netCDF4 as nc4
    
    if not patch_files:
        raise ValueError("No patch files provided")
        
    print(f"Combining {len(patch_files)} patch files with {n_workers} workers...")
    print(f"Domain shape: {domain_shape}")
    print(f"Patch size: {patch_size}")
    print(f"Overlap pixels: {overlap_pix}")
    
    # Calculate grid dimensions using SAME logic as dataset class
    ph, pw = patch_size
    h, w = domain_shape
    stride_y = ph - overlap_pix
    stride_x = pw - overlap_pix
    
    patches_per_row = (w + stride_x - 1) // stride_x
    patches_per_col = (h + stride_y - 1) // stride_y
    
    print(f"Grid: {patches_per_col} x {patches_per_row} patches")
    print(f"Total expected patches: {patches_per_col * patches_per_row}")
    
    # Find first complete patch file with groups (skip incomplete ones)
    first_complete_file = None
    for pf in patch_files:
        try:
            test_nc = nc4.Dataset(pf, 'r')
            groups = list(test_nc.groups.keys())
            test_nc.close()
            if len(groups) > 0:  # Found a complete file with groups
                first_complete_file = pf
                print(f"Using template from: {Path(pf).name}")
                break
        except:
            continue
    
    if first_complete_file is None:
        print("Warning: No files with groups found, using first file")
        first_complete_file = patch_files[0]
    
    # Open first complete patch to inspect structure
    first_nc = nc4.Dataset(first_complete_file, 'r')
    
    # Check if file has groups
    groups = list(first_nc.groups.keys())
    has_groups = len(groups) > 0
    
    if has_groups:
        print(f"  Detected {len(groups)} groups: {groups}")
    
    # Create output file with group structure preserved
    out_nc = nc4.Dataset(output_file, 'w', format='NETCDF4')
    
    # Copy global attributes from first file
    out_nc.setncatts({k: first_nc.getncattr(k) for k in first_nc.ncattrs()})
    
    def create_dimensions(src_grp, dst_grp, inherit_from_root=False):
        """Create dimensions in a group, optionally inheriting from root."""
        for dim_name, dim in src_grp.dimensions.items():
            # Skip if already exists (inherited)
            if dim_name in dst_grp.dimensions:
                continue
                
            if dim_name in ['y', 'lat']:
                dst_grp.createDimension(dim_name, h)
            elif dim_name in ['x', 'lon']:
                dst_grp.createDimension(dim_name, w)
            elif dim.isunlimited():
                # For unlimited dimensions, create with current size from source
                # This preserves dimensions like ensemble that should not be empty
                current_size = len(dim) if len(dim) > 0 else None
                dst_grp.createDimension(dim_name, current_size)
            else:
                dst_grp.createDimension(dim_name, len(dim))
    
    def initialize_group(src_grp, dst_grp, root_dims=None):
        """Recursively initialize group structure and variables."""
        # Copy group attributes
        dst_grp.setncatts({k: src_grp.getncattr(k) for k in src_grp.ncattrs()})
        
        # Create dimensions for this group
        create_dimensions(src_grp, dst_grp)
        
        # Create variables
        for var_name, var in src_grp.variables.items():
            # Create variable with same dtype and dimensions
            out_var = dst_grp.createVariable(
                var_name, 
                var.datatype, 
                var.dimensions,
                zlib=True,
                complevel=1
            )
            
            # Copy variable attributes
            out_var.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
            
            # Initialize with NaN for float types
            if np.issubdtype(var.datatype, np.floating):
                out_var[:] = np.nan
        
        # Recursively process subgroups
        for grp_name in src_grp.groups.keys():
            sub_grp = dst_grp.createGroup(grp_name)
            initialize_group(src_grp.groups[grp_name], sub_grp, root_dims)
    
    # First, create root-level dimensions (these may be needed by group variables)
    create_dimensions(first_nc, out_nc)
    
    # Create root-level variables
    for var_name, var in first_nc.variables.items():
        out_var = out_nc.createVariable(
            var_name, 
            var.datatype, 
            var.dimensions,
            zlib=True,
            complevel=1
        )
        out_var.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
        if np.issubdtype(var.datatype, np.floating):
            out_var[:] = np.nan
    
    # Initialize group structure
    if has_groups:
        # Process each group
        for grp_name in groups:
            sub_grp = out_nc.createGroup(grp_name)
            initialize_group(first_nc.groups[grp_name], sub_grp, first_nc.dimensions)
    
    first_nc.close()
    
    # Collect patch metadata in parallel
    print("Collecting patch metadata in parallel...")
    
    def process_single_patch(patch_file):
        """Process a single patch file and return its metadata."""
        try:
            filename = Path(patch_file).name
            patch_idx_str = filename.split('_patch_')[1].split('_')[0]
            patch_index = int(patch_idx_str)
            
            # Get spatial bounds
            y_start, y_end, x_start, x_end = get_patch_bounds_from_dataset_class(
                patch_index, patch_size, domain_shape, overlap_pix
            )
            
            return {
                'file': patch_file,
                'index': patch_index,
                'bounds': (y_start, y_end, x_start, x_end),
                'success': True
            }
        except Exception as e:
            print(f"    ERROR processing metadata for {patch_file}: {e}")
            return {'file': patch_file, 'success': False, 'error': str(e)}
    
    patch_metadata = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_single_patch, pf) for pf in patch_files]
        for future in as_completed(futures):
            result = future.result()
            if result['success']:
                patch_metadata.append(result)
    
    print(f"Successfully collected metadata for {len(patch_metadata)} patches")
    patch_metadata.sort(key=lambda x: x['index'])
    
    # Process each patch file with progress tracking
    print("Processing patches and placing data into full domain...")
    processed_patches = 0
    
    # Use ThreadPoolExecutor for parallel patch processing
    def copy_patch_data(metadata):
        """Copy data from a single patch to the output file."""
        try:
            patch_file = metadata['file']
            patch_index = metadata['index']
            y_start, y_end, x_start, x_end = metadata['bounds']
            
            filename = Path(patch_file).name
            print(f"  Processing patch {patch_index}: {filename}")
            print(f"    Placing at: y[{y_start}:{y_end}], x[{x_start}:{x_end}]")
            
            # Open patch file
            patch_nc = nc4.Dataset(patch_file, 'r')
            
            # Return data to be written (since we can't write in parallel to netCDF4)
            patch_data = {'index': patch_index, 'bounds': (y_start, y_end, x_start, x_end)}
            
            def collect_variables(src_grp, path=''):
                """Recursively collect variable data from patch."""
                var_data = {}
                for var_name, var in src_grp.variables.items():
                    full_path = f"{path}/{var_name}" if path else var_name
                    var_data[full_path] = {
                        'data': var[:],
                        'dims': var.dimensions
                    }
                
                # Recursively process subgroups
                for grp_name in src_grp.groups.keys():
                    group_path = f"{path}/{grp_name}" if path else grp_name
                    var_data.update(collect_variables(src_grp.groups[grp_name], group_path))
                
                return var_data
            
            patch_data['variables'] = collect_variables(patch_nc)
            patch_nc.close()
            
            return {'success': True, 'data': patch_data}
            
        except Exception as e:
            print(f"    ERROR processing {patch_file}: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    # Read patches in parallel
    print("Reading patch data in parallel...")
    all_patch_data = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(copy_patch_data, metadata) for metadata in patch_metadata]
        for future in as_completed(futures):
            result = future.result()
            if result['success']:
                all_patch_data.append(result['data'])
    
    print(f"Successfully read {len(all_patch_data)} patches")
    all_patch_data.sort(key=lambda x: x['index'])
    
    # Create blend weights for patches
    blend_weights = create_gaussian_blend_weights(patch_size, overlap_pix)
    if overlap_pix > 0:
        print(f"Using gaussian blending with overlap={overlap_pix} pixels")
    
    # Create weight accumulation arrays for proper blending
    # We'll track the sum of weights for each pixel
    weight_accumulators = {}
    
    # Write data to output file with gaussian blending
    print("Writing data to output file with gaussian blending...")
    for patch_data in all_patch_data:
        patch_index = patch_data['index']
        y_start, y_end, x_start, x_end = patch_data['bounds']
        
        print(f"  Writing patch {patch_index}...")
        
        for var_path, var_info in patch_data['variables'].items():
            # Navigate to the correct group/variable
            if '/' in var_path:
                parts = var_path.split('/')
                var_name = parts[-1]
                group_path = '/'.join(parts[:-1])
                
                # Navigate to group
                current_grp = out_nc
                for grp_name in group_path.split('/'):
                    if grp_name:
                        if grp_name not in current_grp.groups:
                            print(f"    Warning: Group '{grp_name}' not found in output file, skipping variable {var_path}")
                            current_grp = None
                            break
                        current_grp = current_grp.groups[grp_name]
                
                if current_grp is None:
                    continue
            else:
                var_name = var_path
                current_grp = out_nc
            
            if var_name in current_grp.variables:
                out_var = current_grp.variables[var_name]
                data = var_info['data']
                dims = var_info['dims']
                
                # Build slicing for output
                slices = []
                spatial_dims = []
                for i, dim_name in enumerate(dims):
                    if dim_name in ['y', 'lat']:
                        slices.append(slice(y_start, y_end))
                        spatial_dims.append(i)
                    elif dim_name in ['x', 'lon']:
                        slices.append(slice(x_start, x_end))
                        spatial_dims.append(i)
                    else:
                        slices.append(slice(None))
                
                # Apply gaussian blending for overlapping patches
                if overlap_pix > 0 and len(spatial_dims) >= 2:
                    # Create weight key for this variable
                    weight_key = var_path
                    
                    # Initialize weight accumulator if needed
                    if weight_key not in weight_accumulators:
                        weight_accumulators[weight_key] = np.zeros_like(out_var[:], dtype=np.float32)
                    
                    # Broadcast blend weights to match data shape
                    # Blend weights are (ph, pw), need to match data shape
                    weight_shape = list(data.shape)
                    weights_broadcast = blend_weights.copy()
                    
                    # Add dimensions before and after spatial dims
                    for i in range(len(dims)):
                        if i not in spatial_dims:
                            weights_broadcast = np.expand_dims(weights_broadcast, 
                                                              axis=0 if i < spatial_dims[0] else -1)
                    
                    # Broadcast to full shape
                    weights_broadcast = np.broadcast_to(weights_broadcast, data.shape)
                    
                    # Read current values
                    current_data = out_var[tuple(slices)]
                    current_weights = weight_accumulators[weight_key][tuple(slices)]
                    
                    # Apply weighted blending
                    # Replace NaN in current data with 0 for blending
                    current_data = np.where(np.isnan(current_data), 0, current_data)
                    
                    # Accumulate weighted data
                    weighted_data = data * weights_broadcast
                    blended_data = current_data * current_weights + weighted_data
                    
                    # Update weights
                    new_weights = current_weights + weights_broadcast
                    
                    # Normalize by accumulated weights (avoid division by zero)
                    final_data = np.where(new_weights > 0, blended_data / new_weights, np.nan)
                    
                    # Write blended data
                    out_var[tuple(slices)] = final_data
                    weight_accumulators[weight_key][tuple(slices)] = new_weights
                else:
                    # No overlap or no spatial dimensions, simple write
                    out_var[tuple(slices)] = data
        
        processed_patches += 1
    
    print(f"Successfully processed {processed_patches}/{len(patch_files)} patches")
    
    # Close output file
    out_nc.close()
    
    print("✓ Spatial combination completed successfully")
    print(f"  Output: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Combine patch NetCDF files spatially")
    parser.add_argument("--input-pattern", required=True, 
                       help="Glob pattern for input patch files")
    parser.add_argument("--output-file", required=True,
                       help="Output combined NetCDF file")
    parser.add_argument("--domain-height", type=int, default=867,
                       help="Height of full domain")
    parser.add_argument("--domain-width", type=int, default=642,
                       help="Width of full domain") 
    parser.add_argument("--patch-height", type=int, default=128,
                       help="Height of each patch")
    parser.add_argument("--patch-width", type=int, default=128,
                       help="Width of each patch")
    parser.add_argument("--overlap-pix", type=int, default=0,
                       help="Number of overlapping pixels")
    parser.add_argument("--n-workers", type=int, default=8,
                       help="Number of parallel workers for reading patches")
    
    args = parser.parse_args()
    
    # Find patch files and sort by patch index numerically
    patch_files_unsorted = glob.glob(args.input_pattern)
    
    if not patch_files_unsorted:
        print(f"ERROR: No files found matching pattern: {args.input_pattern}")
        sys.exit(1)
    
    # Sort by patch index (extracted from filename) numerically
    def extract_patch_index(filepath):
        """Extract patch index from filename like 'patch_10_' -> 10"""
        try:
            filename = Path(filepath).name
            patch_idx_str = filename.split('_patch_')[1].split('_')[0]
            return int(patch_idx_str)
        except:
            return 0
    
    patch_files = sorted(patch_files_unsorted, key=extract_patch_index)
    
    print(f"Found {len(patch_files)} patch files")
    
    # Create output directory if needed
    output_dir = Path(args.output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine patches
    combine_patches_spatially(
        patch_files=patch_files,
        output_file=args.output_file,
        domain_shape=(args.domain_height, args.domain_width),
        patch_size=(args.patch_height, args.patch_width),
        overlap_pix=args.overlap_pix,
        n_workers=args.n_workers
    )


if __name__ == "__main__":
    main()