#!/usr/bin/env python
"""Quick test script to check dataloader output shapes."""

import sys
import numpy as np

# Test _create_lowres_ function
def test_create_lowres():
    """Test that _create_lowres_ preserves shape."""
    import cv2
    
    def _create_lowres_(x, factor=4):
        """Create low-resolution version by downsampling and upsampling, matching input shape exactly."""
        c, h, w = x.shape
        x = x.transpose(1, 2, 0)  # CHW → HWC
        x = x[::factor, ::factor, :]
        # Upsample back to original size, ensuring exact shape match
        x = cv2.resize(x, (w, h), interpolation=cv2.INTER_CUBIC)
        x = x.transpose(2, 0, 1)  # HWC → CHW
        return x
    
    # Test with 128×128
    test_input = np.random.randn(8, 128, 128).astype(np.float32)
    output = _create_lowres_(test_input, factor=4)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Shapes match: {test_input.shape == output.shape}")
    
    if test_input.shape != output.shape:
        print(f"ERROR: Shape mismatch! Expected {test_input.shape} but got {output.shape}")
        return False
    return True

if __name__ == "__main__":
    success = test_create_lowres()
    sys.exit(0 if success else 1)
