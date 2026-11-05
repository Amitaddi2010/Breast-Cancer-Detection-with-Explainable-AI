"""
postprocess.py — Morphological and Connected Component Post-processing for tumor segmentation.

On ultrasound images, segmentation models often produce small scattered false positive
blobs. Removing these via connected component analysis significantly improves Precision and Dice.
"""

import numpy as np
from scipy.ndimage import label, binary_fill_holes
from skimage.morphology import disk, dilation, erosion


def postprocess_mask(pred_mask: np.ndarray, min_size: int = 150) -> np.ndarray:
    """
    Refine predicted binary mask:
      1. Connected Component Filtering (remove small scattered false positive blobs)
      2. Binary Hole Filling (fill internal holes in the tumor mask)
      3. Morphological Dilation/Erosion to smooth edges.

    Args:
        pred_mask: 2D binary numpy array (0/1) or boolean array.
        min_size: Minimum number of pixels a component must have to be kept.

    Returns:
        Refined binary mask (0/1).
    """
    # Ensure binary format
    mask = (pred_mask > 0.5).astype(np.uint8)

    # 1. Connected Component Filtering
    labeled_array, num_features = label(mask)
    if num_features > 0:
        # Count sizes of each component
        sizes = np.bincount(labeled_array.ravel())
        
        # Identify small components
        mask_sizes = sizes < min_size
        remove_pixel = mask_sizes[labeled_array]
        
        # Remove them
        mask[remove_pixel] = 0

    # 2. Binary Hole Filling
    mask = binary_fill_holes(mask).astype(np.uint8)

    # 3. Morphological refinement (closing: dilation followed by erosion)
    # Smooths boundary contours and closes small gaps
    try:
        selem = disk(3)
        mask = dilation(mask, selem)
        mask = erosion(mask, selem).astype(np.uint8)
    except Exception as e:
        # Fallback if skimage morphology fails
        pass

    return mask


if __name__ == "__main__":
    # Smoke test postprocessing
    dummy_mask = np.zeros((100, 100), dtype=np.uint8)
    # Large tumor blob
    dummy_mask[20:60, 20:60] = 1
    # Small noise blob
    dummy_mask[80:85, 80:85] = 1
    
    # Run postprocess
    cleaned = postprocess_mask(dummy_mask, min_size=50)
    
    print("Post-processing test:")
    print(f"  Original positive pixels: {dummy_mask.sum()}")
    print(f"  Cleaned positive pixels: {cleaned.sum()} (Should remove small noise blob)")
    assert cleaned[82, 82] == 0, "Noise blob not removed!"
    assert cleaned[40, 40] == 1, "Main tumor blob corrupted!"
    print("  [OK] Smoke test passed.")
