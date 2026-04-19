import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_all_markers(binary_mask, ignore_x_min=None, ignore_x_max=None, min_area=20,
                     morph_open=2, morph_close=10, debug=False):
    """
    Find all connected components in binary_mask,
    excluding any whose centroid falls within the ignore x range,
    and excluding tiny components below min_area.

    morph_open  : kernel radius for opening (removes speckles). 0 to skip.
    morph_close : kernel radius for closing (fills gaps).        0 to skip.
    debug       : if True, shows before/after morph plots.

    Returns list of component dicts sorted by cx ascending.
    """
    mask = binary_mask.astype(np.uint8)

    # apply morphological operations to clean up the mask before connected components
    if morph_open > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*morph_open+1, 2*morph_open+1))
        after_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    else:
        after_open = mask.copy()

    if morph_close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*morph_close+1, 2*morph_close+1))
        after_close = cv2.morphologyEx(after_open, cv2.MORPH_CLOSE, k)
    else:
        after_close = after_open.copy()

    if debug:
        _, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(mask,        cmap='gray'); axes[0].set_title('Original')
        axes[1].imshow(after_open,  cmap='gray'); axes[1].set_title(f'After open (r={morph_open})')
        axes[2].imshow(after_close, cmap='gray'); axes[2].set_title(f'After close (r={morph_close})')
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        after_close, connectivity=8
    )

    components = []
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        cx, cy = centroids[label]

        if ignore_x_min is not None and ignore_x_max is not None:
            if ignore_x_min <= cx <= ignore_x_max:
                continue

        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]

        components.append({
            'area': area,
            'cx': cx, 'cy': cy,
            'x': x, 'y': y,
            'w': w, 'h': h
        })

    components.sort(key=lambda c: c['cx'])
    return components

def cluster_markers_into_columns(components, n_columns, markers_per_column=4):
    """
    Given components sorted by cx, cluster them into n_columns groups
    by taking markers_per_column consecutive components left to right.

    Returns list of lists, each inner list being the components
    belonging to one column, ordered left to right.
    """
    expected = n_columns * markers_per_column
    if len(components) != expected:
        print(f"Warning: expected {expected} components, "
              f"got {len(components)}. Proceeding anyway.")

    if len(components) == 0:
        return []

    columns = [
        components[i * markers_per_column : (i + 1) * markers_per_column]
        for i in range(n_columns)
    ]

    return columns

def assign_corners_from_bbox(components):
    """
    Given 4 component dicts (each with cx, cy, x, y, w, h),
    assign corner roles based on centroid position,
    then return the specific bounding box corner point
    appropriate for each role.
    
    Corner point logic:
      tl marker -> (x,       y      )  top-left of bbox
      tr marker -> (x+w,     y      )  top-right of bbox
      bl marker -> (x,       y+h    )  bottom-left of bbox
      br marker -> (x+w,     y+h    )  bottom-right of bbox
    
    Returns dict with keys 'tl','tr','bl','br'
    each containing (px, py) float tuple.
    """
    if len(components) != 4:
        raise ValueError(f"Expected 4 components, got {len(components)}")
    
    # sort by y to get top vs bottom
    sorted_by_y = sorted(components, key=lambda c: c['cy'])
    top_two    = sorted(sorted_by_y[:2], key=lambda c: c['cx'])
    bottom_two = sorted(sorted_by_y[2:], key=lambda c: c['cx'])
    
    tl_comp = top_two[0]
    tr_comp = top_two[1]
    bl_comp = bottom_two[0]
    br_comp = bottom_two[1]
    
    return {
        'tl': (tl_comp['x'],              tl_comp['y']             ),
        'tr': (tr_comp['x'] + tr_comp['w'], tr_comp['y']           ),
        'bl': (bl_comp['x'],              bl_comp['y'] + bl_comp['h']),
        'br': (br_comp['x'] + br_comp['w'], br_comp['y'] + br_comp['h'])
    }

def get_column_corners(column_components):
    """
    Given the components belonging to one column (should be 4),
    assign corner roles and return the appropriate bbox corner points.
    
    Returns corner dict with keys tl, tr, bl, br.
    """
    if len(column_components) != 4:
        raise ValueError(
            f"Expected 4 components for column, got {len(column_components)}"
        )
    return assign_corners_from_bbox(column_components)

def get_all_column_corners(frame, lower_color, upper_color,
                            n_columns,
                            ignore_x_min=None, ignore_x_max=None,
                            morph_open=2, morph_close=10,
                            min_area=50,
                            components_per_column=4):
    """
    Full pipeline for one frame.
    Now works with any number of components per column.
    Returns list of corner dicts, one per column."""
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    components = find_all_markers(
        mask,
        ignore_x_min=ignore_x_min,
        ignore_x_max=ignore_x_max,
        morph_open=morph_open,
        morph_close=morph_close,
        min_area=min_area
    )
        
    column_groups = cluster_markers_into_columns(
        components, n_columns=n_columns, markers_per_column=components_per_column
    )
    
    all_corners = []
    for i, group in enumerate(column_groups):
        try:
            # corners is a dict with keys tl, tr, bl, br each (px, py)
            corners = get_column_corners(group)
            all_corners.append(corners)
        except Exception as e:
            print(f"Column {i}: corner extraction failed — {e}")
            all_corners.append(None)
    
    return all_corners

def compute_output_size(all_corner_dicts):
    """
    Given a list of corner dicts (one per column),
    compute mean width and height across all columns
    to use as the target warp output size.
    
    Returns (output_width, output_height) as ints.
    """
    widths = []
    heights = []
    
    for corners in all_corner_dicts:
        tl = np.array(corners['tl'])
        tr = np.array(corners['tr'])
        bl = np.array(corners['bl'])
        br = np.array(corners['br'])
        
        # width: mean of top edge and bottom edge lengths
        top_w    = np.linalg.norm(tr - tl)
        bottom_w = np.linalg.norm(br - bl)
        w = (top_w + bottom_w) / 2
        
        # height: mean of left edge and right edge lengths
        left_h  = np.linalg.norm(bl - tl)
        right_h = np.linalg.norm(br - tr)
        h = (left_h + right_h) / 2
        
        widths.append(w)
        heights.append(h)
    
    output_w = int(np.round(np.mean(widths)))
    output_h = int(np.round(np.mean(heights)))
    
    print(f"Individual widths:  {[f'{w:.1f}' for w in widths]}")
    print(f"Individual heights: {[f'{h:.1f}' for h in heights]}")
    print(f"Output size: {output_w} x {output_h}")
    
    return output_w, output_h

def compute_column_transform(corners, output_w, output_h):
    """
    Compute perspective transform matrix for one column.
    
    corners: dict with keys tl, tr, bl, br each (px, py)
    output_w, output_h: desired output dimensions
    
    Returns M (3x3 perspective transform matrix)
    """
    src = np.array([
        corners['tl'],
        corners['tr'],
        corners['br'],
        corners['bl']
    ], dtype=np.float32)
    
    dst = np.array([
        [0,           0          ],
        [output_w,    0          ],
        [output_w,    output_h   ],
        [0,           output_h   ]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(src, dst)
    return M

def warp_column(frame, corners, output_w, output_h):
    """
    Extract and rectify one column from frame using perspective transform.
    
    Returns:
        warped: rectified column image (output_h x output_w x 3)
        M: the transform matrix (useful to save for applying to all frames)
    """
    M = compute_column_transform(corners, output_w, output_h)
    warped = cv2.warpPerspective(frame, M, (output_w, output_h))
    return warped, M

def compute_all_column_transforms(all_corners, output_w, output_h):
    """
    Given all_corners (list of corner dicts, one per column),
    compute and store perspective transform matrices.
    
    Returns list of M matrices, one per column.
    """
    transforms = []
    for i, corners in enumerate(all_corners):
        if corners is None:
            print(f"Column {i}: no corners found, skipping")
            transforms.append(None)
            continue
        M = compute_column_transform(corners, output_w, output_h)
        transforms.append(M)
        #print(f"Column {i}: transform computed")
    return transforms

def extract_all_transformed_columns(frame, transforms, output_w, output_h):
    
    """
    Apply precomputed transforms to extract all columns from a frame.
    
    Returns list of warped column images, one per column.
    None for columns where transform is missing.
    """
    columns = []
    for i, M in enumerate(transforms):
        if M is None:
            columns.append(None)
            continue
        warped = cv2.warpPerspective(frame, M, (output_w, output_h))
        columns.append(warped)
    return columns