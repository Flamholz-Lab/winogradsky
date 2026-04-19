import numpy as np
from transform import extract_all_transformed_columns, get_all_column_corners, compute_all_column_transforms
import cv2

def compute_layout(n_columns, output_w, output_h, gap=20, padding=20):
    """
    Compute the layout of columns in the composite output image.
    
    Returns:
        canvas_w: total width of output image
        canvas_h: total height of output image
        x_starts: list of x offsets for each column in the output image
    """
    canvas_w = padding * 2 + n_columns * output_w + (n_columns - 1) * gap
    canvas_h = padding * 2 + output_h
    
    x_starts = [
        padding + i * (output_w + gap)
        for i in range(n_columns)
    ]
    
    print(f"Canvas size: {canvas_w} x {canvas_h}")
    print(f"Column x offsets: {x_starts}")
    
    return canvas_w, canvas_h, x_starts


def make_composite_frame(warped_columns, canvas_w, canvas_h, 
                          x_starts, output_w, output_h, padding=20,
                          bg_color=(255, 255, 255)):
    """
    Place warped column images side by side on a canvas.
    
    warped_columns: list of warped images (output_h x output_w x 3)
                    or None for missing columns
    canvas_w, canvas_h: total canvas dimensions
    x_starts: list of x offsets per column
    output_w, output_h: size of each column image
    padding: vertical padding (y offset for each column)
    bg_color: background fill color (default white)
    
    Returns composite image (canvas_h x canvas_w x 3)
    """
    canvas = np.full((canvas_h, canvas_w, 3), bg_color, dtype=np.uint8)
    
    for i, warped in enumerate(warped_columns):
        if warped is None:
            continue
        x0 = x_starts[i]
        x1 = x0 + output_w
        y0 = padding
        y1 = y0 + output_h
        canvas[y0:y1, x0:x1] = warped
    
    return canvas

def build_layout_info(n_columns, output_w, output_h, gap=20, padding=20):
    """
    Compute and store all layout parameters needed for
    composite image construction and kymograph indexing.
    
    Returns layout dict with all relevant parameters.
    """
    canvas_w, canvas_h, x_starts = compute_layout(
        n_columns, output_w, output_h, gap=gap, padding=padding
    )
    
    layout = {
        'n_columns':   n_columns,
        'output_w':    output_w,
        'output_h':    output_h,
        'gap':         gap,
        'padding':     padding,
        'canvas_w':    canvas_w,
        'canvas_h':    canvas_h,
        'x_starts':    x_starts,   # x offset of each column in composite
    }
    return layout

def build_composite_frames(filepaths, transforms, layout):
    """
    Reads frames from disk one at a time
    rather than requiring them all loaded in memory.
    
    Returns list of composite images, one per frame.
    """
    composite_frames = []
    
    for i, fp in enumerate(filepaths):
        frame = cv2.imread(str(fp))
        if frame is None:
            print(f"Warning: skipping {fp} — failed to load")
            continue
        
        warped_cols = extract_all_transformed_columns(
            frame, transforms,
            layout['output_w'], layout['output_h']
        )
        composite = make_composite_frame(
            warped_cols,
            canvas_w=layout['canvas_w'],
            canvas_h=layout['canvas_h'],
            x_starts=layout['x_starts'],
            output_w=layout['output_w'],
            output_h=layout['output_h'],
            padding=layout['padding']
        )
        composite_frames.append(composite)
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i+1} / {len(filepaths)} frames")
    
    print(f"Done — built {len(composite_frames)} composite frames")
    return composite_frames

def build_composite_frames_dynamic(filepaths, layout,
                                               lower_color, upper_color,
                                               n_columns,
                                               ignore_x_min=None,
                                               ignore_x_max=None,
                                               morph_open=2, morph_close=10,
                                               min_area=50,
                                               fallback_transforms=None, 
                                               markers_per_column=4):
    """
    Like build_composite_frames_from_paths but recomputes perspective
    transforms from corner markers on every frame.
    
    If corner detection fails for a frame (wrong number of markers found),
    falls back to fallback_transforms if provided, otherwise skips the frame.
    
    filepaths: list of image paths
    layout: layout dict from build_layout_info
    lower_color, upper_color: HSV bounds for marker detection
    n_columns: number of columns expected
    ignore_x_min, ignore_x_max: x range to ignore
    min_area: minimum component area for marker detection
    fallback_transforms: list of M matrices from calibration frame,
                         used if per-frame detection fails.
                         If None, failed frames are skipped entirely.
    
    Returns list of composite images, one per successfully processed frame.
    """
    composite_frames = []
    built_frames = []
    n_failed = 0
    
    for i, fp in enumerate(filepaths):
        frame = cv2.imread(str(fp))
        if frame is None:
            print(f"Warning: could not load {fp}, skipping")
            n_failed += 1
            continue
        
        # --- detect corners on this frame ---
        try:
            all_corners = get_all_column_corners(
                frame,
                lower_color=lower_color,
                upper_color=upper_color,
                n_columns=n_columns,
                ignore_x_min=ignore_x_min,
                ignore_x_max=ignore_x_max,
                morph_open=morph_open,
                morph_close=morph_close,
                min_area=min_area,
                components_per_column=markers_per_column
            )
            
            # check all columns got valid corners
            if any(c is None for c in all_corners):
                raise ValueError(
                    f"Some columns missing corners: "
                    f"{[i for i,c in enumerate(all_corners) if c is None]}"
                )
            
            # compute transforms from this frame's corners
            transforms = compute_all_column_transforms(
                all_corners,
                layout['output_w'],
                layout['output_h']
            )
        
        except Exception as e:
            if fallback_transforms is not None:
                print(f"Frame {i}: corner detection failed ({e}), "
                      f"using fallback transforms")
                transforms = fallback_transforms
            else:
                print(f"Frame {i}: corner detection failed ({e}), "
                      f"skipping frame")
                n_failed += 1
                continue
        
        # --- warp and composite ---
        warped_cols = extract_all_transformed_columns(
            frame, transforms,
            layout['output_w'], layout['output_h']
        )
        composite = make_composite_frame(
            warped_cols,
            canvas_w=layout['canvas_w'],
            canvas_h=layout['canvas_h'],
            x_starts=layout['x_starts'],
            output_w=layout['output_w'],
            output_h=layout['output_h'],
            padding=layout['padding']
        )
        composite_frames.append(composite)
        built_frames.append(fp)
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i+1} / {len(filepaths)} frames")
    
    print(f"Done — built {len(composite_frames)} composite frames "
          f"({n_failed} failed / skipped)")
    return composite_frames, built_frames