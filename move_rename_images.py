from pathlib import Path
import datetime
import shutil
import sys

INPUT_DIR  = Path("pt3_images/new")        # relative to notebook, or use full path
OUTPUT_DIR = Path("pt3_images")  # where condition subfolders will be created

CONDITIONS = ["negative", "undiluted", "1to10", "1to100"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".gif"}

def move_and_rename_images(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, conditions=CONDITIONS, date_str=None):
    """Move and rename images from input_dir to output_dir based on conditions list.
    date_str can be provided (e.g. "20240601") or will default to today's date."""

    images = sorted(
        [f for f in input_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda f: f.name
    )

    print(f"Found {len(images)} image(s) in '{input_dir}':")
    for img in images:
        print(f"  {img.name}")

    if len(images) != len(conditions):
        raise ValueError(
            f"Expected {len(conditions)} images but found {len(images)}. "
            f"Check INPUT_DIR or CONDITIONS list."
        )
    if date_str is None:
        date_str = datetime.date.today().strftime("%Y%m%d")

    plan = []
    for img, condition in zip(images, conditions):
        new_name  = f"{condition}_{date_str}{img.suffix.lower()}"
        dest_path = output_dir / condition / new_name
        plan.append((img, dest_path))

    for src, dst in plan:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        print(f"Moved: {src.name}  →  {dst}")
    print("\nDone!")

def main():
    # if a command line argument is provided, use it as date_str (e.g. "20240601")
    date_str = sys.argv[1] if len(sys.argv) > 1 else None
    move_and_rename_images(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, conditions=CONDITIONS, date_str=date_str)

if __name__ == "__main__":
    main()