import os
import re
import subprocess
from pathlib import Path
import argparse
import sys

def check_and_run_colmap(input_dir):
    """
    Checks the directory structure and runs COLMAP on the images as described.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"{input_dir} is not a valid directory.")

    # Find subfolders named t* (t followed by a number)
    t_dirs = [d for d in input_dir.iterdir() if d.is_dir() and re.match(r"t\d+$", d.name)]
    if not t_dirs:
        raise ValueError("No subdirectories matching 't*' found.")

    # Sort t_dirs by number
    t_dirs.sort(key=lambda d: int(d.name[1:]))

    # Check each t* subdir contains only images (jpg/png)
    for t_dir in t_dirs:
        files = list(t_dir.iterdir())
        if not files:
            raise ValueError(f"{t_dir} is empty.")
        for f in files:
            if not (f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]):
                raise ValueError(f"Non-image file found in {t_dir}: {f.name}")

    # Prepare COLMAP workspace
    workspace = input_dir / "colmap_workspace"
    images_dir = workspace / "images"
    sparse_dir = workspace / "sparse"
    database_path = workspace / "database.db"
    workspace.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)
    sparse_dir.mkdir(exist_ok=True)

    # Copy images from t0 to images_dir
    t0_dir = t_dirs[0]
    for img in t0_dir.iterdir():
        target = images_dir / img.name
        if not target.exists():
            os.symlink(img.resolve(), target)

    # Run feature extraction on t0 images
    subprocess.run([
        "colmap", "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(images_dir),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "OPENCV",
    ], check=True)

    # Run exhaustive matcher on t0 images
    subprocess.run([
        "colmap", "exhaustive_matcher",
        "--database_path", str(database_path),
    ], check=True)

    # Run mapper on t0 images
    sparse_dir.mkdir(exist_ok=True)
    subprocess.run([
        "colmap", "mapper",
        "--database_path", str(database_path),
        "--image_path", str(images_dir),
        "--output_path", str(sparse_dir),
        "--Mapper.ba_global_function_tolerance", "0.000001"
    ], check=True)

    # Select a single reconstruction to be used downstream.
    recon_dirs = sorted([d for d in sparse_dir.iterdir() if d.is_dir()])
    if not recon_dirs:
        raise RuntimeError("No sparse reconstruction found.")
    if len(recon_dirs) > 1:
        # Prefer reconstruction '0' if present, otherwise take the first.
        zero_dir = sparse_dir / "0"
        if zero_dir in recon_dirs:
            mapper_output = zero_dir
        else:
            mapper_output = recon_dirs[0]
        print(
            f"[preprocessing] Warning: multiple sparse reconstructions found "
            f"({[d.name for d in recon_dirs]}), using '{mapper_output.name}'.",
            file=sys.stderr,
        )
    else:
        mapper_output = recon_dirs[0]

    # For each remaining t* (t1, t2, ...)
    for t_dir in t_dirs[1:]:
        # Copy new images to images_dir if not already present
        for img in t_dir.iterdir():
            target = images_dir / img.name
            if not target.exists():
                os.symlink(img.resolve(), target)

        # Run feature extraction for new images only
        subprocess.run([
            "colmap", "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(images_dir),
            "--ImageReader.single_camera", "1"
        ], check=True)

        # Register new images
        subprocess.run([
            "colmap", "image_registrator",
            "--database_path", str(database_path),
            "--input_path", str(mapper_output),
            "--output_path", str(mapper_output)
        ], check=True)

        # Vocab tree matching (assumes vocab tree is available at default location)
        subprocess.run([
            "colmap", "vocab_tree_matcher",
            "--database_path", str(database_path),
        ], check=True)

        # Bundle adjustment
        subprocess.run([
            "colmap", "bundle_adjuster",
            "--input_path", str(mapper_output),
            "--output_path", str(mapper_output)
        ], check=True)
        
    # --- Undistort all images in images_dir using COLMAP's image_undistorter ---
    undistorted_dir = images_dir.parent / "undistorted"
    undistorted_dir.mkdir(exist_ok=True)
    # Find the camera model and parameters from the sparse reconstruction
    # Use COLMAP's image_undistorter tool
    subprocess.run([
        "colmap", "image_undistorter",
        "--image_path", str(images_dir),
        "--input_path", str(mapper_output),
        "--output_path", str(undistorted_dir),
        "--output_type", "COLMAP"
    ], check=True)

def main():
    parser = argparse.ArgumentParser(description="Check directory structure and run COLMAP on images.")
    parser.add_argument("--input_dir", type=str, help="Input directory containing t* subfolders with images.", required=True)
    args = parser.parse_args()
    check_and_run_colmap(args.input_dir)

if __name__ == "__main__":
    main()
