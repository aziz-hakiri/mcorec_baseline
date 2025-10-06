"""
Convert 360° video frames to perspective projections.

Input: 360° video (equirectangular)
Output: 4 perspective views per frame (0°, 90°, 180°, 270°)
Sampling: 2 fps (every 0.5 seconds)
"""

import cv2
import numpy as np
import py360convert
from pathlib import Path
import json
from tqdm import tqdm

def process_video(video_path: str, output_dir: str, fps: int = 2, num_views: int = 4, fov: int = 100):
    """
    Args:
        video_path (str): Path to 360° video.
        output_dir (str): Where to save results.
        fps (int): Sampling rate (frames per second).
        num_views (int): Number of perspective views (default 4 for 0°,90°,180°,270°).
        fov (int): Field of view in degrees.

    Returns:
        dict: Dictionary with metadata about processed frames.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_skip = int(original_fps / fps)
    if frame_skip == 0:
        frame_skip = 1 # Avoid infinite loop if target fps > original_fps

    metadata = {
        "source_video": str(video_path),
        "target_fps": fps,
        "num_views": num_views,
        "fov": fov,
        "frames": []
    }

    view_angles = np.linspace(0, 360, num_views, endpoint=False)

    frame_number = 0
    sampled_frame_count = 0

    pbar = tqdm(total=frame_count, desc="Processing video")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        pbar.update(1)

        if frame_number % frame_skip == 0:
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            frame_metadata = {
                "frame_index": sampled_frame_count,
                "source_frame_number": frame_number,
                "timestamp": timestamp,
                "views": []
            }

            # Assuming the input frame is BGR from OpenCV
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get perspective image height and width
            h, w, _ = frame_rgb.shape
            pers_height = h // 2
            pers_width = pers_height # to maintain aspect ratio for a given FOV

            for i, yaw in enumerate(view_angles):
                pitch = 0 # Horizontal view

                # Generate perspective projection
                perspective_img = py360convert.e2p(
                    e_img=frame_rgb,
                    fov_deg=fov,
                    u_deg=yaw,
                    v_deg=pitch,
                    out_hw=(pers_height, pers_width)
                )

                # Convert back to BGR for saving with OpenCV
                perspective_img_bgr = cv2.cvtColor(perspective_img, cv2.COLOR_RGB2BGR)

                # Save the perspective view as a JPEG image
                img_filename = f"frame_{sampled_frame_count:04d}_view_{i:02d}_yaw_{int(yaw)}.jpg"
                img_path = output_dir / img_filename
                cv2.imwrite(str(img_path), perspective_img_bgr)

                view_info = {
                    "view_index": i,
                    "yaw": yaw,
                    "pitch": pitch,
                    "path": str(img_path)
                }
                frame_metadata["views"].append(view_info)

            metadata["frames"].append(frame_metadata)
            sampled_frame_count += 1

        frame_number += 1

    pbar.close()
    cap.release()

    # Save metadata to a JSON file
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Processed {sampled_frame_count} frames, generating {sampled_frame_count * num_views} perspective images.")

    return metadata