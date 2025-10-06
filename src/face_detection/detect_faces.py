"""
Detects faces in perspective views and converts their positions to spherical coordinates.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

def perspective_to_spherical(px, py, img_width, img_height, view_yaw, view_pitch, fov):
    """
    Convert pixel coordinates in a perspective view to spherical coordinates.

    Args:
        px (float): Pixel x-coordinate of the face center.
        py (float): Pixel y-coordinate of the face center.
        img_width (int): Width of the perspective image.
        img_height (int): Height of the perspective image.
        view_yaw (float): Yaw of the perspective view's center (in degrees).
        view_pitch (float): Pitch of the perspective view's center (in degrees).
        fov (float): Field of View of the perspective camera (in degrees).

    Returns:
        tuple: (final_yaw, final_pitch) in degrees.
    """
    # Normalize pixel coordinates to the range [-1, 1]
    x_norm = (2 * px / img_width) - 1
    y_norm = (2 * py / img_height) - 1

    # This focal length calculation is an approximation.
    # It assumes the FOV corresponds to the image width.
    fov_rad = np.deg2rad(fov)
    focal = img_width / (2 * np.tan(fov_rad / 2))

    # Calculate angular offsets from the view's center using a pinhole camera model
    theta_x = np.arctan(x_norm * img_width / (2 * focal))
    theta_y = np.arctan(y_norm * img_height / (2 * focal))

    # Convert angular offsets to degrees
    delta_yaw = np.rad2deg(theta_x)
    delta_pitch = -np.rad2deg(theta_y)  # y-axis is inverted in image coordinates

    # Add the offsets to the central yaw and pitch of the view
    final_yaw = (view_yaw + delta_yaw) % 360
    final_pitch = np.clip(view_pitch + delta_pitch, -90, 90)

    return final_yaw, final_pitch

def detect_faces_in_views(metadata_path: str, output_path: str):
    """
    Detects faces in all perspective views listed in the metadata file.

    Args:
        metadata_path (str): Path to the metadata.json file.
        output_path (str): Path to save the output detections.json file.

    Returns:
        dict: A dictionary containing all face detections.
    """
    metadata_path = Path(metadata_path)
    output_path = Path(output_path)

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Note: This path might need to be adjusted depending on the environment.
    # It points to the default location for Haar cascades in many OpenCV installations.
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        raise IOError(f"Failed to load Haar Cascade classifier from {cascade_path}")

    all_detections = {}
    fov = metadata['fov']

    for frame_info in tqdm(metadata['frames'], desc="Detecting faces"):
        frame_index = frame_info['frame_index']
        all_detections[f"frame_{frame_index}"] = []

        for view_info in frame_info['views']:
            img_path = Path(view_info['path'])
            if not img_path.exists():
                print(f"Warning: Image file not found: {img_path}")
                continue

            image = cv2.imread(str(img_path))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            img_height, img_width = image.shape[:2]

            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            for (x, y, w, h) in faces:
                face_center_x = x + w / 2
                face_center_y = y + h / 2

                spherical_yaw, spherical_pitch = perspective_to_spherical(
                    px=face_center_x,
                    py=face_center_y,
                    img_width=img_width,
                    img_height=img_height,
                    view_yaw=view_info['yaw'],
                    view_pitch=view_info['pitch'],
                    fov=fov
                )

                detection = {
                    "view_idx": view_info['view_index'],
                    "view_yaw": view_info['yaw'],
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "spherical_yaw": spherical_yaw,
                    "spherical_pitch": spherical_pitch,
                    "confidence": 1.0  # Haar cascades don't provide confidence, so we use a placeholder
                }
                all_detections[f"frame_{frame_index}"].append(detection)

    # Save all detections to the output JSON file
    with open(output_path, 'w') as f:
        json.dump(all_detections, f, indent=4)

    print(f"Face detection complete. Detections saved to {output_path}")
    return all_detections