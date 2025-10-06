"""
Clusters face detections across time into tracks corresponding to unique individuals.
"""

import json
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict

def angular_distance(p1, p2):
    """
    Calculates the angular distance between two points in spherical coordinates.
    Yaw is in degrees and wraps around 360.
    """
    yaw1, pitch1 = p1
    yaw2, pitch2 = p2

    # Calculate yaw distance with wraparound
    yaw_diff = abs(yaw1 - yaw2)
    yaw_dist = min(yaw_diff, 360 - yaw_diff)

    # Pitch distance is straightforward
    pitch_dist = abs(pitch1 - pitch2)

    # Return a weighted or combined distance. Here, we use Euclidean distance on the angles.
    # This is a simplification; for more accuracy, one could use Haversine distance.
    return np.sqrt(yaw_dist**2 + pitch_dist**2)

def cluster_detections_into_tracks(detections_path: str, output_path: str, yaw_tolerance: float = 25.0, pitch_tolerance: float = 15.0, min_samples: int = 10):
    """
    Clusters face detections by their spherical position to form tracks.

    Args:
        detections_path (str): Path to the detections.json file.
        output_path (str): Path to save the output tracks.json file.
        yaw_tolerance (float): The maximum yaw distance for detections to be considered in the same neighborhood.
        pitch_tolerance (float): The maximum pitch distance.
        min_samples (int): The minimum number of detections required to form a track.

    Returns:
        dict: A dictionary containing the clustered tracks.
    """
    with open(detections_path, 'r') as f:
        detections = json.load(f)

    all_positions = []
    detection_info = []

    for frame_key, frame_detections in detections.items():
        frame_index = int(frame_key.split('_')[1])
        for det in frame_detections:
            all_positions.append([det['spherical_yaw'], det['spherical_pitch']])
            detection_info.append({
                'frame_index': frame_index,
                'detection': det
            })

    if not all_positions:
        print("No detections found to cluster.")
        # Save an empty tracks file
        with open(output_path, 'w') as f:
            json.dump({}, f, indent=4)
        return {}

    # Use DBSCAN for clustering. The 'eps' parameter defines the neighborhood size.
    # We can approximate 'eps' from the tolerances. Let's use the yaw_tolerance as a starting point.
    # A custom metric could also be used for more precise distance calculation.
    positions_rad = np.deg2rad(all_positions) # DBSCAN works better with radians

    # A simple way to combine tolerances is to use them as weights.
    # However, for DBSCAN's `precomputed` metric, we need a distance matrix.
    # Let's use a simple heuristic for eps based on the provided tolerances.
    # The 'eps' for DBSCAN will be the max distance in our custom metric space.
    # Let's set it based on the hypotenuse of the tolerance rectangle.
    eps_dist = np.sqrt(yaw_tolerance**2 + pitch_tolerance**2)

    clustering = DBSCAN(eps=eps_dist, min_samples=min_samples, metric=angular_distance)
    labels = clustering.fit_predict(np.array(all_positions))

    # Group detections by their assigned cluster label
    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        if label != -1:  # -1 is for noise points (outliers)
            clusters[label].append(detection_info[i])

    tracks = {}
    for track_id, track_detections in clusters.items():
        yaws = [d['detection']['spherical_yaw'] for d in track_detections]
        pitches = [d['detection']['spherical_pitch'] for d in track_detections]

        # Compute the average position carefully, handling yaw wraparound
        avg_yaw_rad = np.arctan2(np.mean(np.sin(np.deg2rad(yaws))), np.mean(np.cos(np.deg2rad(yaws))))
        avg_yaw = np.rad2deg(avg_yaw_rad) % 360
        avg_pitch = np.mean(pitches)

        frame_appearances = sorted(list(set([d['frame_index'] for d in track_detections])))

        tracks[f"track_{track_id}"] = {
            "avg_yaw": avg_yaw,
            "avg_pitch": avg_pitch,
            "num_detections": len(track_detections),
            "frame_appearances": frame_appearances,
            "detections": track_detections # Optionally include all detections
        }

    # Save tracks to the output JSON file
    with open(output_path, 'w') as f:
        # A custom encoder might be needed if detections contain numpy types
        json.dump(tracks, f, indent=4)

    print(f"Tracking complete. Found {len(tracks)} tracks. Saved to {output_path}")
    return tracks