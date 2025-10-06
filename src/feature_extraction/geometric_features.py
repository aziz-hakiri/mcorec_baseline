"""
Computes pairwise affinity between tracks based on their geometric positions.
"""

import json
import numpy as np

def compute_geometric_affinity(tracks_path: str):
    """
    Computes a pairwise affinity matrix for tracks based on their angular positions.

    The affinity rules are:
    - 1.0: Tracks are on opposite sides (150-210° apart), likely in the same conversation.
    - 0.5: Tracks form a potential conversational group (90-140° or 220-270° apart).
    - -0.5: Tracks are side-by-side (<40° apart), unlikely to be in the same conversation.
    - 0.0: Neutral affinity for all other configurations.

    Args:
        tracks_path (str): Path to the tracks.json file.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The computed N x N affinity matrix.
            - list: The list of track IDs corresponding to the matrix rows/columns.
    """
    with open(tracks_path, 'r') as f:
        tracks = json.load(f)

    track_ids = list(tracks.keys())
    n_tracks = len(track_ids)

    if n_tracks == 0:
        return np.array([]), []

    affinity_matrix = np.zeros((n_tracks, n_tracks))

    for i in range(n_tracks):
        for j in range(i + 1, n_tracks):
            track_i_id = track_ids[i]
            track_j_id = track_ids[j]

            yaw_i = tracks[track_i_id]['avg_yaw']
            yaw_j = tracks[track_j_id]['avg_yaw']

            # Calculate the absolute difference in yaw
            diff = abs(yaw_i - yaw_j)

            # Handle the wraparound at 360 degrees
            angular_dist = min(diff, 360 - diff)

            # Apply the geometric rules to determine the affinity score
            if 150 < angular_dist < 210:
                score = 1.0
            elif (90 < angular_dist < 140) or (220 < angular_dist < 270):
                score = 0.5
            elif angular_dist < 40:
                score = -0.5
            else:
                score = 0.0

            affinity_matrix[i, j] = score
            affinity_matrix[j, i] = score

    print("Geometric affinity matrix computed successfully.")
    return affinity_matrix, track_ids