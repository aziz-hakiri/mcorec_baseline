"""
End-to-end pipeline script to test the visual conversation clustering on a single video.
"""

import argparse
import json
import numpy as np
from pathlib import Path

# Import all necessary functions from the implemented modules
from src.video_processing.project_to_perspectives import process_video
from src.face_detection.detect_faces import detect_faces_in_views
from src.face_detection.track_faces import cluster_detections_into_tracks
from src.feature_extraction.geometric_features import compute_geometric_affinity
from src.clustering.cluster import cluster_from_affinity
from src.evaluation.metrics import compute_pairwise_f1

def match_tracks_to_speakers(tracks: dict, ground_truth_positions: dict):
    """
    Matches detected tracks to ground truth speaker IDs based on the closest average yaw.

    Args:
        tracks (dict): The dictionary of detected tracks from tracks.json.
        ground_truth_positions (dict): A dictionary mapping speaker IDs to their
                                       expected yaw position in degrees.
                                       Example: {"spk_0": 45.0, "spk_1": 135.0}

    Returns:
        dict: A mapping from track_id to speaker_id.
    """
    track_to_speaker = {}

    unmatched_speakers = list(ground_truth_positions.keys())

    for track_id, track_data in tracks.items():
        min_dist = float('inf')
        best_match_spk = None

        track_yaw = track_data['avg_yaw']

        for spk_id in unmatched_speakers:
            gt_yaw = ground_truth_positions[spk_id]

            # Calculate angular distance, handling wraparound
            diff = abs(track_yaw - gt_yaw)
            dist = min(diff, 360 - diff)

            if dist < min_dist:
                min_dist = dist
                best_match_spk = spk_id

        if best_match_spk:
            track_to_speaker[track_id] = best_match_spk
            # To prevent one speaker from being matched to multiple tracks,
            # we remove it from the pool of available speakers.
            unmatched_speakers.remove(best_match_spk)

    print(f"Matched tracks to speakers: {track_to_speaker}")
    return track_to_speaker


def main():
    parser = argparse.ArgumentParser(description="Run the full visual clustering pipeline on a single video.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the 360° test video.")
    parser.add_argument("--output_dir", type=str, default="results/single_video_test", help="Directory to save all outputs.")
    parser.add_argument("--ground_truth_clusters_path", type=str, required=True, help="Path to the ground truth speaker_to_cluster.json file.")
    parser.add_argument("--ground_truth_positions_path", type=str, required=True, help="Path to a JSON file mapping speaker IDs to their yaw positions.")
    parser.add_argument("--n_clusters", type=int, required=True, help="The number of conversations (clusters) to find.")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define paths for intermediate files
    perspectives_dir = output_dir / "perspective_views"
    metadata_path = perspectives_dir / "metadata.json"
    detections_path = output_dir / "detections.json"
    tracks_path = output_dir / "tracks.json"

    # --- Step 1: Video Processing ---
    print("--- Step 1: Generating Perspective Views ---")
    process_video(args.video_path, str(perspectives_dir))
    print(f"Perspective views and metadata saved to {perspectives_dir}\n")

    # --- Step 2: Face Detection ---
    print("--- Step 2: Detecting Faces ---")
    detect_faces_in_views(str(metadata_path), str(detections_path))
    print(f"Detections saved to {detections_path}\n")

    # --- Step 3: Face Tracking ---
    print("--- Step 3: Tracking Faces ---")
    cluster_detections_into_tracks(str(detections_path), str(tracks_path))
    print(f"Tracks saved to {tracks_path}\n")

    # --- Step 4: Geometric Feature Extraction ---
    print("--- Step 4: Computing Geometric Affinity ---")
    affinity_matrix, track_ids = compute_geometric_affinity(str(tracks_path))
    if not track_ids:
        print("No tracks were found. Halting pipeline.")
        return
    print(f"Affinity matrix computed for tracks: {track_ids}\n")

    # --- Step 5: Clustering ---
    print("--- Step 5: Clustering Tracks ---")
    predicted_labels = cluster_from_affinity(affinity_matrix, args.n_clusters)
    # Create a map from track_id to its predicted cluster label
    track_to_cluster_label = {track_id: label for track_id, label in zip(track_ids, predicted_labels)}
    print(f"Clustering results: {track_to_cluster_label}\n")

    # --- Step 6: Evaluation ---
    print("--- Step 6: Evaluating Results ---")
    # Load ground truth files
    with open(args.ground_truth_clusters_path, 'r') as f:
        gt_clusters = json.load(f)
    with open(args.ground_truth_positions_path, 'r') as f:
        gt_positions = json.load(f)

    # Load the generated tracks to get their data for matching
    with open(tracks_path, 'r') as f:
        tracks = json.load(f)

    # Match generated tracks to ground truth speaker IDs
    track_to_speaker_map = match_tracks_to_speakers(tracks, gt_positions)

    # Create the predicted clusters dictionary in the same format as the ground truth
    predicted_clusters_for_eval = {}
    for track_id, spk_id in track_to_speaker_map.items():
        if spk_id in gt_clusters:
            predicted_clusters_for_eval[spk_id] = int(track_to_cluster_label[track_id])

    # Ensure all speakers from ground truth are accounted for
    for spk_id in gt_clusters:
        if spk_id not in predicted_clusters_for_eval:
            print(f"Warning: Speaker {spk_id} could not be matched to a track and will be excluded from evaluation.")
            # Or assign to a default cluster if needed, e.g., -1

    # Compute and print the F1 score
    if not predicted_clusters_for_eval:
        print("Could not generate predictions for evaluation. Skipping F1 score calculation.")
    else:
        compute_pairwise_f1(gt_clusters, predicted_clusters_for_eval)

if __name__ == "__main__":
    main()