"""
Contains functions for evaluating clustering performance, including the official
CHiME-9 pairwise F1 score.
"""

def compute_pairwise_f1(ground_truth_clusters: dict, predicted_clusters: dict):
    """
    Computes the pairwise F1 score, precision, and recall for conversation clustering.

    This metric evaluates clustering by checking every pair of speakers and determining
    if they were correctly placed in the same or different clusters compared to the
    ground truth.

    Args:
        ground_truth_clusters (dict): A dictionary mapping speaker IDs to their
                                      ground truth cluster ID.
                                      Example: {"spk_0": 0, "spk_1": 0, "spk_2": 1}
        predicted_clusters (dict): A dictionary mapping speaker IDs to their
                                   predicted cluster ID.
                                   Example: {"spk_0": 0, "spk_1": 1, "spk_2": 1}

    Returns:
        tuple: A tuple containing:
            - float: The pairwise F1 score.
            - float: The pairwise precision.
            - float: The pairwise recall.
    """
    # Ensure all speakers in the prediction are in the ground truth
    speakers = list(ground_truth_clusters.keys())
    if set(predicted_clusters.keys()) != set(speakers):
        raise ValueError("Predicted clusters must contain the same set of speakers as ground truth.")

    n_speakers = len(speakers)
    tp = 0  # True Positives
    fp = 0  # False Positives
    fn = 0  # False Negatives

    # Iterate over all unique pairs of speakers
    for i in range(n_speakers):
        for j in range(i + 1, n_speakers):
            spk_i = speakers[i]
            spk_j = speakers[j]

            # Check if the pair is in the same cluster in the ground truth
            gt_same_cluster = (ground_truth_clusters[spk_i] == ground_truth_clusters[spk_j])

            # Check if the pair is in the same cluster in the prediction
            pred_same_cluster = (predicted_clusters[spk_i] == predicted_clusters[spk_j])

            if pred_same_cluster and gt_same_cluster:
                # Correctly clustered together
                tp += 1
            elif pred_same_cluster and not gt_same_cluster:
                # Incorrectly clustered together
                fp += 1
            elif not pred_same_cluster and gt_same_cluster:
                # Incorrectly separated
                fn += 1

    # Calculate precision, recall, and F1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"Evaluation results: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
    return f1, precision, recall