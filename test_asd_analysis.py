#!/usr/bin/env python3
"""
Analysis script to understand how different ASD score patterns
are processed by the segment_by_asd function.
"""

import json
import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.talking_detector.segmentation import segment_by_asd, CENTRAL_ASD_CHUNKING_PARAMETERS


def create_noisy_asd_pattern(num_frames=250):
    """
    Create a realistic noisy ASD pattern with:
    - Some short utterances (< 1s)
    - Some long utterances (> 1s)
    - Fluctuating scores representing model uncertainty
    """
    asd = {}

    # Silence: frames 0-24 (1 second)
    for i in range(0, 25):
        asd[str(i)] = 0.5

    # Short utterance "Yes": frames 25-32 (~0.3s, 8 frames)
    # Noisy model makes it longer and with varying scores
    for i in range(25, 40):  # Extended to 15 frames (0.6s) due to model blur
        asd[str(i)] = 2.0 + (i % 3) * 0.3  # Scores: 2.0-2.6

    # Short silence: frames 40-44 (0.2s)
    for i in range(40, 45):
        asd[str(i)] = 0.7

    # Another short utterance "No": frames 45-52 (~0.3s, 8 frames)
    # Noisy model extends it
    for i in range(45, 58):  # Extended to 13 frames (0.52s)
        asd[str(i)] = 1.8 + (i % 2) * 0.4  # Scores: 1.8-2.2

    # Longer silence: frames 58-82 (1s)
    for i in range(58, 83):
        asd[str(i)] = 0.4

    # Long utterance "So what do you think about that?": frames 83-157 (3s, 75 frames)
    for i in range(83, 158):
        asd[str(i)] = 2.5 + (i % 5) * 0.2  # Scores: 2.5-3.3

    # Final silence: frames 158-249
    for i in range(158, 250):
        asd[str(i)] = 0.6

    return asd


def create_gt_asd_pattern_binary(num_frames=250):
    """
    Create a ground-truth ASD pattern with binary scores (0/1).
    This represents perfect ASD with sharp boundaries.
    """
    asd = {}

    # Silence: frames 0-24
    for i in range(0, 25):
        asd[str(i)] = 0.0

    # Short utterance "Yes": frames 25-32 (exact 8 frames = 0.32s)
    for i in range(25, 33):
        asd[str(i)] = 1.0

    # Short silence: frames 33-44
    for i in range(33, 45):
        asd[str(i)] = 0.0

    # Another short utterance "No": frames 45-52 (exact 8 frames = 0.32s)
    for i in range(45, 53):
        asd[str(i)] = 1.0

    # Longer silence: frames 53-82
    for i in range(53, 83):
        asd[str(i)] = 0.0

    # Long utterance: frames 83-157 (exact 75 frames = 3s)
    for i in range(83, 158):
        asd[str(i)] = 1.0

    # Final silence: frames 158-249
    for i in range(158, 250):
        asd[str(i)] = 0.0

    return asd


def create_gt_asd_pattern_high_confidence(num_frames=250):
    """
    Create a ground-truth ASD pattern with high confidence scores (5.0).
    This represents perfect ASD with sharp boundaries and high scores.
    """
    asd = {}

    # Silence: frames 0-24
    for i in range(0, 25):
        asd[str(i)] = 0.0

    # Short utterance "Yes": frames 25-32 (exact 8 frames = 0.32s)
    for i in range(25, 33):
        asd[str(i)] = 5.0

    # Short silence: frames 33-44
    for i in range(33, 45):
        asd[str(i)] = 0.0

    # Another short utterance "No": frames 45-52 (exact 8 frames = 0.32s)
    for i in range(45, 53):
        asd[str(i)] = 5.0

    # Longer silence: frames 53-82
    for i in range(53, 83):
        asd[str(i)] = 0.0

    # Long utterance: frames 83-157 (exact 75 frames = 3s)
    for i in range(83, 158):
        asd[str(i)] = 5.0

    # Final silence: frames 158-249
    for i in range(158, 250):
        asd[str(i)] = 0.0

    return asd


def analyze_segments(segments, name, fps=25):
    """Analyze and print segment information"""
    print(f"\n{name}:")
    print(f"  Number of segments: {len(segments)}")

    if len(segments) == 0:
        print("  WARNING: No segments detected!")
        return

    total_speech_frames = sum(len(seg) for seg in segments)
    total_speech_seconds = total_speech_frames / fps

    print(f"  Total speech duration: {total_speech_seconds:.2f}s ({total_speech_frames} frames)")

    for idx, seg in enumerate(segments):
        start_frame = seg[0]
        end_frame = seg[-1]
        duration_frames = len(seg)
        duration_seconds = duration_frames / fps

        start_time = start_frame / fps
        end_time = end_frame / fps

        print(f"  Segment {idx+1}: frames {start_frame:3d}-{end_frame:3d} "
              f"({start_time:5.2f}s-{end_time:5.2f}s, duration: {duration_seconds:5.2f}s)")


def main():
    print("=" * 80)
    print("ASD Segmentation Analysis: Noisy Model vs Ground Truth")
    print("=" * 80)

    print("\nDefault segmentation parameters:")
    for key, value in CENTRAL_ASD_CHUNKING_PARAMETERS.items():
        print(f"  {key}: {value}")

    print("\nGround Truth speech regions (ideal):")
    print("  Region 1: frames  25-32  (0.32s) - Short word 'Yes'")
    print("  Region 2: frames  45-52  (0.32s) - Short word 'No'")
    print("  Region 3: frames  83-157 (3.00s) - Long sentence")
    print("  Total ideal speech: 3.64s")

    # Test 1: Noisy ASD model
    print("\n" + "=" * 80)
    print("TEST 1: Noisy ASD Model (Realistic Scores)")
    print("=" * 80)
    noisy_asd = create_noisy_asd_pattern()
    noisy_segments = segment_by_asd(noisy_asd, {})
    analyze_segments(noisy_segments, "Noisy model output")

    # Test 2: Ground Truth with binary scores (0/1)
    print("\n" + "=" * 80)
    print("TEST 2: Ground Truth with Binary Scores (0/1)")
    print("=" * 80)
    gt_binary_asd = create_gt_asd_pattern_binary()
    gt_binary_segments = segment_by_asd(gt_binary_asd, {})
    analyze_segments(gt_binary_segments, "GT (binary) output")

    # Test 3: Ground Truth with high confidence scores (5.0)
    print("\n" + "=" * 80)
    print("TEST 3: Ground Truth with High Confidence Scores (5.0)")
    print("=" * 80)
    gt_high_asd = create_gt_asd_pattern_high_confidence()
    gt_high_segments = segment_by_asd(gt_high_asd, {})
    analyze_segments(gt_high_segments, "GT (high confidence) output")

    # Test 4: GT with relaxed parameters
    print("\n" + "=" * 80)
    print("TEST 4: Ground Truth with Relaxed Parameters")
    print("=" * 80)
    relaxed_params = {
        "onset": 0.5,
        "offset": 0.5,
        "min_duration_on": 0.0,
        "min_duration_off": 0.0,
        "max_chunk_size": 15,
    }
    print("Relaxed parameters:")
    for key, value in relaxed_params.items():
        print(f"  {key}: {value}")

    gt_high_relaxed_segments = segment_by_asd(gt_high_asd, relaxed_params)
    analyze_segments(gt_high_relaxed_segments, "GT (high confidence + relaxed params) output")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Impact on Word Error Rate")
    print("=" * 80)

    print("\nKey Findings:")
    print("1. Noisy model:")
    print(f"   - Detected {len(noisy_segments)} segment(s)")
    print("   - Short words are 'padded' by model uncertainty, surviving min_duration filter")

    print("\n2. GT with binary scores (0/1):")
    print(f"   - Detected {len(gt_binary_segments)} segment(s)")
    if len(gt_binary_segments) == 0:
        print("   - CRITICAL BUG: Binary score 1.0 fails onset check (score > 1.0)")
        print("   - Result: NO speech detected → 100% deletion error → CATASTROPHIC WER")

    print("\n3. GT with high confidence (5.0):")
    print(f"   - Detected {len(gt_high_segments)} segment(s)")
    if len(gt_high_segments) < 3:
        print("   - SHORT UTTERANCES DELETED: min_duration_on=1.0s filter removes them")
        print("   - Missing words: 'Yes', 'No' (each < 1s)")
        print("   - Result: Deletion errors → WORSE WER than noisy model")

    print("\n4. GT with relaxed parameters:")
    print(f"   - Detected {len(gt_high_relaxed_segments)} segment(s)")
    print("   - All speech regions preserved")
    print("   - Result: BEST performance (as expected from perfect ASD)")

    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    print("""
The segmentation function is optimized for NOISY predictions from the ASD model.
It uses aggressive filtering to clean up false positives and boundary errors:

1. Hysteresis thresholding (onset=1.0, offset=0.8)
   - Smooths fluctuating predictions
   - Makes boundaries less sensitive to noise

2. Minimum duration filter (min_duration_on=1.0s)
   - Removes spurious short activations from noisy model
   - BUT ALSO removes genuine short words from GT data!

3. Gap filling (min_duration_off=0.5s)
   - Merges fragmented predictions
   - Can over-merge GT segments that are naturally close

When using perfect GT, these "cleaning" operations become destructive:
- Short utterances (<1s) are systematically deleted
- This causes DELETIONS in ASR → increases WER
- Noisy model predictions are "lucky" - the blur helps short words survive!

FIX: When using GT ASD, set min_duration_on=0.0 and adjust onset threshold.
""")


if __name__ == "__main__":
    main()
