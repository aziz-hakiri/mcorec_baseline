"""
Test case to verify the fix for the min_duration_off parameter bug in segmentation.py

This test demonstrates that the bug in line 37 of src/talking_detector/segmentation.py
incorrectly uses min_duration_on (1.0s) instead of min_duration_off (0.5s) as the
default parameter for merging speech regions.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from talking_detector.segmentation import segment_by_asd, CENTRAL_ASD_CHUNKING_PARAMETERS


def test_min_duration_off_default_parameter():
    """
    Test that verifies the default min_duration_off parameter is correctly used.

    This test creates ASD data with two speech regions separated by a 0.7-second gap
    (17-18 frames at 25 fps). With the correct default min_duration_off of 0.5s,
    these regions should remain separate. With the bug (using 1.0s), they would be
    incorrectly merged.
    """
    print("Testing min_duration_off default parameter...")
    print(f"Expected default min_duration_off: {CENTRAL_ASD_CHUNKING_PARAMETERS['min_duration_off']}s")
    print(f"Expected default min_duration_on: {CENTRAL_ASD_CHUNKING_PARAMETERS['min_duration_on']}s")

    # Create ASD data with two distinct speech regions separated by a gap
    # Region 1: frames 0-49 (2 seconds of speech at 25fps)
    # Gap: frames 50-67 (0.72 seconds at 25fps = 18 frames)
    # Region 2: frames 68-117 (2 seconds of speech at 25fps)

    asd_data = {}

    # First speech region (frames 0-49): high scores
    for frame in range(0, 50):
        asd_data[str(frame)] = 3.0  # Above onset threshold

    # Gap (frames 50-67): low scores - this is a 0.72s gap
    for frame in range(50, 68):
        asd_data[str(frame)] = 0.5  # Below offset threshold

    # Second speech region (frames 68-117): high scores
    for frame in range(68, 118):
        asd_data[str(frame)] = 3.0  # Above onset threshold

    # Call segment_by_asd with default parameters
    segments = segment_by_asd(asd_data, parameters={})

    print(f"\nNumber of segments detected: {len(segments)}")
    for i, seg in enumerate(segments):
        start_frame = seg[0]
        end_frame = seg[-1]
        duration_seconds = (end_frame - start_frame + 1) / 25
        gap_after = None
        if i < len(segments) - 1:
            gap_after = (segments[i+1][0] - seg[-1] - 1) / 25
        print(f"  Segment {i+1}: frames {start_frame}-{end_frame} "
              f"(duration: {duration_seconds:.2f}s)"
              f"{f', gap after: {gap_after:.2f}s' if gap_after is not None else ''}")

    # With the FIX (correct default of 0.5s for min_duration_off):
    # The gap is 17 frames (0.68s), which is GREATER than 0.5s (12.5 frames)
    # Therefore, regions should NOT be merged -> expect 2 segments

    # With the BUG (incorrect default of 1.0s for min_duration_off):
    # The gap is 17 frames (0.68s), which is LESS than 1.0s (25 frames)
    # Therefore, regions would be incorrectly merged -> would get 1 segment

    if len(segments) == 2:
        print("\n✓ PASS: Correctly identified 2 separate speech regions")
        print("  This confirms the bug is FIXED - using correct min_duration_off=0.5s")

        # Verify the segments are approximately correct
        assert segments[0][0] == 0, "First segment should start at frame 0"
        assert 45 <= segments[0][-1] <= 50, "First segment should end around frame 49"
        assert 65 <= segments[1][0] <= 70, "Second segment should start around frame 68"
        assert 115 <= segments[1][-1] <= 120, "Second segment should end around frame 117"

        return True
    elif len(segments) == 1:
        print("\n✗ FAIL: Incorrectly merged regions into 1 segment")
        print("  This indicates the bug is present - using incorrect min_duration_on=1.0s")
        print("  The code is using min_duration_on (1.0s) instead of min_duration_off (0.5s)")
        return False
    else:
        print(f"\n✗ UNEXPECTED: Got {len(segments)} segments (expected 1 with bug or 2 with fix)")
        return False


def test_explicit_min_duration_off_parameter():
    """
    Additional test that explicitly sets min_duration_off to verify parameter handling.
    """
    print("\n" + "="*70)
    print("Testing explicit min_duration_off parameter...")

    # Same ASD data as before
    asd_data = {}
    for frame in range(0, 50):
        asd_data[str(frame)] = 3.0
    for frame in range(50, 68):
        asd_data[str(frame)] = 0.5
    for frame in range(68, 118):
        asd_data[str(frame)] = 3.0

    # Explicitly set min_duration_off to 1.0 - should merge
    segments_merged = segment_by_asd(asd_data, parameters={"min_duration_off": 1.0})
    print(f"With min_duration_off=1.0s: {len(segments_merged)} segment(s)")

    # Explicitly set min_duration_off to 0.5 - should keep separate
    segments_separate = segment_by_asd(asd_data, parameters={"min_duration_off": 0.5})
    print(f"With min_duration_off=0.5s: {len(segments_separate)} segment(s)")

    # With 1.0s threshold, the 0.68s gap should be merged
    assert len(segments_merged) == 1, "Should merge with 1.0s threshold"

    # With 0.5s threshold, the 0.68s gap should keep regions separate
    assert len(segments_separate) == 2, "Should keep separate with 0.5s threshold"

    print("✓ PASS: Explicit parameter setting works correctly")
    return True


if __name__ == "__main__":
    print("="*70)
    print("Testing segmentation.py bug fix")
    print("="*70)

    test1_passed = test_min_duration_off_default_parameter()
    test2_passed = test_explicit_min_duration_off_parameter()

    print("\n" + "="*70)
    if test1_passed and test2_passed:
        print("ALL TESTS PASSED ✓")
        print("The bug has been fixed correctly!")
        sys.exit(0)
    else:
        print("TESTS FAILED ✗")
        print("The bug is still present or the fix introduced new issues.")
        sys.exit(1)
