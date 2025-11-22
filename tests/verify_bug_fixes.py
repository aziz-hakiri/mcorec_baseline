#!/usr/bin/env python3
"""
Standalone verification script for bug fixes that doesn't require heavy dependencies.
This script performs static analysis and basic tests to verify bugs are fixed.
"""
import os
import sys
import re

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def verify_bug1_segmentation():
    """Verify Bug #1: Correct default parameter in segmentation.py"""
    print("Verifying Bug #1: segmentation.py min_duration_off parameter...")

    with open('src/talking_detector/segmentation.py', 'r') as f:
        content = f.read()

    # Check that line 37 uses CENTRAL_ASD_CHUNKING_PARAMETERS["min_duration_off"]
    # not CENTRAL_ASD_CHUNKING_PARAMETERS["min_duration_on"]
    if 'min_duration_off_frames = int(parameters.get("min_duration_off", CENTRAL_ASD_CHUNKING_PARAMETERS["min_duration_off"])' in content:
        print("  ✓ PASS: min_duration_off uses correct default parameter")
        return True
    else:
        print("  ✗ FAIL: min_duration_off still uses wrong default parameter")
        return False


def verify_bug2_norm_text():
    """Verify Bug #2: No duplicate '}' in norm_set"""
    print("Verifying Bug #2: norm_text.py duplicate '}' removal...")

    with open('src/tokenizer/norm_text.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find the norm_set line (around line 136)
    for i, line in enumerate(lines):
        if 'norm_set = set([' in line:
            # Count occurrences of '}'
            # Extract the list content
            match = re.search(r"set\(\[(.*?)\]\)", line)
            if match:
                list_content = match.group(1)
                # Count standalone '}' entries (not part of '}')
                standalone_braces = list_content.count("'}'")

                if standalone_braces == 1:
                    print(f"  ✓ PASS: norm_set has exactly one '}}' character (line {i+1})")
                    return True
                else:
                    print(f"  ✗ FAIL: norm_set has {standalone_braces} '}}' characters (line {i+1})")
                    return False

    print("  ✗ FAIL: Could not find norm_set definition")
    return False


def verify_bug3_adaptive_time_mask():
    """Verify Bug #3: Correct condition in AdaptiveTimeMask"""
    print("Verifying Bug #3: avhubert_dataset.py impossible condition fix...")

    with open('src/dataset/avhubert_dataset.py', 'r') as f:
        content = f.read()

    # Check that the impossible condition is fixed
    if 't_start == t_start + t' in content:
        print("  ✗ FAIL: Still has impossible condition 't_start == t_start + t'")
        return False
    elif 'if t == 0:' in content and 't_end = t_start + t' in content:
        print("  ✓ PASS: Condition changed to 'if t == 0' and assignment fixed")
        return True
    else:
        print("  ✗ FAIL: Could not verify fix")
        return False


def verify_bug4_lip_crop():
    """Verify Bug #4: No undefined variable in error message"""
    print("Verifying Bug #4: lip_crop.py undefined variable fix...")

    with open('script/lip_crop.py', 'r') as f:
        content = f.read()

    # Check that segment_frame is not referenced in error message
    if 'segment_frame[0]' in content or 'segment_frame[-1]' in content:
        print("  ✗ FAIL: Still references undefined segment_frame variable")
        return False
    elif 'Error processing' in content and 'video_path' in content:
        print("  ✓ PASS: Error message fixed to not reference undefined variable")
        return True
    else:
        print("  ✗ FAIL: Could not verify fix")
        return False


def verify_bug5_evaluate():
    """Verify Bug #5: No double-counting of milliseconds in evaluate.py"""
    print("Verifying Bug #5: evaluate.py timestamp double-counting fix...")

    with open('script/evaluate.py', 'r') as f:
        content = f.read()

    # Check that milliseconds are not added to start_in_seconds
    if 'caption.start_time.milliseconds/1000' in content:
        print("  ✗ FAIL: Still adding milliseconds to start_in_seconds")
        return False
    elif 'caption.start_in_seconds < ref_uem_start' in content and \
         'caption.end_in_seconds > ref_uem_end' in content:
        print("  ✓ PASS: Using start_in_seconds and end_in_seconds directly")
        return True
    else:
        print("  ✗ FAIL: Could not verify fix")
        return False


def main():
    """Run all verifications"""
    print("="*70)
    print("Bug Fix Verification Script")
    print("="*70)
    print()

    results = []

    results.append(("Bug #1: segmentation.py parameter", verify_bug1_segmentation()))
    print()
    results.append(("Bug #2: norm_text.py duplicate", verify_bug2_norm_text()))
    print()
    results.append(("Bug #3: avhubert_dataset.py condition", verify_bug3_adaptive_time_mask()))
    print()
    results.append(("Bug #4: lip_crop.py error message", verify_bug4_lip_crop()))
    print()
    results.append(("Bug #5: evaluate.py timestamp", verify_bug5_evaluate()))
    print()

    print("="*70)
    print("SUMMARY")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print()
    print(f"Total: {passed}/{total} tests passed")
    print("="*70)

    return all(result for _, result in results)


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
