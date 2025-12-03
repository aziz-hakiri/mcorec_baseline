#!/usr/bin/env python3
"""
Convert ground-truth VTT transcripts to ASD JSON format for testing.

This script creates "perfect" ASD scores from ground-truth transcripts,
allowing you to test what happens when you have perfect active speaker detection.

Usage:
    python script/convert_vtt_to_asd.py --vtt labels/spk_0.vtt --output_dir test_gt_asd/

    # Process entire session
    python script/convert_vtt_to_asd.py --session_dir data-bin/dev/session_132
"""

import argparse
import json
import os
import re
from pathlib import Path


def parse_vtt_timestamp(timestamp_str):
    """
    Parse VTT timestamp to seconds.
    Format: HH:MM:SS.mmm
    Example: 00:00:21.039 → 21.039 seconds
    """
    match = re.match(r'(\d+):(\d+):(\d+)\.(\d+)', timestamp_str)
    if not match:
        raise ValueError(f"Invalid timestamp format: {timestamp_str}")

    hours, minutes, seconds, milliseconds = map(int, match.groups())
    total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
    return total_seconds


def parse_vtt_file(vtt_path):
    """
    Parse VTT file and extract speech segments.

    Returns:
        List of (start_time, end_time) tuples in seconds
    """
    segments = []

    with open(vtt_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for timestamp line: "HH:MM:SS.mmm --> HH:MM:SS.mmm"
        if '-->' in line:
            parts = line.split('-->')
            if len(parts) == 2:
                start_str = parts[0].strip()
                end_str = parts[1].strip()

                try:
                    start_time = parse_vtt_timestamp(start_str)
                    end_time = parse_vtt_timestamp(end_str)
                    segments.append((start_time, end_time))
                except ValueError as e:
                    print(f"Warning: {e}")

        i += 1

    return segments


def segments_to_asd(segments, fps=25, confidence_score=5.0, video_duration=None):
    """
    Convert speech segments to ASD JSON format.

    Args:
        segments: List of (start_time, end_time) tuples in seconds
        fps: Frames per second (default: 25)
        confidence_score: Score to assign to active frames (default: 5.0)
        video_duration: Total video duration in seconds (optional)

    Returns:
        Dictionary mapping frame indices to confidence scores
    """
    asd = {}

    # Determine max frame
    if video_duration:
        max_frame = int(video_duration * fps)
    elif segments:
        max_frame = int(max([end for _, end in segments]) * fps) + 1
    else:
        max_frame = 0

    # Initialize all frames as non-speech
    for frame_idx in range(max_frame):
        asd[str(frame_idx)] = 0.0

    # Mark active speech frames
    for start_time, end_time in segments:
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        for frame_idx in range(start_frame, end_frame + 1):
            if frame_idx < max_frame:
                asd[str(frame_idx)] = confidence_score

    return asd


def convert_vtt_to_asd(vtt_path, output_path=None, confidence_score=5.0, video_duration=None):
    """
    Convert a VTT file to ASD JSON format.

    Args:
        vtt_path: Path to input VTT file
        output_path: Path to output JSON file (default: same name with _gt_asd.json suffix)
        confidence_score: Score for active frames (default: 5.0)
        video_duration: Total video duration in seconds (optional)

    Returns:
        Path to output JSON file
    """
    # Parse VTT
    print(f"Parsing {vtt_path}...")
    segments = parse_vtt_file(vtt_path)
    print(f"  Found {len(segments)} speech segments")

    if segments:
        total_duration = sum(end - start for start, end in segments)
        print(f"  Total speech duration: {total_duration:.2f}s")

        # Show segment length distribution
        short_count = sum(1 for start, end in segments if (end - start) < 0.5)
        medium_count = sum(1 for start, end in segments if 0.5 <= (end - start) < 1.0)
        long_count = sum(1 for start, end in segments if (end - start) >= 1.0)

        print(f"  Segment distribution:")
        print(f"    < 0.5s: {short_count} segments")
        print(f"    0.5-1.0s: {medium_count} segments")
        print(f"    >= 1.0s: {long_count} segments")

        if short_count + medium_count > 0:
            print(f"  ⚠️  WARNING: {short_count + medium_count} segments are <1.0s")
            print(f"     These will be DELETED with default min_duration_on=1.0s parameter!")

    # Convert to ASD format
    asd = segments_to_asd(segments, confidence_score=confidence_score, video_duration=video_duration)

    # Determine output path
    if output_path is None:
        output_path = vtt_path.replace('.vtt', '_gt_asd.json')

    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(asd, f, indent=2)

    print(f"✓ Saved GT ASD to {output_path}")
    print(f"  Total frames: {len(asd)}")

    return output_path


def process_session(session_dir, output_subdir='gt_asd', confidence_score=5.0):
    """
    Process all VTT files in a session directory.

    Args:
        session_dir: Path to session directory
        output_subdir: Name of subdirectory for GT ASD files (default: 'gt_asd')
        confidence_score: Score for active frames (default: 5.0)
    """
    session_path = Path(session_dir)
    labels_dir = session_path / 'labels'

    if not labels_dir.exists():
        print(f"Error: Labels directory not found: {labels_dir}")
        return

    # Find all VTT files
    vtt_files = list(labels_dir.glob('*.vtt'))

    if not vtt_files:
        print(f"Error: No VTT files found in {labels_dir}")
        return

    print(f"Processing {len(vtt_files)} VTT files from {session_dir}")
    print()

    # Create output directory
    output_dir = session_path / output_subdir
    output_dir.mkdir(exist_ok=True)

    # Process each VTT file
    for vtt_file in sorted(vtt_files):
        output_path = output_dir / f"{vtt_file.stem}_gt_asd.json"
        convert_vtt_to_asd(str(vtt_file), str(output_path), confidence_score=confidence_score)
        print()

    print(f"✓ All files processed. Output directory: {output_dir}")
    print()
    print("Next steps:")
    print(f"  1. Modify your inference script to use GT ASD files from {output_subdir}/")
    print(f"  2. Test with different segmentation parameters:")
    print(f"     - Default params: min_duration_on=1.0s (current baseline)")
    print(f"     - Relaxed params: min_duration_on=0.0s (recommended for GT)")
    print(f"  3. Compare WER results")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ground-truth VTT transcripts to ASD JSON format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single VTT file
  python script/convert_vtt_to_asd.py --vtt data-bin/dev/session_132/labels/spk_0.vtt

  # Process entire session
  python script/convert_vtt_to_asd.py --session_dir data-bin/dev/session_132

  # Use different confidence score
  python script/convert_vtt_to_asd.py --session_dir data-bin/dev/session_132 --confidence 10.0

Notes:
  - Default confidence score is 5.0 (high enough to pass onset threshold)
  - Output files are saved with '_gt_asd.json' suffix
  - For session processing, files are saved in 'gt_asd/' subdirectory
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--vtt', type=str, help='Path to input VTT file')
    group.add_argument('--session_dir', type=str, help='Path to session directory')

    parser.add_argument('--output', type=str, help='Path to output JSON file (for --vtt mode)')
    parser.add_argument('--output_subdir', type=str, default='gt_asd',
                       help='Output subdirectory name for session mode (default: gt_asd)')
    parser.add_argument('--confidence', type=float, default=5.0,
                       help='Confidence score for active frames (default: 5.0)')
    parser.add_argument('--duration', type=float,
                       help='Total video duration in seconds (optional)')

    args = parser.parse_args()

    if args.vtt:
        # Single file mode
        convert_vtt_to_asd(args.vtt, args.output, args.confidence, args.duration)
    else:
        # Session directory mode
        process_session(args.session_dir, args.output_subdir, args.confidence)


if __name__ == '__main__':
    main()
