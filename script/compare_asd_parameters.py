#!/usr/bin/env python3
"""
Compare different ASD segmentation parameters and their impact on segment detection.

This script helps analyze how different parameter choices affect the segmentation
of ASD scores, which is crucial for understanding why GT ASD might give worse WER.

Usage:
    # Analyze a single ASD file with different parameters
    python script/compare_asd_parameters.py --asd_file path/to/asd.json

    # Compare model predictions vs GT
    python script/compare_asd_parameters.py \
        --model_asd path/to/model_asd.json \
        --gt_asd path/to/gt_asd.json
"""

import argparse
import json
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.talking_detector.segmentation import segment_by_asd, CENTRAL_ASD_CHUNKING_PARAMETERS


PARAMETER_CONFIGS = {
    "default": {
        "name": "Default (for noisy model)",
        "params": CENTRAL_ASD_CHUNKING_PARAMETERS,
    },
    "relaxed": {
        "name": "Relaxed (for GT)",
        "params": {
            "onset": 0.5,
            "offset": 0.5,
            "min_duration_on": 0.0,
            "min_duration_off": 0.0,
            "max_chunk_size": 15,
        }
    },
    "minimal": {
        "name": "Minimal filtering",
        "params": {
            "onset": 0.5,
            "offset": 0.5,
            "min_duration_on": 0.16,  # 4 frames @ 25fps
            "min_duration_off": 0.08,  # 2 frames
            "max_chunk_size": 15,
        }
    },
    "strict": {
        "name": "Strict (more aggressive)",
        "params": {
            "onset": 1.5,
            "offset": 1.0,
            "min_duration_on": 1.5,
            "min_duration_off": 0.3,
            "max_chunk_size": 15,
        }
    }
}


def load_asd_file(asd_path):
    """Load ASD JSON file"""
    with open(asd_path, 'r') as f:
        asd = json.load(f)
    return asd


def analyze_asd_scores(asd):
    """Analyze the distribution of ASD scores"""
    scores = [float(v) for v in asd.values()]

    if not scores:
        return None

    stats = {
        "num_frames": len(scores),
        "min": min(scores),
        "max": max(scores),
        "mean": sum(scores) / len(scores),
        "median": sorted(scores)[len(scores) // 2],
    }

    # Count frames by score ranges
    stats["score_ranges"] = {
        "0.0-0.5": sum(1 for s in scores if 0.0 <= s < 0.5),
        "0.5-1.0": sum(1 for s in scores if 0.5 <= s < 1.0),
        "1.0-2.0": sum(1 for s in scores if 1.0 <= s < 2.0),
        "2.0-3.0": sum(1 for s in scores if 2.0 <= s < 3.0),
        ">=3.0": sum(1 for s in scores if s >= 3.0),
    }

    return stats


def analyze_segments(segments, fps=25):
    """Analyze segment characteristics"""
    if not segments:
        return {
            "num_segments": 0,
            "total_speech_frames": 0,
            "total_speech_seconds": 0.0,
            "avg_duration": 0.0,
            "segments": []
        }

    segment_info = []
    for seg in segments:
        duration_frames = len(seg)
        duration_seconds = duration_frames / fps
        segment_info.append({
            "start_frame": seg[0],
            "end_frame": seg[-1],
            "duration_frames": duration_frames,
            "duration_seconds": duration_seconds,
        })

    total_frames = sum(len(seg) for seg in segments)
    total_seconds = total_frames / fps

    # Duration distribution
    duration_bins = {
        "< 0.5s": sum(1 for s in segment_info if s["duration_seconds"] < 0.5),
        "0.5-1.0s": sum(1 for s in segment_info if 0.5 <= s["duration_seconds"] < 1.0),
        "1.0-2.0s": sum(1 for s in segment_info if 1.0 <= s["duration_seconds"] < 2.0),
        "2.0-5.0s": sum(1 for s in segment_info if 2.0 <= s["duration_seconds"] < 5.0),
        ">= 5.0s": sum(1 for s in segment_info if s["duration_seconds"] >= 5.0),
    }

    return {
        "num_segments": len(segments),
        "total_speech_frames": total_frames,
        "total_speech_seconds": total_seconds,
        "avg_duration": total_seconds / len(segments),
        "duration_bins": duration_bins,
        "segments": segment_info
    }


def print_asd_stats(asd_name, asd_path, asd):
    """Print ASD score statistics"""
    print(f"\n{'='*80}")
    print(f"{asd_name}: {asd_path}")
    print(f"{'='*80}")

    stats = analyze_asd_scores(asd)

    print(f"\nScore Statistics:")
    print(f"  Total frames: {stats['num_frames']}")
    print(f"  Score range: [{stats['min']:.2f}, {stats['max']:.2f}]")
    print(f"  Mean score: {stats['mean']:.2f}")
    print(f"  Median score: {stats['median']:.2f}")

    print(f"\nScore Distribution:")
    for range_name, count in stats['score_ranges'].items():
        pct = 100 * count / stats['num_frames']
        bar = '█' * int(pct / 2)
        print(f"  {range_name:>10}: {count:5d} ({pct:5.1f}%) {bar}")


def print_segmentation_results(config_name, config_info, analysis):
    """Print segmentation analysis results"""
    print(f"\n{'-'*80}")
    print(f"Configuration: {config_info['name']}")
    print(f"{'-'*80}")

    print(f"\nParameters:")
    for key, value in config_info['params'].items():
        print(f"  {key:20s}: {value}")

    print(f"\nResults:")
    print(f"  Segments detected: {analysis['num_segments']}")

    if analysis['num_segments'] == 0:
        print(f"  ⚠️  WARNING: No segments detected!")
        return

    print(f"  Total speech: {analysis['total_speech_seconds']:.2f}s ({analysis['total_speech_frames']} frames)")
    print(f"  Average segment duration: {analysis['avg_duration']:.2f}s")

    print(f"\nSegment Duration Distribution:")
    for bin_name, count in analysis['duration_bins'].items():
        if analysis['num_segments'] > 0:
            pct = 100 * count / analysis['num_segments']
            bar = '█' * int(pct / 2)
            print(f"  {bin_name:>10}: {count:3d} ({pct:5.1f}%) {bar}")

    # Detailed segment list
    if analysis['num_segments'] <= 20:  # Only print if not too many
        print(f"\nDetailed Segments:")
        for i, seg in enumerate(analysis['segments'], 1):
            print(f"  {i:2d}. Frames {seg['start_frame']:5d}-{seg['end_frame']:5d} "
                  f"({seg['duration_seconds']:5.2f}s)")


def compare_configurations(asd, configs=None):
    """Compare different parameter configurations on the same ASD data"""
    if configs is None:
        configs = PARAMETER_CONFIGS

    results = {}

    for config_name, config_info in configs.items():
        segments = segment_by_asd(asd, config_info['params'])
        analysis = analyze_segments(segments)
        results[config_name] = {
            "config": config_info,
            "segments": segments,
            "analysis": analysis
        }
        print_segmentation_results(config_name, config_info, analysis)

    return results


def print_comparison_summary(results):
    """Print a summary comparison table"""
    print(f"\n{'='*80}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*80}")

    print(f"\n{'Configuration':<25} {'Segments':>10} {'Duration':>12} {'Avg/Seg':>12} {'<0.5s':>8} {'<1.0s':>8}")
    print(f"{'-'*80}")

    for config_name, result in results.items():
        config = result['config']
        analysis = result['analysis']

        short_segs = analysis['duration_bins'].get('< 0.5s', 0)
        medium_segs = analysis['duration_bins'].get('0.5-1.0s', 0)
        under_1s = short_segs + medium_segs

        print(f"{config['name']:<25} "
              f"{analysis['num_segments']:>10} "
              f"{analysis['total_speech_seconds']:>11.2f}s "
              f"{analysis['avg_duration']:>11.2f}s "
              f"{short_segs:>8} "
              f"{under_1s:>8}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare ASD segmentation with different parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single ASD file
  python script/compare_asd_parameters.py --asd_file model_asd.json

  # Compare model vs GT
  python script/compare_asd_parameters.py --model_asd model_asd.json --gt_asd gt_asd.json

  # Analyze with specific configuration only
  python script/compare_asd_parameters.py --asd_file model_asd.json --config relaxed
        """
    )

    parser.add_argument('--asd_file', type=str, help='Single ASD file to analyze')
    parser.add_argument('--model_asd', type=str, help='Model-predicted ASD file')
    parser.add_argument('--gt_asd', type=str, help='Ground-truth ASD file')
    parser.add_argument('--config', type=str, choices=list(PARAMETER_CONFIGS.keys()),
                       help='Use specific configuration only')

    args = parser.parse_args()

    # Determine which files to process
    files_to_process = []

    if args.asd_file:
        files_to_process.append(("ASD File", args.asd_file))
    elif args.model_asd or args.gt_asd:
        if args.model_asd:
            files_to_process.append(("Model ASD", args.model_asd))
        if args.gt_asd:
            files_to_process.append(("Ground-Truth ASD", args.gt_asd))
    else:
        parser.error("Must specify --asd_file OR --model_asd/--gt_asd")

    # Determine which configs to use
    if args.config:
        configs = {args.config: PARAMETER_CONFIGS[args.config]}
    else:
        configs = PARAMETER_CONFIGS

    # Process each file
    for asd_name, asd_path in files_to_process:
        if not os.path.exists(asd_path):
            print(f"Error: File not found: {asd_path}")
            continue

        # Load ASD
        asd = load_asd_file(asd_path)

        # Print statistics
        print_asd_stats(asd_name, asd_path, asd)

        # Compare configurations
        print(f"\n{'='*80}")
        print(f"SEGMENTATION ANALYSIS")
        print(f"{'='*80}")

        results = compare_configurations(asd, configs)
        print_comparison_summary(results)


if __name__ == '__main__':
    main()
