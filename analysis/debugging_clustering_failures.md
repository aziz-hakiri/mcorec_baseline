# **Debugging Baseline Clustering Failures: A Practical Guide**

## **Overview**

This document provides a systematic procedure for analyzing sessions where the baseline's time-based conversation clustering achieves F1 < 1.0. The goal is to understand **why** the baseline failed in terms of its own temporal overlap logic, and to distinguish between:
- **ASD quality issues** (wrong activity detection → wrong overlap statistics)
- **Clustering logic issues** (correct activity, but overlap-based scoring misinterprets the conversational structure)

---

## **Part 1: Step-by-Step Debugging Procedure**

### **Prerequisites**

You have:
- ✅ Baseline predictions: `data-bin/dev/session_XXX/output/speaker_to_cluster.json`
- ✅ Ground truth: `data-bin/dev/session_XXX/labels/speaker_to_cluster.json`
- ✅ ASD scores: `data-bin/dev/session_XXX/speakers/spk_X/central_crops/track_YY_asd.json`
- ✅ Metadata: `data-bin/dev/session_XXX/metadata.json`

---

### **Step 1: Identify Problematic Sessions**

**Goal:** Find sessions with clustering errors to debug.

**Method 1: Using your external script**
```bash
# You already have this
python your_script.py --compute_f1 > session_f1_scores.txt

# Identify failures
grep -v "F1: 1.0" session_f1_scores.txt | sort -k2 -n
```

**Method 2: Using baseline evaluation script**
```bash
# Evaluate all sessions individually
for session in data-bin/dev/session_*; do
    session_name=$(basename $session)
    python script/evaluate.py --session_dir "$session" 2>&1 | \
        grep "Conversation clustering F1" | \
        awk -v s="$session_name" '{print s, $NF}'
done | sort -k2 -n > session_clustering_f1.txt

# Sessions with F1 < 1.0
awk '$2 < 1.0' session_clustering_f1.txt
```

**Output:** List of problematic sessions, e.g.:
```
session_142  0.6667
session_158  0.7500
session_171  0.8000
```

---

### **Step 2: Load and Compare Cluster Assignments**

**Goal:** Identify which speaker pairs are mis-clustered.

**Create debugging script:** `analysis/debug_session_clustering.py`

```python
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import itertools

def load_clustering(path: str) -> Dict[str, int]:
    """Load speaker_to_cluster.json"""
    with open(path, 'r') as f:
        return json.load(f)

def get_pairwise_errors(pred: Dict[str, int], gt: Dict[str, int]) -> Tuple[List, List, List]:
    """
    Identify misclassified speaker pairs

    Returns:
        false_merges: Pairs incorrectly merged (pred_same=True, gt_same=False)
        false_splits: Pairs incorrectly split (pred_same=False, gt_same=True)
        correct_pairs: Correctly classified pairs
    """
    speakers = list(pred.keys())
    pairs = list(itertools.combinations(speakers, 2))

    false_merges = []
    false_splits = []
    correct_pairs = []

    for i, j in pairs:
        pred_same = (pred[i] == pred[j])
        gt_same = (gt[i] == gt[j])

        if pred_same and not gt_same:
            false_merges.append((i, j))
        elif not pred_same and gt_same:
            false_splits.append((i, j))
        else:
            correct_pairs.append((i, j))

    return false_merges, false_splits, correct_pairs

def print_clustering_comparison(session_dir: str):
    """Print detailed clustering comparison"""
    session_name = Path(session_dir).name

    # Load predictions and ground truth
    pred = load_clustering(f"{session_dir}/output/speaker_to_cluster.json")
    gt = load_clustering(f"{session_dir}/labels/speaker_to_cluster.json")

    # Get errors
    false_merges, false_splits, correct = get_pairwise_errors(pred, gt)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Session: {session_name}")
    print(f"{'='*60}")

    print("\nGround Truth Clustering:")
    for cluster_id in sorted(set(gt.values())):
        members = [spk for spk, cid in gt.items() if cid == cluster_id]
        print(f"  Conversation {cluster_id}: {members}")

    print("\nBaseline Prediction:")
    for cluster_id in sorted(set(pred.values())):
        members = [spk for spk, cid in pred.items() if cid == cluster_id]
        print(f"  Conversation {cluster_id}: {members}")

    print(f"\nPairwise Classification:")
    print(f"  Correct pairs: {len(correct)}")
    print(f"  False merges:  {len(false_merges)}")
    print(f"  False splits:  {len(false_splits)}")

    if false_merges:
        print(f"\n⚠️  FALSE MERGES (incorrectly put in same conversation):")
        for i, j in false_merges:
            print(f"    {i} ↔ {j}  (GT: conv{gt[i]} vs conv{gt[j]}, Pred: conv{pred[i]})")

    if false_splits:
        print(f"\n⚠️  FALSE SPLITS (incorrectly separated):")
        for i, j in false_splits:
            print(f"    {i} ↔ {j}  (GT: conv{gt[i]}, Pred: conv{pred[i]} vs conv{pred[j]})")

    return false_merges, false_splits

if __name__ == "__main__":
    session_dir = sys.argv[1] if len(sys.argv) > 1 else "data-bin/dev/session_132"
    print_clustering_comparison(session_dir)
```

**Usage:**
```bash
python analysis/debug_session_clustering.py data-bin/dev/session_142
```

**Example Output:**
```
============================================================
Session: session_142
============================================================

Ground Truth Clustering:
  Conversation 0: ['spk_0', 'spk_1']
  Conversation 1: ['spk_2', 'spk_3']
  Conversation 2: ['spk_4', 'spk_5']

Baseline Prediction:
  Conversation 0: ['spk_0', 'spk_1', 'spk_2']  ← ERROR: spk_2 merged incorrectly
  Conversation 1: ['spk_3', 'spk_4', 'spk_5']

Pairwise Classification:
  Correct pairs: 11
  False merges:  2
  False splits:  2

⚠️  FALSE MERGES (incorrectly put in same conversation):
    spk_0 ↔ spk_2  (GT: conv0 vs conv1, Pred: conv0)
    spk_1 ↔ spk_2  (GT: conv0 vs conv1, Pred: conv0)

⚠️  FALSE SPLITS (incorrectly separated):
    spk_2 ↔ spk_3  (GT: conv1, Pred: conv0 vs conv1)
    spk_2 ↔ spk_4  (GT: conv1 vs conv2, Pred: conv0 vs conv1)  ← This might actually be correct
```

---

### **Step 3: Extract Speaker Activity Timelines**

**Goal:** Visualize when each speaker talks to understand temporal overlap patterns.

**Create activity extraction script:** `analysis/extract_activity_timelines.py`

```python
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
sys.path.append('.')
from src.cluster.conv_spks import get_speaker_activity_segments

def load_metadata(session_dir: str):
    """Load session metadata"""
    with open(f"{session_dir}/metadata.json", 'r') as f:
        return json.load(f)

def extract_all_activities(session_dir: str):
    """Extract activity segments for all speakers"""
    metadata = load_metadata(session_dir)

    speaker_activities = {}

    for speaker_name, speaker_data in metadata.items():
        # Get ASD paths
        list_tracks_asd = []
        for track in speaker_data['central']['crops']:
            list_tracks_asd.append(f"{session_dir}/{track['asd']}")

        # Get UEM boundaries
        uem_start = speaker_data['central']['uem']['start']
        uem_end = speaker_data['central']['uem']['end']

        # Extract activity segments (uses baseline code)
        segments = get_speaker_activity_segments(list_tracks_asd, uem_start, uem_end)

        speaker_activities[speaker_name] = {
            'segments': segments,
            'uem_start': uem_start,
            'uem_end': uem_end,
            'total_duration': sum(end - start for start, end in segments)
        }

    return speaker_activities

def visualize_timelines(session_dir: str, activities: dict, gt: dict, pred: dict):
    """Create timeline visualization"""
    session_name = Path(session_dir).name
    speakers = sorted(activities.keys())

    fig, ax = plt.subplots(figsize=(16, len(speakers)*0.8))

    # Plot activity segments
    for idx, speaker in enumerate(speakers):
        segments = activities[speaker]['segments']
        gt_conv = gt[speaker]
        pred_conv = pred[speaker]

        # Use different colors for ground truth conversations
        color_map = {0: 'blue', 1: 'orange', 2: 'green', 3: 'red'}
        color = color_map.get(gt_conv, 'gray')

        for start, end in segments:
            ax.barh(idx, end - start, left=start, height=0.8,
                   color=color, alpha=0.6, edgecolor='black', linewidth=0.5)

        # Add speaker label with GT and Pred info
        label = f"{speaker}  (GT:C{gt_conv}, Pred:C{pred_conv})"
        if gt_conv != pred_conv:
            label += " ❌"
        ax.text(-5, idx, label, va='center', ha='right', fontsize=10)

    # Formatting
    ax.set_yticks(range(len(speakers)))
    ax.set_yticklabels([''] * len(speakers))
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_title(f'Speaker Activity Timelines - {session_name}', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', label='GT Conv 0'),
                      Patch(facecolor='orange', label='GT Conv 1'),
                      Patch(facecolor='green', label='GT Conv 2'),
                      Patch(facecolor='red', label='GT Conv 3')]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(f'{session_name}_timelines.png', dpi=150, bbox_inches='tight')
    print(f"Saved timeline visualization to {session_name}_timelines.png")
    plt.show()

def print_activity_summary(activities: dict):
    """Print activity statistics"""
    print("\nSpeaker Activity Summary:")
    print(f"{'Speaker':<10} {'Total Time':>12} {'Num Segments':>12} {'Avg Length':>12}")
    print("-" * 50)

    for speaker in sorted(activities.keys()):
        total_time = activities[speaker]['total_duration']
        segments = activities[speaker]['segments']
        num_segments = len(segments)
        avg_length = total_time / num_segments if num_segments > 0 else 0

        print(f"{speaker:<10} {total_time:>10.1f}s {num_segments:>12} {avg_length:>10.1f}s")

if __name__ == "__main__":
    session_dir = sys.argv[1] if len(sys.argv) > 1 else "data-bin/dev/session_132"

    # Extract activities
    activities = extract_all_activities(session_dir)
    print_activity_summary(activities)

    # Load clusterings
    from debug_session_clustering import load_clustering
    gt = load_clustering(f"{session_dir}/labels/speaker_to_cluster.json")
    pred = load_clustering(f"{session_dir}/output/speaker_to_cluster.json")

    # Visualize
    visualize_timelines(session_dir, activities, gt, pred)
```

**Usage:**
```bash
python analysis/extract_activity_timelines.py data-bin/dev/session_142
```

**Example Output:**
```
Speaker Activity Summary:
Speaker    Total Time   Num Segments   Avg Length
--------------------------------------------------
spk_0          45.2s           12        3.8s
spk_1          38.7s            9        4.3s
spk_2          52.1s           15        3.5s  ← High activity
spk_3          28.4s            7        4.1s
spk_4          31.2s            8        3.9s
spk_5          22.8s            6        3.8s

Saved timeline visualization to session_142_timelines.png
```

**What to look for in the timeline:**
- ✅ **Perfect clustering (F1=1.0):** Clear separation, conversations don't overlap temporally
- ⚠️ **False merges:** Speakers from different conversations have low overlap (look similar to same-conversation speakers)
- ⚠️ **False splits:** Speakers from same conversation have high overlap (look like different conversations)

---

### **Step 4: Compute Pairwise Overlap Statistics**

**Goal:** Understand the exact overlap ratios that drove the clustering decisions.

**Create overlap analysis script:** `analysis/analyze_pairwise_overlaps.py`

```python
import json
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.append('.')
from src.cluster.conv_spks import (
    calculate_overlap_duration,
    calculate_conversation_scores
)
from analysis.extract_activity_timelines import extract_all_activities
from analysis.debug_session_clustering import load_clustering, get_pairwise_errors

def compute_pairwise_overlaps(activities: Dict) -> Dict:
    """
    Compute overlap statistics for all speaker pairs

    Returns dict with structure:
    {
        ('spk_0', 'spk_1'): {
            'overlap': 5.2,
            'non_overlap': 15.8,
            'total': 21.0,
            'overlap_ratio': 0.248,
            'score': 0.752
        },
        ...
    }
    """
    speakers = sorted(activities.keys())
    overlaps = {}

    for i in range(len(speakers)):
        for j in range(i + 1, len(speakers)):
            spk_i = speakers[i]
            spk_j = speakers[j]

            seg_i = activities[spk_i]['segments']
            seg_j = activities[spk_j]['segments']

            overlap, non_overlap = calculate_overlap_duration(seg_i, seg_j)
            total = overlap + non_overlap
            overlap_ratio = overlap / total if total > 0 else 0
            score = 1 - overlap_ratio

            overlaps[(spk_i, spk_j)] = {
                'overlap': overlap,
                'non_overlap': non_overlap,
                'total': total,
                'overlap_ratio': overlap_ratio,
                'score': score,
                'distance': 1 - score
            }

    return overlaps

def print_overlap_analysis(session_dir: str):
    """Print detailed overlap analysis for problematic pairs"""
    session_name = Path(session_dir).name

    # Load data
    activities = extract_all_activities(session_dir)
    gt = load_clustering(f"{session_dir}/labels/speaker_to_cluster.json")
    pred = load_clustering(f"{session_dir}/output/speaker_to_cluster.json")

    # Get errors
    false_merges, false_splits, _ = get_pairwise_errors(pred, gt)

    # Compute overlaps
    overlaps = compute_pairwise_overlaps(activities)

    # Print header
    print(f"\n{'='*80}")
    print(f"Pairwise Overlap Analysis - {session_name}")
    print(f"{'='*80}")

    # FALSE MERGES: Should have HIGH overlap (different conversations), but score is HIGH
    if false_merges:
        print(f"\n🔴 FALSE MERGES (incorrectly put together):")
        print(f"{'Pair':<15} {'GT':<10} {'Overlap':>10} {'Non-Overlap':>12} {'Ratio':>8} {'Score':>8} {'Distance':>10}")
        print("-" * 80)

        for pair in false_merges:
            stats = overlaps[pair]
            gt_info = f"C{gt[pair[0]]} vs C{gt[pair[1]]}"
            print(f"{pair[0]}-{pair[1]:<12} {gt_info:<10} "
                  f"{stats['overlap']:>9.1f}s {stats['non_overlap']:>11.1f}s "
                  f"{stats['overlap_ratio']:>7.2%} {stats['score']:>8.3f} {stats['distance']:>10.3f}")

        print(f"\n💡 Analysis: These pairs have LOW overlap ratio (high score > 0.7)")
        print(f"   → Baseline thinks they're in same conversation")
        print(f"   → But they're actually in different conversations (GT)")
        print(f"   → Likely: Turn-taking between different conversations")

    # FALSE SPLITS: Should have LOW overlap (same conversation), but score is LOW
    if false_splits:
        print(f"\n🔴 FALSE SPLITS (incorrectly separated):")
        print(f"{'Pair':<15} {'GT':<10} {'Overlap':>10} {'Non-Overlap':>12} {'Ratio':>8} {'Score':>8} {'Distance':>10}")
        print("-" * 80)

        for pair in false_splits:
            stats = overlaps[pair]
            gt_info = f"C{gt[pair[0]]}"
            print(f"{pair[0]}-{pair[1]:<12} {gt_info:<10} "
                  f"{stats['overlap']:>9.1f}s {stats['non_overlap']:>11.1f}s "
                  f"{stats['overlap_ratio']:>7.2%} {stats['score']:>8.3f} {stats['distance']:>10.3f}")

        print(f"\n💡 Analysis: These pairs have HIGH overlap ratio (low score < 0.7)")
        print(f"   → Baseline thinks they're in different conversations")
        print(f"   → But they're actually in same conversation (GT)")
        print(f"   → Likely: Frequent interruptions within same conversation")

    # CORRECT CLASSIFICATIONS (for comparison)
    print(f"\n✅ CORRECT CLASSIFICATIONS (sample):")
    print(f"{'Pair':<15} {'GT':<10} {'Overlap':>10} {'Non-Overlap':>12} {'Ratio':>8} {'Score':>8} {'Distance':>10}")
    print("-" * 80)

    # Show a few correct same-conversation pairs
    same_conv_pairs = [(p, s) for p, s in overlaps.items()
                       if gt[p[0]] == gt[p[1]]]
    if same_conv_pairs:
        print("  Same conversation pairs (should have low overlap):")
        for pair, stats in same_conv_pairs[:3]:
            gt_info = f"C{gt[pair[0]]}"
            print(f"  {pair[0]}-{pair[1]:<10} {gt_info:<10} "
                  f"{stats['overlap']:>9.1f}s {stats['non_overlap']:>11.1f}s "
                  f"{stats['overlap_ratio']:>7.2%} {stats['score']:>8.3f} {stats['distance']:>10.3f}")

    # Show a few correct different-conversation pairs
    diff_conv_pairs = [(p, s) for p, s in overlaps.items()
                       if gt[p[0]] != gt[p[1]]]
    if diff_conv_pairs:
        print("  Different conversation pairs (should have high overlap):")
        for pair, stats in diff_conv_pairs[:3]:
            gt_info = f"C{gt[pair[0]]} vs C{gt[pair[1]]}"
            print(f"  {pair[0]}-{pair[1]:<10} {gt_info:<10} "
                  f"{stats['overlap']:>9.1f}s {stats['non_overlap']:>11.1f}s "
                  f"{stats['overlap_ratio']:>7.2%} {stats['score']:>8.3f} {stats['distance']:>10.3f}")

    # Print score distribution
    print(f"\n📊 Score Distribution:")
    all_scores = [s['score'] for s in overlaps.values()]
    print(f"  Min score:  {min(all_scores):.3f}")
    print(f"  Max score:  {max(all_scores):.3f}")
    print(f"  Mean score: {np.mean(all_scores):.3f}")
    print(f"  Median:     {np.median(all_scores):.3f}")
    print(f"  Threshold:  0.700 (distance threshold = 0.300)")

if __name__ == "__main__":
    session_dir = sys.argv[1] if len(sys.argv) > 1 else "data-bin/dev/session_132"
    print_overlap_analysis(session_dir)
```

**Usage:**
```bash
python analysis/analyze_pairwise_overlaps.py data-bin/dev/session_142
```

**Example Output:**
```
================================================================================
Pairwise Overlap Analysis - session_142
================================================================================

🔴 FALSE MERGES (incorrectly put together):
Pair            GT         Overlap  Non-Overlap    Ratio    Score   Distance
--------------------------------------------------------------------------------
spk_0-spk_2     C0 vs C1      8.2s       75.1s     9.82%    0.902      0.098
spk_1-spk_2     C0 vs C1      6.5s       68.3s     8.69%    0.913      0.087

💡 Analysis: These pairs have LOW overlap ratio (high score > 0.7)
   → Baseline thinks they're in same conversation
   → But they're actually in different conversations (GT)
   → Likely: Turn-taking between different conversations

🔴 FALSE SPLITS (incorrectly separated):
Pair            GT         Overlap  Non-Overlap    Ratio    Score   Distance
--------------------------------------------------------------------------------
spk_2-spk_3     C1          18.4s       62.1s    22.85%    0.771      0.229

💡 Analysis: These pairs have HIGH overlap ratio (low score < 0.7)
   → Baseline thinks they're in different conversations
   → But they're actually in same conversation (GT)
   → Likely: Frequent interruptions within same conversation

✅ CORRECT CLASSIFICATIONS (sample):
Pair            GT         Overlap  Non-Overlap    Ratio    Score   Distance
--------------------------------------------------------------------------------
  Same conversation pairs (should have low overlap):
  spk_0-spk_1    C0           5.2s       78.7s     6.19%    0.938      0.062
  spk_4-spk_5    C2           3.8s       50.2s     7.03%    0.930      0.070
  Different conversation pairs (should have high overlap):
  spk_0-spk_3    C0 vs C1    22.1s       51.5s    30.00%    0.700      0.300  ← Right at threshold!
  spk_1-spk_4    C0 vs C2    25.8s       43.1s    37.44%    0.626      0.374

📊 Score Distribution:
  Min score:  0.626
  Max score:  0.938
  Mean score: 0.781
  Median:     0.771
  Threshold:  0.700 (distance threshold = 0.300)
```

**Key insights from this output:**
1. **False merge (spk_0-spk_2):** Only 9.82% overlap → score=0.902 → distance=0.098 < 0.3 → **merged**
   - But they're in different conversations! This is the failure.
   - **Root cause:** Different conversations happened to take turns (low temporal overlap)

2. **False split (spk_2-spk_3):** 22.85% overlap → score=0.771 → distance=0.229 < 0.3 → should merge
   - Actually, distance is below threshold! Why did they split?
   - Need to check **clustering linkage effect** (complete linkage may have prevented merge due to transitive distances)

---

### **Step 5: Visualize Distance Matrix and Dendrogram**

**Goal:** Understand hierarchical clustering decisions.

**Create clustering visualization script:** `analysis/visualize_clustering_process.py`

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import sys

sys.path.append('.')
from src.cluster.conv_spks import calculate_conversation_scores
from analysis.extract_activity_timelines import extract_all_activities
from analysis.debug_session_clustering import load_clustering

def visualize_clustering(session_dir: str):
    """Visualize distance matrix and clustering dendrogram"""
    from pathlib import Path
    session_name = Path(session_dir).name

    # Extract activities
    activities = extract_all_activities(session_dir)
    speakers = sorted(activities.keys())

    # Compute conversation scores (NxN symmetric matrix)
    speaker_segments = {spk: activities[spk]['segments'] for spk in speakers}
    scores = calculate_conversation_scores(speaker_segments)

    # Convert to distance matrix
    distances = 1 - scores

    # Load ground truth and predictions
    gt = load_clustering(f"{session_dir}/labels/speaker_to_cluster.json")
    pred = load_clustering(f"{session_dir}/output/speaker_to_cluster.json")

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 6))

    # Subplot 1: Distance matrix heatmap
    ax1 = plt.subplot(131)
    im = ax1.imshow(distances, cmap='RdYlGn_r', vmin=0, vmax=1)
    ax1.set_xticks(range(len(speakers)))
    ax1.set_yticks(range(len(speakers)))
    ax1.set_xticklabels(speakers, rotation=45)
    ax1.set_yticklabels(speakers)
    ax1.set_title('Distance Matrix\n(1 - overlap_score)', fontweight='bold')

    # Add values and threshold line
    for i in range(len(speakers)):
        for j in range(len(speakers)):
            color = 'white' if distances[i, j] > 0.5 else 'black'
            ax1.text(j, i, f'{distances[i, j]:.2f}',
                    ha='center', va='center', color=color, fontsize=9)

    # Add threshold annotation
    ax1.axhline(y=-0.5, color='red', linewidth=3, linestyle='--', label='Threshold=0.3')
    ax1.text(len(speakers)/2, -0.5, 'Threshold = 0.30',
            ha='center', va='bottom', color='red', fontweight='bold', fontsize=11)

    plt.colorbar(im, ax=ax1, label='Distance')

    # Subplot 2: Hierarchical clustering dendrogram
    ax2 = plt.subplot(132)

    # Convert to condensed distance matrix for scipy
    condensed_distances = squareform(distances)

    # Perform hierarchical clustering
    Z = linkage(condensed_distances, method='complete')

    # Create dendrogram
    dendrogram(Z, labels=speakers, ax=ax2, orientation='top',
              color_threshold=0.3, above_threshold_color='gray')

    # Add horizontal line at threshold
    ax2.axhline(y=0.3, color='red', linewidth=2, linestyle='--', label='Threshold')
    ax2.set_ylabel('Distance', fontsize=12)
    ax2.set_title('Hierarchical Clustering\n(Complete Linkage)', fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # Subplot 3: Ground truth vs prediction comparison
    ax3 = plt.subplot(133)
    ax3.axis('off')

    # Create text summary
    text = f"Session: {session_name}\n\n"
    text += "GROUND TRUTH:\n"
    for cluster_id in sorted(set(gt.values())):
        members = [spk for spk, cid in gt.items() if cid == cluster_id]
        text += f"  Conv {cluster_id}: {', '.join(members)}\n"

    text += "\nBASELINE PREDICTION:\n"
    for cluster_id in sorted(set(pred.values())):
        members = [spk for spk, cid in pred.items() if cid == cluster_id]
        # Mark errors
        marker = ""
        for spk in members:
            if gt[spk] != gt[members[0]]:
                marker = " ❌"
                break
        text += f"  Conv {cluster_id}: {', '.join(members)}{marker}\n"

    text += "\nKEY:\n"
    text += "  ❌ = Incorrectly clustered\n"
    text += f"  Distance threshold = 0.30\n"
    text += f"  Linkage = complete\n"

    ax3.text(0.1, 0.95, text, transform=ax3.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle(f'Clustering Analysis: {session_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save
    plt.savefig(f'{session_name}_clustering_debug.png', dpi=150, bbox_inches='tight')
    print(f"Saved clustering visualization to {session_name}_clustering_debug.png")
    plt.show()

if __name__ == "__main__":
    session_dir = sys.argv[1] if len(sys.argv) > 1 else "data-bin/dev/session_132"
    visualize_clustering(session_dir)
```

**Usage:**
```bash
python analysis/visualize_clustering_process.py data-bin/dev/session_142
```

**What to look for in the dendrogram:**
- **Merge height:** Where do speakers merge? Below or above threshold (0.3)?
- **False merges:** Two speakers merge at distance < 0.3, but shouldn't be together
- **False splits:** Two speakers merge at distance > 0.3 (or via a long path), but should be together

---

### **Step 6: Inspect Raw ASD Scores (Quality Check)**

**Goal:** Determine if clustering failures are due to noisy/incorrect ASD.

**Create ASD quality check script:** `analysis/inspect_asd_quality.py`

```python
import json
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

def load_asd_scores(asd_path: str):
    """Load ASD scores from JSON"""
    with open(asd_path, 'r') as f:
        asd_data = json.load(f)

    # Convert to sorted list of (frame, score)
    frames = sorted([(int(frame), score) for frame, score in asd_data.items()])
    return frames

def check_asd_quality(session_dir: str, speaker: str, track_idx: int = 0):
    """Inspect ASD scores for a specific speaker/track"""
    session_name = Path(session_dir).name

    # Load metadata to get track path
    with open(f"{session_dir}/metadata.json", 'r') as f:
        metadata = json.load(f)

    asd_path = f"{session_dir}/{metadata[speaker]['central']['crops'][track_idx]['asd']}"

    # Load scores
    frames = load_asd_scores(asd_path)
    frame_nums, scores = zip(*frames)

    # Statistics
    print(f"\n{'='*60}")
    print(f"ASD Quality Check: {session_name} - {speaker} - track {track_idx}")
    print(f"{'='*60}")
    print(f"Total frames: {len(frames)}")
    print(f"Score range: [{min(scores):.2f}, {max(scores):.2f}]")
    print(f"Mean score: {np.mean(scores):.2f}")
    print(f"Std dev: {np.std(scores):.2f}")

    # Threshold analysis
    onset_threshold = 1.0
    offset_threshold = 0.8

    high_scores = sum(1 for s in scores if s > onset_threshold)
    medium_scores = sum(1 for s in scores if offset_threshold < s <= onset_threshold)
    low_scores = sum(1 for s in scores if s <= offset_threshold)

    print(f"\nScore distribution:")
    print(f"  > {onset_threshold} (onset):  {high_scores} frames ({100*high_scores/len(frames):.1f}%)")
    print(f"  {offset_threshold}-{onset_threshold}:     {medium_scores} frames ({100*medium_scores/len(frames):.1f}%)")
    print(f"  <= {offset_threshold} (offset): {low_scores} frames ({100*low_scores/len(frames):.1f}%)")

    # Detect potential issues
    issues = []

    # Issue 1: Too many high scores (always talking?)
    if high_scores / len(frames) > 0.8:
        issues.append("⚠️ WARNING: >80% frames above onset threshold (possible false positives)")

    # Issue 2: Very few high scores (mostly silent?)
    if high_scores / len(frames) < 0.1:
        issues.append("⚠️ WARNING: <10% frames above onset threshold (possibly too quiet/missed speech)")

    # Issue 3: Extreme score variance
    if np.std(scores) > 2.0:
        issues.append("⚠️ WARNING: High variance in ASD scores (unstable detection)")

    # Issue 4: Many scores near threshold (ambiguous)
    near_threshold = sum(1 for s in scores if 0.7 < s < 1.1)
    if near_threshold / len(frames) > 0.3:
        issues.append("⚠️ WARNING: >30% frames near threshold (ambiguous detections)")

    if issues:
        print(f"\nPotential ASD quality issues:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print(f"\n✅ ASD scores appear reasonable")

    # Visualization
    plt.figure(figsize=(14, 5))

    # Plot 1: ASD scores over time
    plt.subplot(121)
    plt.plot(frame_nums, scores, linewidth=0.8, alpha=0.7)
    plt.axhline(onset_threshold, color='green', linestyle='--', label=f'Onset ({onset_threshold})')
    plt.axhline(offset_threshold, color='orange', linestyle='--', label=f'Offset ({offset_threshold})')
    plt.xlabel('Frame Number')
    plt.ylabel('ASD Score')
    plt.title(f'ASD Scores: {speaker} track {track_idx}')
    plt.legend()
    plt.grid(alpha=0.3)

    # Plot 2: Score histogram
    plt.subplot(122)
    plt.hist(scores, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(onset_threshold, color='green', linestyle='--', linewidth=2, label=f'Onset ({onset_threshold})')
    plt.axvline(offset_threshold, color='orange', linestyle='--', linewidth=2, label=f'Offset ({offset_threshold})')
    plt.xlabel('ASD Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.suptitle(f'{session_name} - {speaker} track {track_idx}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = f'{session_name}_{speaker}_track{track_idx}_asd.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved ASD visualization to {output_file}")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python inspect_asd_quality.py <session_dir> <speaker> [track_idx]")
        sys.exit(1)

    session_dir = sys.argv[1]
    speaker = sys.argv[2]
    track_idx = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    check_asd_quality(session_dir, speaker, track_idx)
```

**Usage:**
```bash
# Check ASD quality for speakers involved in clustering errors
python analysis/inspect_asd_quality.py data-bin/dev/session_142 spk_0
python analysis/inspect_asd_quality.py data-bin/dev/session_142 spk_2
```

**Example Output:**
```
============================================================
ASD Quality Check: session_142 - spk_2 - track 0
============================================================
Total frames: 1250
Score range: [-1.23, 4.56]
Mean score: 0.82
Std dev: 1.45

Score distribution:
  > 1.0 (onset):  425 frames (34.0%)
  0.8-1.0:        175 frames (14.0%)
  <= 0.8 (offset): 650 frames (52.0%)

Potential ASD quality issues:
  ⚠️ WARNING: >30% frames near threshold (ambiguous detections)

Saved ASD visualization to session_142_spk_2_track0_asd.png
```

---

### **Summary of Step-by-Step Procedure**

| Step | Script | Purpose | Key Outputs |
|------|--------|---------|-------------|
| **1** | External / evaluate.py | Find problematic sessions | List of sessions with F1 < 1.0 |
| **2** | debug_session_clustering.py | Identify mis-clustered pairs | False merges, false splits |
| **3** | extract_activity_timelines.py | Visualize temporal patterns | Timeline plot, activity stats |
| **4** | analyze_pairwise_overlaps.py | Compute overlap statistics | Overlap ratios, scores, distances |
| **5** | visualize_clustering_process.py | Understand clustering decisions | Distance matrix, dendrogram |
| **6** | inspect_asd_quality.py | Check ASD reliability | Score distributions, quality flags |

---

## **Part 2: Failure Pattern Taxonomy**

### **Pattern 1: Turn-Taking Across Conversations**

**Scenario:**
- Two conversations (A and B) happen to take turns speaking
- Conversation A speaks during [0-30s, 60-90s]
- Conversation B speaks during [30-60s, 90-120s]
- **Result:** Very low temporal overlap between A and B

**How overlap-based scoring misinterprets it:**
```python
overlap(A, B) ≈ 0s
non_overlap(A, B) = 60s + 60s = 120s
score(A, B) = 1 - 0/120 = 1.0
distance(A, B) = 0.0 < threshold(0.3)
→ A and B merged into same conversation ❌
```

**Diagnostic signature:**
- False merge between speakers from different conversations
- Very high conversation score (>0.9)
- Timeline shows clear alternation, no overlap
- ASD scores are clean (no quality issues)

**Example in real session:**
```
Session: session_142
False merge: spk_0 (Conv0) ↔ spk_2 (Conv1)
  Overlap: 8.2s / Total: 83.3s = 9.8%
  Score: 0.902, Distance: 0.098 < 0.3 → MERGED
```

**Root cause:**
- **Clustering logic failure**, NOT ASD error
- Baseline assumption violated: "low overlap → same conversation"
- Need additional features: spatial proximity, visual cues, speaker embeddings

---

### **Pattern 2: Interruption-Heavy Same Conversation**

**Scenario:**
- Speakers in the same conversation frequently interrupt each other
- Heated debate, excited discussion, multiple participants talking over each other
- **Result:** High temporal overlap within conversation

**How overlap-based scoring misinterprets it:**
```python
overlap(spk_i, spk_j) = 15s  # both in same conversation, but interrupt
non_overlap = 45s
score = 1 - 15/60 = 0.75 (borderline)
# If overlap slightly higher:
overlap = 20s, score = 1 - 20/65 = 0.692 < 0.7
distance = 0.308 > 0.3 → SPLIT ❌
```

**Diagnostic signature:**
- False split between speakers actually in same conversation
- Low conversation score (<0.7)
- Timeline shows significant temporal overlap
- ASD scores may show many simultaneous activations

**Example:**
```
Session: session_158
False split: spk_1 (Conv0) vs spk_3 (Conv0)
  Overlap: 22.5s / Total: 68.3s = 32.9%
  Score: 0.671, Distance: 0.329 > 0.3 → SPLIT
```

**Root cause:**
- Clustering logic failure (assumption: same conversation → low overlap)
- Multi-party dynamics not modeled
- Complete linkage may exacerbate (transitive distances)

---

### **Pattern 3: Speaker Bridging Conversations**

**Scenario:**
- One speaker (e.g., moderator, shared participant) talks to multiple conversations
- Speaker X belongs to Conv0 but occasionally addresses Conv1
- **Result:** Low overlap between X and Conv1 speakers (because they take turns when X talks to them)

**How overlap-based scoring misinterprets it:**
```python
score(X, Conv0_members) ≈ 0.85 (correct, low overlap)
score(X, Conv1_members) ≈ 0.75 (incorrect, also low overlap)
# With complete linkage:
# If X merges with Conv0 first, then Conv0-Conv1 distance via X is small
→ Entire network merges ❌
```

**Diagnostic signature:**
- One speaker has high scores with multiple distinct groups
- False merges cascade through the bridging speaker
- Dendrogram shows early merge of bridging speaker, then subsequent merges
- Activity timeline shows bridging speaker alternating between groups

**Example:**
```
Session: session_165
Speaker: spk_4 (GT: Conv2, but talks to Conv1 occasionally)
  score(spk_4, spk_2_Conv1) = 0.78
  score(spk_4, spk_5_Conv2) = 0.82
  → spk_4 merges Conv1 and Conv2 together
```

**Root cause:**
- Clustering logic failure (transitivity assumption)
- Complete linkage partially helps but not enough
- Need conversation-level modeling, not just pairwise

---

### **Pattern 4: Dominant Speaker in Multi-Party Conversation**

**Scenario:**
- One speaker talks much more than others in their conversation
- Dominant speaker: 60s total speech
- Quiet partners: 10s each
- **Result:** Overlap ratio becomes unstable due to small denominators

**How overlap-based scoring misinterprets it:**
```python
# Dominant (60s) vs Quiet (10s)
overlap = 5s  # quiet speaker talks during some of dominant's speech
non_overlap = 60 + 10 - 2*5 = 60s
score = 1 - 5/65 = 0.923 (high, correct ✓)

# But if quiet speaker happens to talk during pauses:
overlap = 0s
non_overlap = 70s
score = 1.0 (very high, correct ✓)

# Problem: Small perturbations in quiet speaker's timing have large effect
# If quiet speaker accidentally overlaps more (e.g., ASD error):
overlap = 8s (false positive from ASD)
non_overlap = 62s
score = 1 - 8/70 = 0.886 (still high, robust ✓)
```

**Actually, this pattern is relatively ROBUST!**

**But failure can occur if:**
- Quiet speaker has bad ASD (many false positives during dominant's speech)
- Creates artificially high overlap → low score → false split

**Diagnostic signature:**
- Large imbalance in speaking time
- Quiet speaker has noisy ASD (check with Step 6)
- False split if overlap ratio > 30% despite same conversation

---

### **Pattern 5: Synchronized Reactions Across Conversations**

**Scenario:**
- All speakers react simultaneously to external stimulus
- Shared laughter at a joke, surprise at an event
- **Result:** Temporary overlap spikes across all conversations

**How overlap-based scoring misinterprets it:**
```python
# Conv0: spk_0, spk_1
# Conv1: spk_2, spk_3
# Everyone laughs at t=45s for 5 seconds

overlap(spk_0, spk_2) += 5s
# If this is significant portion of total speaking time:
# spk_0 total: 30s, spk_2 total: 25s
overlap = 5s, non_overlap = 50s
score = 1 - 5/55 = 0.909 (still high, probably safe)

# But if speakers are generally quiet:
# spk_0 total: 15s, spk_2 total: 12s
overlap = 5s, non_overlap = 22s
score = 1 - 5/27 = 0.815 (borderline)
```

**Diagnostic signature:**
- Short burst of overlap affecting many pairs
- Timeline shows synchronized activity spike
- Otherwise low overlap outside the spike
- May or may not cause false merge depending on total speaking time

**Root cause:**
- Clustering logic issue (global events not modeled)
- Could be mitigated by outlier removal in overlap calculation

---

### **Pattern 6: Temporal Drift in ASD Onset/Offset**

**Scenario:**
- ASD model has systematic bias (late onset, early offset, or vice versa)
- Two speakers in same conversation, but ASD timing is off
- **Result:** Artificial overlap or gap created

**Example: Late onset bias**
```
Ground truth:
  spk_0: [0-5s speaking]
  spk_1: [6-10s speaking]  # Turn-taking, no overlap

ASD detects:
  spk_0: [0-6s]  # 1s late offset
  spk_1: [6-10s]  # Correct
  → Overlap = 1s (false positive)

If this happens consistently:
  False overlap accumulates → lower score → false split ❌
```

**Diagnostic signature:**
- ASD scores hover near threshold (0.8-1.0) at boundaries
- Systematic bias visible in ASD plots (Step 6)
- Overlap occurs primarily at segment boundaries, not mid-speech
- Manual inspection of video shows no actual simultaneous speech

**Root cause:**
- **ASD quality issue** (not clustering logic)
- Hysteresis thresholds may need tuning
- Temporal smoothing could help (Low-Hanging Fruit #1.1)

---

## **Part 3: ASD Errors vs. Clustering Logic Failures**

### **Diagnostic Decision Tree**

```
Is clustering wrong?
│
├─→ YES: Investigate further
│   │
│   ├─→ Step A: Check overlap statistics (Step 4)
│   │   │
│   │   ├─→ Overlap ratio matches ground truth conversation structure?
│   │   │   │
│   │   │   ├─→ YES (low overlap for same conv, high for diff conv)
│   │   │   │   └─→ ✅ CLUSTERING LOGIC FAILURE
│   │   │   │       • Pattern: Turn-taking, interruptions, bridging
│   │   │   │       • Fix: Better clustering algorithm, add features
│   │   │   │
│   │   │   └─→ NO (overlap ratio doesn't match GT)
│   │   │       └─→ ⚠️  Suspect ASD error, continue to Step B
│   │   │
│   │   └─→ Step B: Inspect ASD quality (Step 6)
│   │       │
│   │       ├─→ ASD scores noisy/unstable?
│   │       │   └─→ ✅ ASD QUALITY ISSUE
│   │       │       • Fix: Temporal smoothing, threshold tuning
│   │       │
│   │       └─→ ASD scores reasonable?
│   │           └─→ Step C: Manual video inspection
│   │               │
│   │               ├─→ Video shows actual simultaneous speech?
│   │               │   └─→ ✅ CLUSTERING LOGIC FAILURE (interruptions)
│   │               │
│   │               └─→ Video shows turn-taking, ASD wrong?
│   │                   └─→ ✅ ASD QUALITY ISSUE (false positives/negatives)
│   │
│   └─→ NO: Clustering correct, no investigation needed
```

---

### **How to Separate ASD Errors from Clustering Logic**

#### **Method 1: Compare Overlap Statistics to Manual Annotation**

**Procedure:**
1. Pick a problematic speaker pair (false merge or split)
2. Extract their activity segments from baseline (Step 3)
3. **Manually watch video** for 2-3 minute segment
4. Count actual simultaneous speech vs. turn-taking
5. Compare to baseline's overlap calculation

**If overlap statistics match video:**
→ **Clustering logic failure** (correct ASD, wrong interpretation)

**If overlap statistics don't match video:**
→ **ASD error** (wrong activity detection)

**Script template:**
```python
# analysis/manual_validation.py
def manual_overlap_check(session_dir, speaker_i, speaker_j, start_time, end_time):
    """
    Manually annotate overlap in a time window
    Compare to baseline's automatic calculation
    """
    print(f"Manual annotation for {speaker_i} vs {speaker_j} from {start_time}s to {end_time}s")
    print("Instructions:")
    print("  1. Watch video carefully")
    print("  2. Mark when BOTH speakers talk simultaneously")
    print("  3. Input overlap intervals below")
    print()

    manual_overlaps = []
    while True:
        overlap_start = input("Overlap start time (or 'done'): ")
        if overlap_start.lower() == 'done':
            break
        overlap_end = input("Overlap end time: ")
        manual_overlaps.append((float(overlap_start), float(overlap_end)))

    # Calculate manual overlap
    manual_overlap_duration = sum(end - start for start, end in manual_overlaps)

    # Load baseline's calculation
    activities = extract_all_activities(session_dir)
    seg_i = [s for s in activities[speaker_i]['segments'] if s[0] < end_time and s[1] > start_time]
    seg_j = [s for s in activities[speaker_j]['segments'] if s[0] < end_time and s[1] > start_time]

    auto_overlap, auto_non_overlap = calculate_overlap_duration(seg_i, seg_j)

    # Compare
    print(f"\n{'='*50}")
    print(f"Comparison:")
    print(f"  Manual overlap:    {manual_overlap_duration:.1f}s")
    print(f"  Baseline overlap:  {auto_overlap:.1f}s")
    print(f"  Difference:        {abs(manual_overlap_duration - auto_overlap):.1f}s")

    if abs(manual_overlap_duration - auto_overlap) < 2.0:
        print(f"\n✅ ASD is CORRECT (within 2s tolerance)")
        print(f"   → Failure is CLUSTERING LOGIC issue")
    else:
        print(f"\n❌ ASD is INCORRECT (>2s error)")
        print(f"   → Failure is ASD QUALITY issue")
```

---

#### **Method 2: ASD Sensitivity Analysis**

**Procedure:**
1. Artificially perturb ASD scores (add noise, shift thresholds)
2. Recompute clustering
3. If clustering changes dramatically → sensitive to ASD, likely ASD error
4. If clustering robust → clustering logic issue

**Script:**
```python
# analysis/asd_sensitivity_test.py
def test_asd_sensitivity(session_dir, speaker, noise_level=0.2):
    """
    Add Gaussian noise to ASD scores and see if clustering changes
    """
    # Load original ASD
    asd_path = f"{session_dir}/speakers/{speaker}/central_crops/track_00_asd.json"
    with open(asd_path, 'r') as f:
        original_asd = json.load(f)

    # Add noise
    noisy_asd = {}
    for frame, score in original_asd.items():
        noisy_asd[frame] = score + np.random.normal(0, noise_level)

    # Save temporarily
    noisy_path = asd_path.replace('_asd.json', '_asd_noisy.json')
    with open(noisy_path, 'w') as f:
        json.dump(noisy_asd, f)

    # Recompute clustering
    # ... (re-run clustering with noisy ASD)

    # Compare results
    print(f"Clustering changed: {original_clusters != noisy_clusters}")
    if original_clusters != noisy_clusters:
        print("→ Clustering is SENSITIVE to ASD noise")
        print("→ Likely ASD quality issue")
    else:
        print("→ Clustering is ROBUST to ASD noise")
        print("→ Likely clustering logic issue")
```

---

#### **Method 3: Check ASD-Overlap Correlation**

**Hypothesis:**
- If ASD is good: overlap ratio should correlate with ground truth conversation structure
- If ASD is bad: overlap ratio is random noise

**Procedure:**
```python
# analysis/asd_overlap_correlation.py
def check_overlap_correlation(session_dir):
    """
    Check if overlap statistics correlate with GT conversation structure
    """
    activities = extract_all_activities(session_dir)
    overlaps = compute_pairwise_overlaps(activities)
    gt = load_clustering(f"{session_dir}/labels/speaker_to_cluster.json")

    # Separate pairs by GT relationship
    same_conv_overlaps = []
    diff_conv_overlaps = []

    for (spk_i, spk_j), stats in overlaps.items():
        if gt[spk_i] == gt[spk_j]:
            same_conv_overlaps.append(stats['overlap_ratio'])
        else:
            diff_conv_overlaps.append(stats['overlap_ratio'])

    # Statistics
    print(f"Same conversation pairs:")
    print(f"  Mean overlap ratio: {np.mean(same_conv_overlaps):.2%}")
    print(f"  Std dev: {np.std(same_conv_overlaps):.2%}")

    print(f"\nDifferent conversation pairs:")
    print(f"  Mean overlap ratio: {np.mean(diff_conv_overlaps):.2%}")
    print(f"  Std dev: {np.std(diff_conv_overlaps):.2%}")

    # Separation analysis
    separation = np.mean(diff_conv_overlaps) - np.mean(same_conv_overlaps)
    print(f"\nSeparation: {separation:.2%}")

    if separation > 0.15:
        print("✅ Good separation: Different convs have >15% more overlap")
        print("   → ASD likely correct, overlap-based logic should work")
    elif separation > 0:
        print("⚠️  Weak separation: Different convs have slightly more overlap")
        print("   → Overlap-based logic is challenging, but ASD may be correct")
    else:
        print("❌ No separation or reversed: Same convs have MORE overlap!")
        print("   → Either ASD error OR assumption violated (interruption-heavy)")
```

**Expected output for good ASD:**
```
Same conversation pairs:
  Mean overlap ratio: 8.5%
  Std dev: 4.2%

Different conversation pairs:
  Mean overlap ratio: 28.3%
  Std dev: 12.1%

Separation: 19.8%
✅ Good separation: Different convs have >15% more overlap
   → ASD likely correct, overlap-based logic should work
```

**If no separation:**
```
Same conversation pairs:
  Mean overlap ratio: 22.1%
  Std dev: 15.3%

Different conversation pairs:
  Mean overlap ratio: 24.7%
  Std dev: 18.9%

Separation: 2.6%
⚠️  Weak separation: Different convs have slightly more overlap
   → Overlap-based logic is challenging, but ASD may be correct
```
→ Likely **clustering logic failure** (interruptions within conversations, or turn-taking across conversations)

---

### **How ASD Errors Propagate to Clustering**

#### **Error Type 1: False Positive ASD (detects speech when silent)**

**Propagation:**
```
Ground truth: Speaker A silent [10-15s]
ASD detects:  Speaker A active [10-15s] (false positive)

If Speaker B active [10-15s]:
  → Artificial overlap created
  → overlap_ratio inflated
  → score decreased
  → distance increased
  → May cause false split ❌
```

**Detection:**
- ASD scores near or above threshold during silent periods
- Check video: speaker not actually talking
- Manual annotation shows lower overlap than baseline

---

#### **Error Type 2: False Negative ASD (misses actual speech)**

**Propagation:**
```
Ground truth: Speaker A active [20-25s]
ASD detects:  Speaker A silent [20-25s] (false negative)

If Speaker B active [20-25s]:
  → Real overlap missed
  → overlap_ratio underestimated
  → score increased (false high)
  → distance decreased
  → May cause false merge ❌
```

**Detection:**
- ASD scores below offset threshold during active speech
- Check video: speaker clearly talking
- Manual annotation shows higher overlap than baseline

---

#### **Error Type 3: Temporal Drift (systematic early/late onset)**

**Propagation:**
```
Ground truth: Speaker A [0-5s], Speaker B [5-10s] (perfect turn-taking)
ASD detects:  Speaker A [0-6s], Speaker B [5-10s] (1s late offset for A)
  → 1s false overlap at transition

If this happens for every turn:
  → Overlap accumulates: 10 turns × 1s = 10s artificial overlap
  → Significantly affects clustering
```

**Detection:**
- Overlap occurs primarily at segment boundaries
- ASD scores linger near threshold at speech offsets
- Temporal smoothing (Low-Hanging Fruit #1.1) would help

---

## **Part 4: Complete Debugging Workflow Example**

### **Case Study: session_142 with F1 = 0.6667**

**Step 1: Identify errors**
```bash
python analysis/debug_session_clustering.py data-bin/dev/session_142
```
**Output:**
```
⚠️ FALSE MERGES: spk_0 ↔ spk_2, spk_1 ↔ spk_2
⚠️ FALSE SPLITS: spk_2 ↔ spk_3
```

**Step 2: Visualize timelines**
```bash
python analysis/extract_activity_timelines.py data-bin/dev/session_142
```
**Observation:**
- spk_0 (Conv0) and spk_2 (Conv1) have very little temporal overlap (take turns)
- spk_2 and spk_3 (both Conv1) have moderate overlap (20%)

**Step 3: Check overlap statistics**
```bash
python analysis/analyze_pairwise_overlaps.py data-bin/dev/session_142
```
**Output:**
```
FALSE MERGES:
  spk_0-spk_2: Overlap=8.2s, Ratio=9.8%, Score=0.902 → MERGED

FALSE SPLITS:
  spk_2-spk_3: Overlap=18.4s, Ratio=22.8%, Score=0.771
```

**Step 4: Visualize clustering**
```bash
python analysis/visualize_clustering_process.py data-bin/dev/session_142
```
**Observation:**
- Distance matrix shows spk_0-spk_2 distance = 0.098 (very small)
- Dendrogram shows spk_0-spk_2 merge very early
- spk_2-spk_3 merge at height 0.229 < 0.3, but complete linkage via spk_0 prevents merge

**Step 5: Check ASD quality**
```bash
python analysis/inspect_asd_quality.py data-bin/dev/session_142 spk_0
python analysis/inspect_asd_quality.py data-bin/dev/session_142 spk_2
```
**Output:**
```
spk_0: ✅ ASD scores appear reasonable
spk_2: ✅ ASD scores appear reasonable
```

**Step 6: Check overlap correlation**
```bash
python analysis/asd_overlap_correlation.py data-bin/dev/session_142
```
**Output:**
```
Same conversation pairs: Mean overlap = 12.3%
Different conversation pairs: Mean overlap = 11.8%
Separation: -0.5%

❌ No separation: Same convs have MORE overlap!
   → Assumption violated (interruption-heavy OR turn-taking across convs)
```

**Diagnosis:**
- **Root cause:** CLUSTERING LOGIC FAILURE (Pattern #1: Turn-taking across conversations)
- **NOT an ASD error:** ASD scores are clean and reasonable
- **Explanation:** Conversations 0 and 1 happened to take turns, so low overlap → baseline incorrectly merged them
- **Fix direction:** Need additional features (spatial, visual, speaker identity), not better ASD

---

## **Summary**

### **Debugging Checklist**

- [ ] **Step 1:** Identify sessions with F1 < 1.0
- [ ] **Step 2:** Identify mis-clustered speaker pairs (false merges, false splits)
- [ ] **Step 3:** Visualize activity timelines
- [ ] **Step 4:** Compute pairwise overlap statistics
- [ ] **Step 5:** Visualize distance matrix and dendrogram
- [ ] **Step 6:** Inspect ASD quality for problematic speakers
- [ ] **Step 7:** Check overlap-GT correlation
- [ ] **Step 8:** Manual video validation (if needed)

### **Common Failure Patterns**

| Pattern | Signature | Root Cause | Fix |
|---------|-----------|------------|-----|
| Turn-taking across convs | False merge, high score, no overlap | Clustering logic | Add features |
| Interruption-heavy | False split, low score, high overlap | Clustering logic | Model multi-party |
| Speaker bridging | Cascading false merges | Clustering logic | Better algorithm |
| Dominant speaker | Variable scores, imbalanced activity | Usually robust | Check ASD if fails |
| Synchronized reactions | Borderline scores, burst overlap | Clustering logic | Outlier removal |
| ASD temporal drift | Boundary overlaps, threshold hovering | ASD quality | Smoothing, tuning |

### **ASD vs Logic Decision**

- **ASD error:** Overlap statistics don't match video, noisy scores, poor separation
- **Clustering logic error:** Overlap statistics match video, clean ASD, but wrong interpretation

---

## **Document Information**

- **Generated**: 2025-11-18
- **Repository**: https://github.com/aziz-hakiri/mcorec_baseline
- **Analysis Type**: Debugging Guide for Clustering Failures
- **Purpose**: Systematic procedure to diagnose why baseline clustering fails
