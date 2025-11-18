# **Deep Dive: Time-Based Conversation Clustering & Pairwise F1**

## **Overview**

This document provides a code-grounded analysis of the baseline's **time-based conversation clustering** algorithm and the **pairwise F1 metric** used to evaluate clustering performance. The analysis traces through the actual implementation in the CHiME-9 MCoRec baseline repository.

---

## **1. Time-Based Clustering Algorithm Implementation**

### **1.1 High-Level Pipeline** (`script/inference.py:313-334`)

The clustering algorithm is invoked during inference in `mcorec_session_infer()`:

```python
# Process speaker clustering
speaker_segments = {}
for speaker_name, speaker_data in metadata.items():
    list_tracks_asd = []
    for track in speaker_data['central']['crops']:
        list_tracks_asd.append(os.path.join(session_dir, track['asd']))
    uem_start = speaker_data['central']['uem']['start']
    uem_end = speaker_data['central']['uem']['end']
    speaker_activity_segments = get_speaker_activity_segments(list_tracks_asd, uem_start, uem_end)
    speaker_segments[speaker_name] = speaker_activity_segments

scores = calculate_conversation_scores(speaker_segments)
clusters = cluster_speakers(scores, list(speaker_segments.keys()))
output_clusters_file = os.path.join(output_dir, "speaker_to_cluster.json")
with open(output_clusters_file, "w") as f:
    json.dump(clusters, f, indent=4)
```

**Data flow:**
1. For each speaker → extract activity segments from ASD scores
2. Compute pairwise conversation scores from temporal overlaps
3. Cluster speakers using agglomerative clustering
4. Output `speaker_to_cluster.json`

---

### **1.2 Stage 1: Extract Speaker Activity Segments**

**Function:** `get_speaker_activity_segments()` (`src/cluster/conv_spks.py:168-209`)

**Purpose:** Convert frame-wise ASD scores into time segments where the speaker is actively talking.

#### **Input:**
- `pycrop_asd_path`: List of JSON files containing ASD scores (e.g., `['track_00_asd.json', 'track_01_asd.json']`)
- `uem_start`, `uem_end`: Evaluation interval boundaries (in seconds)

#### **Implementation:**

```python
def get_speaker_activity_segments(pycrop_asd_path: List[str], uem_start: int, uem_end: int):
    pycrop_asd_path = sorted(pycrop_asd_path)
    all_frames = dict({})

    # Step 1: Merge all track ASD scores for this speaker
    for asd_path in pycrop_asd_path:
        with open(asd_path, "r") as f:
            asd_data = json.load(f)
            all_frames.update(asd_data)

    sorted_frames = sorted(all_frames.items(), key=lambda x: int(x[0]))

    # Step 2: Apply ASD-based segmentation (hysteresis thresholding)
    activity_segments = segment_by_asd(all_frames)

    # Step 3: Convert frame indices to time (25 fps)
    activity_segments = [
        (int(segment[0])/25, int(segment[-1])/25)
        for segment in activity_segments
    ]

    # Step 4: Align segments to evaluation interval
    aligned_activity_segments = []
    for segment in activity_segments:
        if segment[1] < uem_start:
            continue
        if segment[0] > uem_end:
            break
        aligned_activity_segments.append([
            segment[0] - uem_start,
            segment[1] - uem_start
        ])

    return aligned_activity_segments
```

#### **Key Steps:**

1. **Merge tracks**: Speakers may have multiple face tracks (e.g., when they move or turn). All ASD scores are merged into a single timeline.

2. **Segment by ASD**: Calls `segment_by_asd()` (from `src/talking_detector/segmentation.py`) which applies:
   - **Hysteresis thresholding**: onset=1.0, offset=0.8
   - **Gap filling**: merges segments separated by <0.5s
   - **Duration filtering**: removes segments <1.0s
   - **Chunking**: splits long segments into ≤15s chunks

3. **Frame-to-time conversion**: Converts frame indices to seconds (25 fps)

4. **UEM alignment**: Shifts segment times relative to evaluation start time (`uem_start`)

#### **Output Example:**
```python
speaker_segments = {
    'spk_0': [(0.5, 3.2), (5.1, 8.7), (10.2, 15.3)],
    'spk_1': [(1.8, 4.5), (6.0, 9.2)],
    # ...
}
```

---

### **1.3 Stage 2: Calculate Pairwise Conversation Scores**

**Function:** `calculate_conversation_scores()` (`src/cluster/conv_spks.py:76-115`)

**Purpose:** Compute a conversation likelihood score for each pair of speakers based on temporal overlap patterns.

#### **Implementation:**

```python
def calculate_conversation_scores(speaker_segments: Dict[str, List[Tuple[float, float]]]) -> np.ndarray:
    n_speakers = len(speaker_segments)
    scores = np.zeros((n_speakers, n_speakers))
    speaker_ids = list(speaker_segments.keys())

    for i in range(n_speakers):
        for j in range(i + 1, n_speakers):
            spk1 = speaker_ids[i]
            spk2 = speaker_ids[j]

            overlap, non_overlap = calculate_overlap_duration(
                speaker_segments[spk1],
                speaker_segments[spk2]
            )

            # Calculate conversation likelihood score
            # Higher score when there's less overlap (more likely to be in same conversation)
            if overlap + non_overlap > 0:
                # Normalize overlap by total duration to get overlap ratio
                total_duration = overlap + non_overlap
                overlap_ratio = overlap / total_duration
                # Convert to conversation likelihood (1 - overlap_ratio)
                score = 1 - overlap_ratio
            else:
                score = 0

            scores[i, j] = score
            scores[j, i] = score  # Symmetric

    return scores
```

#### **Key Intuition:**

The algorithm assumes:
- **Same conversation**: Speakers take turns (sequential speech) → **low overlap** → **high score**
- **Different conversations**: Speakers talk simultaneously → **high overlap** → **low score**

The score formula:
```
score(i,j) = 1 - (overlap_duration / total_duration)
```

where `total_duration = overlap + non_overlap = duration_i + duration_j - overlap`

#### **Overlap Calculation Details:**

The helper function `calculate_overlap_duration()` (`src/cluster/conv_spks.py:43-74`):

```python
def calculate_overlap_duration(segments1: List[Tuple[float, float]],
                             segments2: List[Tuple[float, float]]) -> Tuple[float, float]:
    total_overlap = 0.0
    total_non_overlap = 0.0

    # Calculate total duration of each speaker's segments
    total_duration1 = sum(end - start for start, end in segments1)
    total_duration2 = sum(end - start for start, end in segments2)

    # Calculate overlaps
    for start1, end1 in segments1:
        for start2, end2 in segments2:
            # Calculate overlap
            overlap_start = max(start1, start2)
            overlap_end = min(end1, end2)
            if overlap_end > overlap_start:
                total_overlap += overlap_end - overlap_start

    # Calculate non-overlap
    total_non_overlap = total_duration1 + total_duration2 - 2 * total_overlap

    return total_overlap, total_non_overlap
```

#### **Numerical Example:**

Suppose:
- Speaker A talks: [(0, 3), (10, 15)] → total duration = 8s
- Speaker B talks: [(2, 5), (12, 14)] → total duration = 5s

Overlaps:
- (0,3) ∩ (2,5) = (2,3) → 1s
- (10,15) ∩ (12,14) = (12,14) → 2s
- Total overlap = 3s

Non-overlap = 8 + 5 - 2×3 = 7s

Conversation score = 1 - 3/(3+7) = 1 - 0.3 = **0.7**

**Interpretation:** 70% of their combined speaking time is non-overlapping → moderately likely same conversation.

---

### **1.4 Stage 3: Distance Matrix Construction**

The scores are converted to distances for clustering:

```python
# From cluster_speakers() (src/cluster/conv_spks.py:136-137)
distances = 1 - scores
```

**Reasoning:**
- High score (little overlap) → **small distance** → more likely to be clustered together
- Low score (much overlap) → **large distance** → less likely to be clustered together

---

### **1.5 Stage 4: Agglomerative Clustering**

**Function:** `cluster_speakers()` (`src/cluster/conv_spks.py:117-166`)

#### **Implementation:**

```python
def cluster_speakers(scores: np.ndarray,
                    speaker_ids: List[str],
                    threshold: float = 0.7,
                    n_clusters: int = None) -> Dict[int, List[str]]:

    if n_clusters is not None and n_clusters > MAX_CONVERSATIONS:
        raise ValueError(f"Maximum number of conversations is {MAX_CONVERSATIONS}")

    # Convert scores to distance matrix (1 - score)
    distances = 1 - scores

    # Perform hierarchical clustering
    if n_clusters is None:
        # Use threshold to determine number of clusters
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1-threshold,  # = 0.3
            metric='precomputed',
            linkage='complete'
        )
    else:
        clustering = AgglomerativeClustering(
            n_clusters=min(n_clusters, MAX_CONVERSATIONS),
            metric='precomputed',
            linkage='complete'
        )

    cluster_labels = clustering.fit_predict(distances)

    # Speaker to cluster mapping
    spk_to_cluster = {spk_id: label.item() for spk_id, label in zip(speaker_ids, cluster_labels)}
    return spk_to_cluster
```

#### **Clustering Configuration:**

1. **Algorithm:** Hierarchical agglomerative clustering (bottom-up)
   - Starts with each speaker as a singleton cluster
   - Iteratively merges closest clusters

2. **Linkage:** `'complete'` (maximum/farthest-point linkage)
   - Distance between clusters A and B = max distance between any pair of points in A and B
   - Tends to produce compact, spherical clusters
   - More conservative merging (prevents chain effects)

3. **Metric:** `'precomputed'`
   - Uses the pre-computed distance matrix directly

4. **Stopping criterion:**
   - **Default (baseline):** `distance_threshold = 1 - 0.7 = 0.3`
     - Clusters are merged only if their distance ≤ 0.3
     - Equivalently: conversation score ≥ 0.7
     - Means: speakers must have ≤30% overlap ratio to be in same conversation
   - **Alternative:** Fixed `n_clusters` (e.g., if known beforehand)

5. **Constraints:**
   - `MAX_CONVERSATIONS = 4` (enforced in challenge rules)
   - `MAX_SPEAKERS = 8`

#### **Output Format:**

```json
{
    "spk_0": 0,
    "spk_1": 0,
    "spk_2": 1,
    "spk_3": 1,
    "spk_4": 2,
    "spk_5": 2
}
```

This indicates:
- Conversation 0: spk_0, spk_1
- Conversation 1: spk_2, spk_3
- Conversation 2: spk_4, spk_5

---

## **2. Pairwise F1 for Clustering Evaluation**

### **2.1 Metric Definition**

**Function:** `pairwise_f1_score()` (`src/cluster/eval.py:5-44`)

The pairwise F1 metric treats clustering as a **binary classification problem on speaker pairs**:
- **Positive class:** Two speakers in the same conversation
- **Negative class:** Two speakers in different conversations

#### **Implementation:**

```python
def pairwise_f1_score(true_labels: List[int], pred_labels: List[int]) -> float:
    # Generate all unique unordered pairs of indices
    pairs = list(itertools.combinations(range(len(true_labels)), 2))

    # Initialize counts
    tp = fp = fn = 0

    for i, j in pairs:
        # True same-cluster?
        true_same = (true_labels[i] == true_labels[j])
        # Predicted same-cluster?
        pred_same = (pred_labels[i] == pred_labels[j])

        if pred_same and true_same:
            tp += 1
        elif pred_same and not true_same:
            fp += 1
        elif not pred_same and true_same:
            fn += 1
        # True negatives (not same in both) are not used in F1

    # Handle edge cases
    if tp == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    return f1
```

#### **Formal Definition:**

For all unordered speaker pairs `(i, j)`:

- **True Positive (TP):** `pred[i] == pred[j]` **AND** `gt[i] == gt[j]`
  - Correctly predicted as same conversation

- **False Positive (FP):** `pred[i] == pred[j]` **AND** `gt[i] != gt[j]`
  - Incorrectly merged into same conversation (false merge)

- **False Negative (FN):** `pred[i] != pred[j]` **AND** `gt[i] == gt[j]`
  - Incorrectly split into different conversations (false split)

- **True Negative (TN):** `pred[i] != pred[j]` **AND** `gt[i] != gt[j]`
  - Correctly predicted as different conversations (**not used in F1**)

Then:
```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × Precision × Recall / (Precision + Recall)
```

#### **Numerical Example:**

Session with 6 speakers:
- **Ground truth:** `[0, 0, 1, 1, 2, 2]` (3 conversations, 2 speakers each)
- **Prediction:** `[0, 0, 0, 1, 1, 1]` (2 conversations: {spk0,spk1,spk2}, {spk3,spk4,spk5})

All pairs (15 total):
```
Pair    GT_same  Pred_same  Result
(0,1)   True     True       TP ✓
(0,2)   False    True       FP ✗ (merged spk2 into conv0)
(0,3)   False    False      TN
(0,4)   False    False      TN
(0,5)   False    False      TN
(1,2)   False    True       FP ✗
(1,3)   False    False      TN
(1,4)   False    False      TN
(1,5)   False    False      TN
(2,3)   False    False      TN
(2,4)   False    False      TN
(2,5)   False    False      TN
(3,4)   True     True       TP ✓
(3,5)   True     False      FN ✗ (split spk5 from conv1)
(4,5)   True     True       TP ✓
```

Counts: TP=3, FP=2, FN=1, TN=9

```
Precision = 3 / (3+2) = 0.6
Recall    = 3 / (3+1) = 0.75
F1        = 2 × 0.6 × 0.75 / (0.6 + 0.75) = 0.667
```

---

### **2.2 Per-Speaker F1 (for Joint Metric)**

**Function:** `pairwise_f1_score_per_speaker()` (`src/cluster/eval.py:46-87`)

This variant computes F1 for each speaker individually by considering only pairs involving that speaker.

#### **Implementation:**

```python
def pairwise_f1_score_per_speaker(true_labels: List[int], pred_labels: List[int]) -> Dict[int, float]:
    n = len(true_labels)
    scores = {}

    for i in range(n):
        tp = fp = fn = 0
        for j in range(n):
            if i == j:
                continue

            # True and predicted same-cluster relationships between i and j
            true_same = (true_labels[i] == true_labels[j])
            pred_same = (pred_labels[i] == pred_labels[j])

            if pred_same and true_same:
                tp += 1
            elif pred_same and not true_same:
                fp += 1
            elif not pred_same and true_same:
                fn += 1

        # Compute F1 for this speaker
        if tp == 0:
            f1 = 0.0
        else:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        scores[i] = f1

    return scores
```

#### **Usage in Joint Error Metric:**

From `script/evaluate.py:108-113`:

```python
cluster_speaker_to_wer = {}
for speaker, wer in speaker_to_wer.items():
    cluster_speaker_wer = 0.5 * wer + 0.5 * (1 - speaker_clustering_f1_score[speaker])
    cluster_speaker_to_wer[speaker] = cluster_speaker_wer
```

**Joint ASR-Clustering Error Rate:**
```
Error(speaker_i) = 0.5 × WER(speaker_i) + 0.5 × (1 - F1_clustering(speaker_i))
```

This allows the primary metric to reflect both ASR quality and clustering quality at a per-speaker granularity.

---

### **2.3 Evaluation Pipeline** (`script/evaluate.py`)

```python
# Conversation-level F1
conversation_clustering_f1_score = evaluate_conversation_clustering(label_path, output_path)

# Per-speaker F1
speaker_clustering_f1_score = evaluate_speaker_clustering(label_path, output_path)

# Per-speaker WER
speaker_to_wer = evaluate_speaker_transcripts(label_path, output_path, speaker_list, ...)

# Joint metric
for speaker, wer in speaker_to_wer.items():
    joint_error = 0.5 * wer + 0.5 * (1 - speaker_clustering_f1_score[speaker])
```

---

## **3. Failure Modes of Time-Based Clustering**

Based on the algorithm implementation, we can identify several systematic failure modes:

### **3.1 Multi-Party Overlaps (High Concurrent Speech)**

**Scenario:** All speakers in a session have high overlap ratios due to:
- Heated debate with frequent interruptions
- Multiple conversations overlapping temporally
- Laughter, backchannels, or simultaneous agreements

**Failure mechanism:**
```python
score(i,j) = 1 - (overlap / total_duration)
```
If speakers in the **same conversation** frequently interrupt each other:
- High overlap → low score → large distance → **incorrectly split**

**Example:**
- True: `[0, 0]` (both in conversation 0, but interrupt frequently)
- Overlap = 40% of total duration
- Score = 0.6 < threshold (0.7) → clustered separately
- Prediction: `[0, 1]` → **False Negative**

**Code location:** `calculate_conversation_scores()` at `src/cluster/conv_spks.py:103-108`

---

### **3.2 Speaker Bridging Two Conversations**

**Scenario:** One speaker participates in multiple conversations (e.g., moderator, shared participant).

**Failure mechanism:**
- Speaker A talks with both conversation 1 and conversation 2
- Low overlap between A and each group → high scores → **merges conversations**

**Example:**
- True: `[0, 0, 1, 1]` (spk0,spk1 in conv0; spk2,spk3 in conv1)
- Speaker 0 occasionally speaks with speakers in conv1
- Score(0,2) = 0.8, Score(0,3) = 0.8 (high scores with conv1)
- Agglomerative clustering merges all → Prediction: `[0, 0, 0, 0]`
- **False Positives** for pairs (0,2), (0,3), (1,2), (1,3)

**Mitigation in code:**
- Complete linkage helps: requires **all pairs** to have high score for merge
- But may still fail if bridging speaker has many interactions

**Code location:** `cluster_speakers()` linkage choice at `src/cluster/conv_spks.py:147`

---

### **3.3 Uneven Participation (Silent Partners)**

**Scenario:** One speaker in a conversation rarely speaks (shy participant, listener-only).

**Failure mechanism:**
- Speaker B rarely talks → very short `total_duration`
- Even small overlap can produce low score
- Or: no temporal overlap with conversation partners → no strong clustering signal

**Example:**
- True: `[0, 0]` (both in conversation 0)
- Speaker 0: total speaking time = 60s
- Speaker 1: total speaking time = 5s (mostly listening)
- Overlap = 1s
- total_duration = 60 + 5 - 2×1 = 63s
- Score = 1 - 1/63 = 0.984 (very high) ✓

But if speaker 1 speaks during **different times** (e.g., asks questions when others pause):
- Overlap = 0s
- Score = 1.0 (maximum) ✓

Actually, this case is **robust**! The issue is the opposite:

**Reverse problem:** Speaker rarely speaks, so **any small overlap** has outsized impact.

**Better example:**
- Speaker 1 speaks 3s total, all during moments others also speak (backchannels)
- Overlap = 2s, non-overlap = 2s
- Score = 1 - 2/4 = 0.5 < threshold → **incorrectly split**

**Code location:** Division by `overlap + non_overlap` at `src/cluster/conv_spks.py:105-106`

---

### **3.4 Long Silences / Sparse Speech**

**Scenario:** Conversation with long pauses (e.g., formal meeting, waiting for speaker to finish).

**Failure mechanism:**
- Perfect turn-taking → overlap ≈ 0 → score ≈ 1.0
- This actually **works well** for the algorithm ✓

**Not a failure mode** — time-based clustering excels here!

---

### **3.5 Synchronized Laughter or Reactions**

**Scenario:** All speakers laugh or react simultaneously to an external event (video, joke, announcement).

**Failure mechanism:**
- Speakers from **different conversations** have temporary overlap
- If this is a large portion of their total speaking time → score decreases
- May cause **incorrect merging**

**Example:**
- True: `[0, 0, 1, 1]` (2 conversations)
- All 4 speakers laugh simultaneously for 10s
- Speakers in conv0 talk for 20s total each
- Speakers in conv1 talk for 20s total each
- Overlap between (spk0, spk2) = 10s
- Non-overlap = 20 + 20 - 20 = 20s
- Score = 1 - 10/30 = 0.67 < 0.7 → might still split correctly
- But with more laughter: score → 0.5 → **incorrectly merge**

**Code location:** `calculate_overlap_duration()` at `src/cluster/conv_spks.py:62-70`

---

### **3.6 ASD Errors Propagate**

**Scenario:** Active Speaker Detection makes errors (false positives/negatives).

**Failure mechanism:**
- False positive ASD → speaker appears to be talking when silent → creates spurious overlaps
- False negative ASD → misses actual speech → underestimates total duration
- Garbage in → garbage out

**Example:**
- True: Speaker A talks 0-10s (same conv as B)
- ASD detects: Speaker A talks 0-5s only (missed 5-10s)
- Speaker B talks 6-10s
- Detected overlap = 0s (should be 4s)
- Score inflated → may cause **incorrect merge** with other conversations

**Code location:** Input to `get_speaker_activity_segments()` at `src/cluster/conv_spks.py:168`

**Upstream dependency:** `script/asd.py` and `src/talking_detector/segmentation.py`

---

### **3.7 Threshold Sensitivity**

**Scenario:** Performance heavily depends on the fixed threshold `distance_threshold = 0.3` (equivalently, `score_threshold = 0.7`).

**Failure mechanism:**
- Too low threshold → over-splitting (many small clusters)
- Too high threshold → over-merging (few large clusters)
- No adaptive mechanism for different session characteristics

**Code location:** Hardcoded in `cluster_speakers()` at `src/cluster/conv_spks.py:119, 143`

**Current setting:**
```python
threshold: float = 0.7  # default parameter
distance_threshold = 1 - threshold  # = 0.3
```

**Impact:**
- Session with noisy overlaps (e.g., cafeteria) may need higher threshold
- Session with clean turn-taking (e.g., formal meeting) may work with lower threshold
- Current baseline: **one-size-fits-all**

---

### **3.8 Cluster Labeling Ambiguity**

**Scenario:** Cluster IDs are arbitrary (permutation-invariant).

**Not really a failure**, but important for evaluation:
- Pairwise F1 is **permutation-invariant** ✓
- Correct by design: `true_labels=[0,0,1,1]` vs `pred_labels=[5,5,3,3]` gives same F1 as `pred_labels=[0,0,1,1]`

**Code location:** Handled correctly in `pairwise_f1_score()` via `true_same`/`pred_same` comparisons

---

## **4. Summary Table: Failure Modes**

| **Failure Mode** | **Mechanism** | **Effect** | **Severity** |
|------------------|---------------|------------|--------------|
| **Multi-party overlaps** | High interruption rate in same conversation → low score → split | False Negatives | **High** |
| **Speaker bridging** | One speaker talks to multiple conversations → merges them | False Positives | **Medium** |
| **Sparse speech** | Small speaking time makes overlap ratio unstable | False Negatives | **Low** |
| **Synchronized reactions** | Cross-conversation simultaneous events → overlap | False Positives | **Medium** |
| **ASD errors** | Upstream detection errors propagate to clustering | Both FP/FN | **High** |
| **Fixed threshold** | No adaptation to session characteristics | Both FP/FN | **Medium** |

---

## **5. Code Traceability Map**

| **Component** | **File** | **Lines** | **Purpose** |
|---------------|----------|-----------|-------------|
| Main clustering pipeline | `script/inference.py` | 313-334 | Orchestrates clustering during inference |
| Activity extraction | `src/cluster/conv_spks.py` | 168-209 | `get_speaker_activity_segments()` |
| ASD segmentation | `src/talking_detector/segmentation.py` | 23-111 | `segment_by_asd()` - hysteresis thresholding |
| Overlap calculation | `src/cluster/conv_spks.py` | 43-74 | `calculate_overlap_duration()` |
| Score computation | `src/cluster/conv_spks.py` | 76-115 | `calculate_conversation_scores()` |
| Clustering algorithm | `src/cluster/conv_spks.py` | 117-166 | `cluster_speakers()` - agglomerative |
| Pairwise F1 | `src/cluster/eval.py` | 5-44 | `pairwise_f1_score()` |
| Per-speaker F1 | `src/cluster/eval.py` | 46-87 | `pairwise_f1_score_per_speaker()` |
| Evaluation script | `script/evaluate.py` | 15-113 | Computes all metrics |

---

## **6. Key Takeaways**

### **Algorithm Strengths:**
1. **Simple and interpretable**: Clear temporal overlap rationale
2. **Unsupervised**: No training data required for clustering
3. **Fast**: O(N²) pairwise comparisons, efficient agglomerative clustering
4. **Works well for clean turn-taking**: Perfect for formal conversations with minimal overlap

### **Algorithm Weaknesses:**
1. **Assumes overlap = different conversations**: Fails when same-conversation speakers interrupt frequently
2. **No speaker identity**: Purely temporal; doesn't use visual/acoustic speaker embeddings
3. **Sensitive to ASD quality**: Errors in activity detection propagate directly
4. **Fixed threshold**: No adaptation to session-specific characteristics
5. **No multi-modal fusion**: Ignores spatial proximity, gaze, body orientation

### **Evaluation Metric Properties:**
1. **Pairwise F1** is **permutation-invariant** (cluster label order doesn't matter) ✓
2. **Ignores True Negatives** (correct separate pairs) — appropriate for clustering
3. **Per-speaker variant** enables joint ASR-clustering metric
4. **Matches challenge definition** exactly (as specified in docs)

---

## **Document Information**

- **Generated**: 2025-11-18
- **Repository**: https://github.com/aziz-hakiri/mcorec_baseline
- **Analysis Type**: Deep Dive - Time-Based Clustering & Pairwise F1
- **Code Version**: Based on commit 4a0f071
