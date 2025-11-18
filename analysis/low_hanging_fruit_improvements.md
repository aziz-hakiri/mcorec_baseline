# **Low-Hanging Fruit Improvements: CHiME-9 MCoRec Baseline**

## **Overview**

This document provides a practical, component-by-component analysis of the baseline pipeline with **actionable improvements** that can be implemented within the existing framework. Each suggestion includes:
- What the code currently does and its assumptions
- Specific improvement proposals
- Metrics to monitor
- Ablation experiment designs

---

## **Component 1: Active Speaker Detection (ASD)**

### **Current Implementation** (`script/asd.py`, `src/talking_detector/ASD.py`)

**Model:**
- **Light-ASD** with audio-visual fusion (MFCC + visual features → BGRU → binary classification)
- Pretrained weights: `model-bin/finetuning_TalkSet.model`

**Processing:**
```python
# Multi-scale evaluation across 11 windows
durationSet = {1,1,1,2,2,2,3,3,4,5,6}  # seconds
for duration in durationSet:
    scores = model(audio_chunk, video_chunk)
final_scores = np.mean(all_scores, axis=0)  # Average across scales
```

**Output:** Frame-wise continuous scores (higher = more likely speaking)

**Assumptions:**
1. Raw ASD scores directly reflect speech activity (no calibration)
2. Equal weighting across all temporal scales
3. No temporal smoothing of outputs
4. Fixed model works across all speakers/environments

---

### **Low-Hanging Improvements**

#### **1.1 Temporal Smoothing of ASD Scores**

**Problem:** Frame-by-frame ASD predictions can be noisy, causing jittery segmentation.

**Solution:** Apply temporal smoothing filter before segmentation.

```python
# In script/asd.py, after final_scores calculation
from scipy.ndimage import gaussian_filter1d

# Apply Gaussian smoothing with sigma=3 frames (~0.12s at 25fps)
smoothed_scores = gaussian_filter1d(final_scores, sigma=3)
```

**Implementation:**
- **File:** `script/asd.py:87` (after `final_scores` computation)
- **Effort:** 5 lines of code
- **Parameters to tune:** `sigma` ∈ {1, 2, 3, 5, 7} frames

**Metric to monitor:**
- **Conversation Clustering F1** (primary): Smoother ASD → better clustering
- **Speaker WER** (secondary): Should not degrade (segmentation still correct)

**Ablation design:**
```bash
# Baseline
python script/asd.py --video data-bin/dev/*/speakers/*/central_crops/track_*.mp4

# Add smoothing variants
for sigma in 1 2 3 5 7; do
    python script/asd.py --video ... --temporal_smooth_sigma $sigma
done

# Compare clustering F1 and WER
python script/evaluate.py --session_dir "data-bin/dev/*"
```

---

#### **1.2 Score Normalization/Calibration**

**Problem:** ASD scores are unbounded and may have different distributions per speaker/track.

**Current range:** Observed scores typically in [-2, 5] but not normalized.

**Solution:** Apply per-track score normalization.

```python
# After final_scores computation
from sklearn.preprocessing import StandardScaler

# Z-score normalization (mean=0, std=1)
scaler = StandardScaler()
normalized_scores = scaler.fit_transform(final_scores.reshape(-1, 1)).flatten()

# Or min-max scaling to [0, 1]
score_min, score_max = np.min(final_scores), np.max(final_scores)
normalized_scores = (final_scores - score_min) / (score_max - score_min + 1e-8)
```

**Impact:** Hysteresis thresholds (onset=1.0, offset=0.8) are currently absolute values. Normalization makes thresholds more consistent across tracks.

**Implementation:**
- **File:** `script/asd.py:87`
- **Effort:** 3-5 lines

**Metric to monitor:**
- **Per-speaker WER variance:** Should decrease (more consistent segmentation)
- **Conversation Clustering F1:** May improve

**Ablation design:**
```bash
# Test normalization methods
python script/inference.py --model_type avsr_cocktail \
    --session_dir "data-bin/dev/*" \
    --asd_normalization [none|zscore|minmax]
```

---

#### **1.3 Adaptive Multi-Scale Weighting**

**Problem:** All temporal scales (1s-6s) weighted equally, but some may be more reliable.

**Current:**
```python
allScore.append(scores)  # Equal weight
final_scores = np.round(np.mean(np.array(allScore), axis=0), 1)
```

**Solution:** Learn or heuristically weight scales.

**Heuristic approach:**
```python
# Weight longer scales more heavily (more context)
scale_weights = np.array([1, 1, 1, 1.5, 1.5, 1.5, 2, 2, 2.5, 3, 3.5])
scale_weights = scale_weights / scale_weights.sum()

weighted_scores = np.average(np.array(allScore), axis=0, weights=scale_weights)
```

**Implementation:**
- **File:** `script/asd.py:87`
- **Effort:** 5 lines

**Metric to monitor:**
- **ASD precision/recall** (if ground truth speech activity available)
- **Downstream WER** (better ASD → better segmentation → better transcripts)

**Ablation design:**
```bash
# Test different weighting schemes
# uniform, linear_increasing, exponential
python script/asd.py --video ... --scale_weighting uniform
python script/asd.py --video ... --scale_weighting linear
python script/asd.py --video ... --scale_weighting exponential
```

---

#### **1.4 Post-Processing: Remove Isolated High Scores**

**Problem:** Occasional false positive spikes (single-frame high scores) create tiny segments.

**Solution:** Median filter to remove isolated outliers.

```python
from scipy.signal import medfilt

# Apply median filter with kernel size 5 (0.2s)
filtered_scores = medfilt(final_scores, kernel_size=5)
```

**Implementation:**
- **File:** `script/asd.py:87`
- **Effort:** 2 lines
- **Complements:** Can combine with Gaussian smoothing (median then Gaussian)

**Metric to monitor:**
- **Number of segments per speaker:** Should decrease (fewer spurious segments)
- **Speaker WER:** Should improve or stay same

---

## **Component 2: Face Detection & Mouth Cropping**

### **Current Implementation** (`script/lip_crop.py`, `src/retinaface/`)

**Models:**
- **RetinaFace** (threshold=0.8): Face detection
- **FAN (Face Alignment Network)**: 68 facial landmarks
- **VideoProcess**: Affine transformation + mouth crop (96×96)

**Processing:**
```python
for frame in video:
    detected_faces = face_detector(frame, threshold=0.8)
    landmarks = landmark_detector(frame, detected_faces)
    # Select largest face
    max_face = max(detected_faces, key=lambda bbox: (bbox[2]-bbox[0])+(bbox[3]-bbox[1]))
    mouth_crop = crop_and_warp(frame, landmarks[max_face])
```

**Assumptions:**
1. Largest face is the target speaker ✓ (valid for single-speaker tracks)
2. All frames have detectable faces
3. Landmark interpolation handles missing detections
4. No quality check on detected landmarks

---

### **Low-Hanging Improvements**

#### **2.1 Increase RetinaFace Confidence Threshold**

**Problem:** Low threshold (0.8) may include false positive face detections.

**Current:** `threshold=0.8` in `src/retinaface/detector.py:20`

**Solution:** Experiment with higher thresholds.

```python
self.face_detector = RetinaFacePredictor(
    device=device,
    threshold=0.9,  # Increase from 0.8
    model=RetinaFacePredictor.get_model(model_name),
)
```

**Implementation:**
- **File:** `src/retinaface/detector.py:20`
- **Effort:** 1 line
- **Trade-off:** Higher precision vs. lower recall

**Metric to monitor:**
- **Face detection rate:** `len(detected_faces) > 0` per frame
- **Interpolation rate:** How often linear interpolation is used
- **Speaker WER:** Should improve if false faces were causing bad crops

**Ablation design:**
```bash
# Test thresholds
for thresh in 0.7 0.8 0.9 0.95; do
    python script/lip_crop.py --video ... --face_threshold $thresh
done
```

---

#### **2.2 Landmark Quality Filtering**

**Problem:** FAN may produce low-quality landmarks (e.g., during occlusions, profile views).

**Current:** No quality check; all landmarks accepted or interpolated.

**Solution:** Filter landmarks by confidence or geometric consistency.

```python
def is_landmark_valid(landmarks):
    """Check if landmarks form a reasonable face geometry"""
    # Check mouth landmarks (48-68) are present
    mouth_points = landmarks[48:68]

    # Check aspect ratio (width/height should be reasonable)
    mouth_width = np.max(mouth_points[:, 0]) - np.min(mouth_points[:, 0])
    mouth_height = np.max(mouth_points[:, 1]) - np.min(mouth_points[:, 1])
    aspect_ratio = mouth_width / (mouth_height + 1e-8)

    # Mouth aspect ratio typically 1.5-3.5
    if aspect_ratio < 1.0 or aspect_ratio > 5.0:
        return False

    # Check landmark variance (too clustered = bad detection)
    if np.std(mouth_points) < 1.0:
        return False

    return True

# In detector.py
if len(detected_faces) > 0:
    if not is_landmark_valid(face_points[max_id]):
        landmarks.append(None)  # Trigger interpolation
    else:
        landmarks.append(face_points[max_id])
```

**Implementation:**
- **File:** `src/retinaface/detector.py:29-38`
- **Effort:** 15-20 lines

**Metric to monitor:**
- **Visual quality:** Manual inspection of mouth crops
- **WER improvement:** Especially for hard cases (profile views, occlusions)

**Ablation design:**
```bash
# Compare with/without quality filtering
python script/lip_crop.py --video ... --landmark_quality_check [true|false]

# Evaluate on full dev set
python script/inference.py --model_type avsr_cocktail \
    --session_dir "data-bin/dev/*" \
    --use_landmark_filtering
```

---

#### **2.3 Temporal Smoothing of Landmarks**

**Problem:** Frame-to-frame landmark jitter causes shaky mouth crops.

**Current:** Window-based smoothing (window_margin=12) already implemented! ✓

**Already in code** (`src/retinaface/video_process.py:93-107`):
```python
smoothed_landmarks = np.mean([
    landmarks[x] for x in range(frame_idx - window_margin, frame_idx + window_margin + 1)
], axis=0)
```

**Potential improvement:** Tune `window_margin` parameter.

**Current default:** 12 frames (±0.48s at 25fps)

**Ablation:**
```python
# Try different smoothing windows
for window_margin in [6, 9, 12, 15, 18]:
    video_process = VideoProcess(window_margin=window_margin)
```

**Metric to monitor:**
- **Visual stability:** Compute inter-frame optical flow variance (should decrease)
- **WER:** May improve if crops are more stable

---

#### **2.4 Handle Missing Faces More Gracefully**

**Problem:** If no face detected in a frame, interpolation may fail at boundaries.

**Current fallback** (`src/retinaface/video_process.py:134-141`):
```python
# Extend first/last valid landmark to edges
landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
```

**Improvement:** Detect excessive missing frames and skip the track entirely.

```python
# In VideoProcess.__call__
valid_ratio = len([l for l in landmarks if l is not None]) / len(landmarks)
if valid_ratio < 0.7:  # If >30% frames missing
    print(f"Warning: {100*(1-valid_ratio):.1f}% frames missing, skipping track")
    return None
```

**Implementation:**
- **File:** `src/retinaface/video_process.py:80-84`
- **Effort:** 5 lines

**Metric to monitor:**
- **Track skip rate:** How many tracks get skipped
- **Overall WER:** Should improve (avoiding bad tracks)

---

## **Component 3: Segmentation & Chunking**

### **Current Implementation** (`src/talking_detector/segmentation.py`)

**Algorithm:** Three-stage hysteresis-based segmentation

**Parameters:**
```python
CENTRAL_ASD_CHUNKING_PARAMETERS = {
    "onset": 1.0,        # Start threshold
    "offset": 0.8,       # End threshold
    "min_duration_on": 1.0,  # Minimum speech segment (seconds)
    "min_duration_off": 0.5, # Gap filling (seconds)
    "max_chunk_size": 10,    # Maximum chunk (seconds) - overridden to 15 in inference
    "min_chunk_size": 1      # Minimum chunk (seconds)
}
```

**Assumptions:**
1. Fixed thresholds work across all speakers/sessions
2. 1-second minimum segment is appropriate
3. 0.5-second gaps should be filled
4. 15-second max chunk (from `inference.py --max_length 15`)

---

### **Low-Hanging Improvements**

#### **3.1 Tune Hysteresis Thresholds**

**Problem:** Thresholds (onset=1.0, offset=0.8) were likely set heuristically, not optimized for MCoRec.

**Current:** Hardcoded in `segmentation.py:4-11`

**Solution:** Grid search over threshold pairs.

```python
# Test threshold combinations
onset_values = [0.5, 0.8, 1.0, 1.2, 1.5]
offset_values = [0.3, 0.5, 0.8, 1.0]

for onset in onset_values:
    for offset in offset_values:
        if offset < onset:  # Maintain hysteresis property
            params = CENTRAL_ASD_CHUNKING_PARAMETERS.copy()
            params['onset'] = onset
            params['offset'] = offset
            segments = segment_by_asd(asd_scores, params)
```

**Implementation:**
- **File:** `script/inference.py` (add CLI arguments)
- **Effort:** 10 lines

```bash
python script/inference.py --model_type avsr_cocktail \
    --session_dir "data-bin/dev/*" \
    --asd_onset 1.2 --asd_offset 0.8
```

**Metric to monitor:**
- **Primary:** Speaker WER (better segmentation → better transcripts)
- **Secondary:** Number of segments per speaker (too many = over-segmentation)

**Ablation design:**
```bash
# Grid search
for onset in 0.8 1.0 1.2 1.5; do
  for offset in 0.5 0.8 1.0; do
    if [ $(echo "$offset < $onset" | bc) -eq 1 ]; then
      python script/inference.py --asd_onset $onset --asd_offset $offset ...
      python script/evaluate.py ...
    fi
  done
done

# Find best (onset, offset) pair by WER
```

---

#### **3.2 Adaptive Min Duration Based on Speaker Activity**

**Problem:** `min_duration_on=1.0s` is fixed, but some speakers have short utterances (backchannels, "yeah", "uh-huh").

**Current:** Drops all segments <1s, potentially losing short but valid speech.

**Solution:** Use adaptive threshold based on overall speaker activity.

```python
def get_adaptive_min_duration(asd_scores, base_duration=1.0):
    """Adjust min duration based on speaker's overall speaking rate"""
    # Estimate total speaking time
    total_active_frames = sum(1 for score in asd_scores.values() if score > 1.0)
    total_frames = len(asd_scores)
    speaking_ratio = total_active_frames / total_frames

    # If speaker is sparse (talks <20% of time), use shorter min_duration
    if speaking_ratio < 0.2:
        return base_duration * 0.5  # 0.5s for sparse speakers
    else:
        return base_duration  # 1.0s for active speakers
```

**Implementation:**
- **File:** `src/talking_detector/segmentation.py:23`
- **Effort:** 15 lines

**Metric to monitor:**
- **Coverage:** Percentage of ground truth speech captured
- **WER:** May improve by including short utterances

---

#### **3.3 Smarter Gap Filling**

**Problem:** Fixed 0.5s gap filling may be too aggressive (merges false gaps) or too conservative (misses true continuations).

**Current:**
```python
gap = next_region[0] - current_region[-1] - 1
if gap <= min_duration_off_frames:  # 12.5 frames (0.5s)
    # Merge regions
```

**Solution:** Context-aware gap filling.

```python
def should_fill_gap(gap_duration, prev_scores, next_scores, threshold=0.5):
    """Decide whether to fill a gap based on surrounding ASD scores"""
    # Don't fill if gap has consistently low scores
    gap_scores = prev_scores[-5:] + next_scores[:5]  # Look at edges
    if np.mean(gap_scores) < 0.5:  # Low activity around gap
        return False

    # Fill if gap is short and surrounded by high activity
    if gap_duration < 0.5 and np.mean(gap_scores) > 1.0:
        return True

    return gap_duration < 0.3  # Default: only very short gaps
```

**Implementation:**
- **File:** `src/talking_detector/segmentation.py:74-82`
- **Effort:** 20 lines

**Metric to monitor:**
- **Segmentation precision:** Fewer false merges
- **WER:** Should improve

---

#### **3.4 Dynamic Chunk Size Optimization**

**Problem:** Fixed `max_length=15s` may not be optimal for all models/GPUs.

**Current:** CLI parameter but not automatically tuned.

**Solution:** Provide guidance based on GPU memory.

```python
def get_optimal_chunk_size(gpu_memory_gb, model_type):
    """Suggest chunk size based on available GPU memory"""
    if model_type == "auto_avsr":
        # Auto-AVSR is heavier
        if gpu_memory_gb >= 24:
            return 15
        elif gpu_memory_gb >= 16:
            return 10
        else:
            return 8
    else:  # avsr_cocktail, muavic_en
        if gpu_memory_gb >= 24:
            return 20  # Can handle longer segments
        elif gpu_memory_gb >= 16:
            return 15
        else:
            return 10
```

**Implementation:**
- **File:** `script/inference.py` (add auto-detection)
- **Effort:** 20 lines

```bash
# Auto-detect optimal chunk size
python script/inference.py --model_type avsr_cocktail \
    --session_dir "data-bin/dev/*" \
    --max_length auto
```

**Metric to monitor:**
- **Inference time:** Longer chunks = fewer forward passes = faster
- **GPU memory usage:** Should stay <90% utilization
- **WER:** Should not change (chunk size doesn't affect model quality, only efficiency)

---

## **Component 4: AVSR Models & Decoding**

### **Current Implementation** (`script/inference.py`)

**Models:**
- BL1/BL4: AV-HuBERT CTC/Attention with beam search (beam_size=3)
- BL2: MuAViC with greedy decoding (no beam search)
- BL3: Auto-AVSR with beam search (beam_size=3)

**Decoding:**
```python
# AV-HuBERT / Auto-AVSR
nbest_hyps = beam_search(audiovisual_feat)
predicted = text_transform.post_process(nbest_hyps[0]["yseq"])

# MuAViC
output = model.generate(audios, video=videos)
output = tokenizer.batch_decode(output)
```

**Assumptions:**
1. Beam size=3 is optimal
2. Top-1 hypothesis is always selected
3. No language model rescoring
4. No test-time augmentation

---

### **Low-Hanging Improvements**

#### **4.1 Tune Beam Size**

**Problem:** `beam_size=3` is default, not optimized.

**Current:** CLI parameter `--beam_size 3`

**Solution:** Ablate beam sizes.

```bash
# Test different beam sizes
for beam in 1 3 5 10; do
    python script/inference.py --model_type avsr_cocktail \
        --session_dir "data-bin/dev/*" \
        --beam_size $beam \
        --output_dir_name output_beam${beam}

    python script/evaluate.py --session_dir "data-bin/dev/*" \
        --output_dir_name output_beam${beam}
done
```

**Expected trade-offs:**
- Larger beam: Better WER (more hypotheses explored) but slower inference
- Beam=1 (greedy): Faster but may miss optimal path

**Metric to monitor:**
- **Speaker WER:** Primary target
- **Inference time:** Track degradation
- **Sweet spot:** Likely beam ∈ {3, 5}

**Effort:** 0 lines (already supported!)

---

#### **4.2 Add Length Penalty to Beam Search**

**Problem:** Beam search may favor shorter hypotheses (lower total score).

**Current:** No length normalization in `beam_search.py`

**Solution:** Add length penalty parameter.

```python
# In beam search scoring
def score_hypothesis(log_prob, length, length_penalty=0.6):
    """
    Score = log_prob / length^length_penalty

    length_penalty:
    - 0.0: No normalization (favors short)
    - 1.0: Full normalization (length-neutral)
    - >1.0: Favors longer hypotheses
    """
    return log_prob / (length ** length_penalty)
```

**Implementation:**
- **File:** `src/nets/beam_search.py` or `src/nets/batch_beam_search.py`
- **Effort:** 10-15 lines

**Metric to monitor:**
- **Speaker WER:** Should improve if current outputs are too short
- **Average transcript length:** Should increase with penalty

**Ablation design:**
```bash
for penalty in 0.0 0.3 0.6 0.8 1.0; do
    python script/inference.py --beam_size 5 --length_penalty $penalty ...
done
```

---

#### **4.3 Ensemble Multiple Models**

**Problem:** BL1 (WER=0.5536) and BL4 (WER=0.4990) may make complementary errors.

**Current:** Models run independently.

**Solution:** Simple late fusion of top-k hypotheses.

```python
def ensemble_transcripts(hyp1, hyp2, weights=[0.5, 0.5]):
    """
    Combine hypotheses from two models using ROVER-style voting
    or simple weighted average of log probabilities
    """
    # Option 1: Select based on confidence
    if hyp1['score'] * weights[0] > hyp2['score'] * weights[1]:
        return hyp1['text']
    else:
        return hyp2['text']

    # Option 2: Word-level voting (more complex)
    # Use ROVER algorithm to align and vote on word sequences
```

**Implementation:**
- **File:** New script `script/inference_ensemble.py`
- **Effort:** 50-100 lines

**Metric to monitor:**
- **Speaker WER:** Should improve (ensemble typically 1-2% better)
- **Inference time:** 2× slower (two forward passes)

**Ablation design:**
```bash
# Run both BL1 and BL4
python script/inference.py --model_type avsr_cocktail \
    --checkpoint_path ./model-bin/avsr_cocktail \
    --output_dir_name output_bl1

python script/inference.py --model_type avsr_cocktail \
    --checkpoint_path ./model-bin/avsr_cocktail_mcorec_finetune \
    --output_dir_name output_bl4

# Ensemble outputs
python script/inference_ensemble.py --inputs output_bl1 output_bl4 \
    --weights 0.4 0.6  # BL4 is better, weight more
```

---

#### **4.4 Test-Time Augmentation (TTA)**

**Problem:** Single-pass inference may miss information due to input variations.

**Solution:** Average predictions over augmented inputs.

**Augmentations:**
- Horizontal flip of mouth region (controversial, but worth testing)
- Slight temporal shifts (±1 frame offset)
- Audio pitch/speed perturbations

```python
def inference_with_tta(video, audio, model, augmentations=['hflip', 'shift']):
    predictions = []

    # Original
    pred = model.inference(video, audio)
    predictions.append(pred)

    # Horizontal flip
    if 'hflip' in augmentations:
        video_flipped = torch.flip(video, dims=[-1])  # Flip width
        pred_flip = model.inference(video_flipped, audio)
        predictions.append(pred_flip)

    # Temporal shift
    if 'shift' in augmentations:
        video_shift = video[:, 1:]  # Shift by 1 frame
        pred_shift = model.inference(video_shift, audio[:, :-640])  # Match audio
        predictions.append(pred_shift)

    # Vote or average log-probs
    return majority_vote(predictions)
```

**Implementation:**
- **File:** `script/inference.py` (add `--use_tta` flag)
- **Effort:** 30-40 lines

**Metric to monitor:**
- **Speaker WER:** Target 0.5-1% improvement
- **Inference time:** 2-3× slower

**Ablation design:**
```bash
python script/inference.py --model_type avsr_cocktail \
    --session_dir "data-bin/dev/*" \
    --use_tta --tta_augmentations hflip,shift
```

---

#### **4.5 Adaptive Decoding Based on Segment Properties**

**Problem:** All segments decoded with same beam size, regardless of difficulty.

**Solution:** Use larger beam for longer/noisier segments.

```python
def get_adaptive_beam_size(segment_duration, base_beam=3, max_beam=10):
    """Increase beam size for longer segments"""
    if segment_duration < 3:
        return base_beam  # Short segment: beam=3
    elif segment_duration < 8:
        return base_beam + 1  # Medium: beam=4
    else:
        return base_beam + 2  # Long: beam=5
```

**Implementation:**
- **File:** `script/inference.py:279`
- **Effort:** 10 lines

**Metric to monitor:**
- **WER on long vs. short segments:** Should improve for long segments
- **Overall WER:** Target 0.3-0.5% improvement
- **Average inference time:** Minimal increase (most segments are short)

---

## **Component 5: Time-Based Conversation Clustering**

### **Current Implementation** (`src/cluster/conv_spks.py`)

**Algorithm:**
```python
score(i,j) = 1 - (overlap_duration / total_duration)
distance(i,j) = 1 - score(i,j)
clusters = AgglomerativeClustering(
    distance_threshold=0.3,  # score_threshold=0.7
    linkage='complete'
).fit(distances)
```

**Assumptions:**
1. Overlap indicates different conversations
2. Threshold=0.7 works for all sessions
3. Complete linkage is optimal
4. Temporal features alone suffice (no visual/acoustic)

---

### **Low-Hanging Improvements**

#### **5.1 Tune Clustering Threshold**

**Problem:** `threshold=0.7` is arbitrary, not optimized.

**Current:** Default parameter in `cluster_speakers()` (`conv_spks.py:119`)

**Solution:** Grid search over thresholds.

```bash
for thresh in 0.5 0.6 0.65 0.7 0.75 0.8 0.85; do
    python script/inference.py --model_type avsr_cocktail \
        --session_dir "data-bin/dev/*" \
        --clustering_threshold $thresh \
        --output_dir_name output_thresh${thresh}

    python script/evaluate.py --session_dir "data-bin/dev/*" \
        --output_dir_name output_thresh${thresh}
done

# Find threshold with best Conversation Clustering F1
```

**Implementation:**
- **File:** `script/inference.py` (add CLI argument)
- **Effort:** 5 lines

**Metric to monitor:**
- **Conversation Clustering F1:** Primary target (currently 0.8153)
- **Joint Error Rate:** Overall impact

**Expected result:** Threshold ∈ [0.65, 0.75] likely optimal (current is in middle of range)

---

#### **5.2 Experiment with Linkage Methods**

**Problem:** Complete linkage may be suboptimal for this task.

**Current:** `linkage='complete'`

**Alternatives:**
- **Average linkage:** Distance = average of all pairwise distances
- **Single linkage:** Distance = minimum pairwise distance (prone to chaining)
- **Ward linkage:** Minimizes within-cluster variance (not compatible with precomputed)

```python
for linkage in ['complete', 'average', 'single']:
    clustering = AgglomerativeClustering(
        distance_threshold=0.3,
        metric='precomputed',
        linkage=linkage
    )
```

**Implementation:**
- **File:** `src/cluster/conv_spks.py:147`
- **Effort:** 3 lines + CLI argument

**Metric to monitor:**
- **Conversation Clustering F1**
- **Cluster size distribution:** Average linkage may produce more balanced clusters

**Ablation design:**
```bash
for linkage in complete average single; do
    python script/inference.py --clustering_linkage $linkage ...
done
```

**Expected:** Average linkage may perform better (less sensitive to outliers than complete)

---

#### **5.3 Normalize Overlap Score by Speaker Activity**

**Problem:** Sparse speakers (talk infrequently) have unstable overlap ratios.

**Current:**
```python
score = 1 - (overlap / (overlap + non_overlap))
```

**Issue:** If speaker A talks 60s and speaker B talks 5s, overlap ratio is sensitive to B's short duration.

**Solution:** Weighted normalization.

```python
def calculate_weighted_score(overlap, non_overlap, duration_i, duration_j, alpha=0.5):
    """
    Combine overlap-based score with duration-based confidence weighting

    alpha: Weight for raw overlap score
    (1-alpha): Weight for duration-adjusted confidence
    """
    # Raw overlap score
    overlap_score = 1 - (overlap / (overlap + non_overlap + 1e-8))

    # Confidence based on total speaking time
    min_duration = min(duration_i, duration_j)
    confidence = min(min_duration / 10.0, 1.0)  # Confident if speaker talks >10s

    # Weighted combination
    final_score = alpha * overlap_score + (1 - alpha) * confidence * overlap_score

    return final_score
```

**Implementation:**
- **File:** `src/cluster/conv_spks.py:103-108`
- **Effort:** 15 lines

**Metric to monitor:**
- **Per-speaker Clustering F1:** Should improve for sparse speakers
- **Overall Conversation Clustering F1**

---

#### **5.4 Add Minimum Speaking Time Requirement**

**Problem:** Speakers who barely talk (e.g., <3 seconds total) are hard to cluster accurately.

**Solution:** Mark low-activity speakers as "unassigned" or use default assignment.

```python
def get_speaker_activity_segments(pycrop_asd_path, uem_start, uem_end):
    segments = ...  # Current implementation

    # Check total speaking time
    total_duration = sum(end - start for start, end in segments)

    if total_duration < 3.0:  # Less than 3 seconds
        print(f"Warning: Speaker has only {total_duration:.1f}s speech, unreliable clustering")
        return []  # Signal to handle specially

    return segments
```

**Implementation:**
- **File:** `src/cluster/conv_spks.py:168-209`
- **Effort:** 10 lines

**Metric to monitor:**
- **Clustering F1 for active speakers:** Should improve
- **Overall F1:** May slightly decrease if unassigned speakers penalized

**Alternative:** Assign low-activity speakers to singleton clusters or use modal cluster assignment.

---

#### **5.5 Post-Processing: Merge Singleton Clusters**

**Problem:** Time-based clustering may create singleton clusters (1 speaker) when that speaker actually belongs to a conversation.

**Current:** No post-processing of cluster assignments.

**Solution:** Heuristic merge of singletons.

```python
def merge_singletons(clusters, scores, speaker_ids):
    """
    Merge singleton clusters into nearest multi-speaker cluster
    """
    cluster_sizes = {}
    for spk, cluster_id in clusters.items():
        cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1

    # Find singletons
    singletons = [spk for spk, cid in clusters.items() if cluster_sizes[cid] == 1]

    for singleton in singletons:
        singleton_idx = speaker_ids.index(singleton)

        # Find best multi-speaker cluster to merge into
        best_cluster = None
        best_score = -1

        for other_spk, other_cluster in clusters.items():
            if other_spk != singleton and cluster_sizes[other_cluster] > 1:
                other_idx = speaker_ids.index(other_spk)
                score = scores[singleton_idx, other_idx]

                if score > best_score:
                    best_score = score
                    best_cluster = other_cluster

        # Merge if score is reasonable (>0.5)
        if best_score > 0.5:
            clusters[singleton] = best_cluster

    return clusters
```

**Implementation:**
- **File:** `src/cluster/conv_spks.py:166` (after clustering)
- **Effort:** 30 lines

**Metric to monitor:**
- **Conversation Clustering F1:** May improve by reducing false singletons
- **Number of clusters:** Should decrease slightly

---

#### **5.6 Use Median Instead of Mean for Activity Extraction**

**Problem:** Outlier ASD scores may distort overlap calculations.

**Current:** `segment_by_asd()` uses threshold-based detection (robust ✓)

**Potential improvement:** When computing overlap, use median-filtered scores for stability.

```python
# In calculate_overlap_duration, before computing overlaps
from scipy.ndimage import median_filter

# Apply median filter to segment boundaries
filtered_segments1 = median_filter_segments(segments1, window=3)
filtered_segments2 = median_filter_segments(segments2, window=3)
```

**Implementation:**
- **File:** `src/cluster/conv_spks.py:43-74`
- **Effort:** 15 lines

**Metric to monitor:**
- **Conversation Clustering F1**
- **Robustness to ASD noise:** Test on sessions with noisy ASD

---

## **Cross-Component Improvements**

### **CC.1 End-to-End Hyperparameter Optimization**

**Problem:** Each component has parameters tuned independently, not jointly.

**Solution:** Use Bayesian optimization to tune critical parameters together.

**Parameters to optimize:**
```python
search_space = {
    # ASD
    'asd_smooth_sigma': [1, 2, 3, 5, 7],
    # Segmentation
    'asd_onset': [0.8, 1.0, 1.2, 1.5],
    'asd_offset': [0.5, 0.8, 1.0],
    'min_duration_on': [0.5, 0.7, 1.0, 1.5],
    # AVSR
    'beam_size': [1, 3, 5, 10],
    # Clustering
    'clustering_threshold': [0.5, 0.6, 0.7, 0.8],
    'clustering_linkage': ['complete', 'average']
}
```

**Tool:** Use Optuna or scikit-optimize

```python
import optuna

def objective(trial):
    # Sample parameters
    asd_onset = trial.suggest_float('asd_onset', 0.8, 1.5)
    beam_size = trial.suggest_int('beam_size', 1, 10)
    clustering_threshold = trial.suggest_float('clustering_threshold', 0.5, 0.8)

    # Run inference with these parameters
    run_inference_with_params(asd_onset, beam_size, clustering_threshold)

    # Evaluate
    metrics = run_evaluation()

    # Optimize joint error rate
    return metrics['joint_error_rate']

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print(f"Best params: {study.best_params}")
print(f"Best joint error: {study.best_value}")
```

**Implementation:**
- **File:** New script `script/optimize_hyperparameters.py`
- **Effort:** 100-150 lines

**Metric to monitor:**
- **Joint ASR-Clustering Error Rate:** Primary optimization target
- **Individual metrics:** WER, Clustering F1

**Computational cost:** High (50-100 trials × 2 hours/trial = 100-200 GPU-hours)

**Recommendation:** Start with grid search on most critical parameters, then refine with Bayesian optimization.

---

### **CC.2 Multi-Stage Pipeline with Feedback**

**Problem:** Pipeline is strictly feedforward (ASD → Segmentation → AVSR → Clustering). Errors in early stages propagate.

**Solution:** Add feedback loop.

**Example: Confidence-based ASD re-scoring**

```python
# Stage 1: Initial AVSR on all segments
initial_transcripts = avsr_inference(all_segments)

# Stage 2: Identify low-confidence segments
low_confidence = [seg for seg in initial_transcripts if seg['confidence'] < 0.3]

# Stage 3: Re-segment low-confidence regions with relaxed thresholds
for seg in low_confidence:
    relaxed_params = {
        'onset': 0.8,  # Lower from 1.0
        'offset': 0.6   # Lower from 0.8
    }
    new_segments = segment_by_asd(seg['asd_scores'], relaxed_params)

    # Re-run AVSR on new segments
    refined_transcripts = avsr_inference(new_segments)
```

**Implementation:**
- **File:** New script `script/inference_iterative.py`
- **Effort:** 200+ lines

**Metric to monitor:**
- **Speaker WER:** Should improve
- **Inference time:** 1.5-2× slower (partial re-processing)

**Recommendation:** Implement as separate "refined baseline" after optimizing individual components.

---

## **Implementation Priority & Effort Matrix**

| Improvement | Component | Effort | Expected Gain | Priority |
|-------------|-----------|--------|---------------|----------|
| **ASD temporal smoothing** | 1 | Low (5 lines) | Medium (+1-2% F1) | **HIGH** |
| **Tune clustering threshold** | 5 | Low (5 lines) | High (+2-3% F1) | **HIGH** |
| **Tune beam size** | 4 | None (0 lines) | Medium (+0.5-1% WER) | **HIGH** |
| **Tune segmentation thresholds** | 3 | Low (10 lines) | Medium (+0.5-1% WER) | **HIGH** |
| **ASD score normalization** | 1 | Low (5 lines) | Low-Med (+0.5-1% F1) | MEDIUM |
| **Clustering linkage experiment** | 5 | Low (3 lines) | Low-Med (+1% F1) | MEDIUM |
| **Landmark quality filtering** | 2 | Medium (20 lines) | Low (+0.3-0.5% WER) | MEDIUM |
| **Merge singleton clusters** | 5 | Medium (30 lines) | Medium (+1-2% F1) | MEDIUM |
| **Add length penalty** | 4 | Medium (15 lines) | Low-Med (+0.5% WER) | MEDIUM |
| **Dynamic chunk size** | 3 | Medium (20 lines) | None (efficiency only) | LOW |
| **Model ensemble** | 4 | High (50 lines) | High (+1-2% WER) | MEDIUM |
| **Test-time augmentation** | 4 | Medium (40 lines) | Low-Med (+0.5-1% WER) | LOW |
| **End-to-end optimization** | All | Very High (150 lines) | High (+2-3% joint error) | MEDIUM |

---

## **Recommended Evaluation Protocol**

### **Quick Ablation (Single Session)**

```bash
# Test on one session for rapid iteration
SESSION="data-bin/dev/session_132"

# Baseline
python script/inference.py --model_type avsr_cocktail --session_dir $SESSION

# Test improvement
python script/inference.py --model_type avsr_cocktail --session_dir $SESSION \
    --asd_smooth_sigma 3 --asd_onset 1.2 --beam_size 5

# Compare
python script/evaluate.py --session_dir $SESSION
```

---

### **Full Dev Set Evaluation**

```bash
# Run on all dev sessions
python script/inference.py --model_type avsr_cocktail \
    --session_dir "data-bin/dev/*" \
    --asd_smooth_sigma 3 \
    --asd_onset 1.2 --asd_offset 0.8 \
    --beam_size 5 \
    --clustering_threshold 0.7 \
    --output_dir_name output_improved

# Evaluate
python script/evaluate.py --session_dir "data-bin/dev/*" \
    --output_dir_name output_improved
```

---

### **Metric Tracking Template**

Create a CSV to track all experiments:

```csv
experiment_id,asd_smooth,asd_onset,asd_offset,beam_size,cluster_thresh,cluster_linkage,conv_f1,speaker_wer,joint_error
baseline,0,1.0,0.8,3,0.7,complete,0.8153,0.5536,0.3821
exp001,3,1.0,0.8,3,0.7,complete,0.8201,0.5489,0.3755
exp002,3,1.2,0.8,3,0.7,complete,0.8234,0.5412,0.3694
exp003,3,1.2,0.8,5,0.7,complete,0.8234,0.5301,0.3583
exp004,3,1.2,0.8,5,0.75,complete,0.8312,0.5301,0.3545
exp005,3,1.2,0.8,5,0.75,average,0.8401,0.5301,0.3450
```

**Analysis script:**

```python
import pandas as pd
df = pd.read_csv('experiments.csv')

# Find best by joint error
best_joint = df.loc[df['joint_error'].idxmin()]
print(f"Best joint error: {best_joint['experiment_id']} = {best_joint['joint_error']}")

# Find best by clustering
best_cluster = df.loc[df['conv_f1'].idxmax()]
print(f"Best clustering: {best_cluster['experiment_id']} = {best_cluster['conv_f1']}")

# Find best by WER
best_wer = df.loc[df['speaker_wer'].idxmin()]
print(f"Best WER: {best_wer['experiment_id']} = {best_wer['speaker_wer']}")
```

---

## **Expected Overall Improvement**

Combining **top 5 high-priority improvements**:

| Metric | Baseline | Improved | Gain |
|--------|----------|----------|------|
| **Conversation Clustering F1** | 0.8153 | ~0.84-0.85 | +2-3% |
| **Speaker WER (BL1)** | 0.5536 | ~0.53-0.54 | -2-3% |
| **Joint Error Rate (BL1)** | 0.3821 | ~0.36-0.37 | -2-3% |

**Conservative estimate:** 2-3% absolute improvement in joint error rate

**Optimistic estimate:** 4-5% with end-to-end optimization

---

## **Summary: Quick Wins**

**Week 1: Zero-code improvements**
1. Tune beam size (try 5, 7, 10)
2. Tune clustering threshold (try 0.65, 0.75, 0.8)
3. Document baseline variance (run 3 times, report std dev)

**Week 2: Low-effort coding (5-20 lines each)**
4. Add ASD temporal smoothing
5. Add ASD score normalization
6. Tune segmentation thresholds (onset, offset)
7. Try average linkage for clustering

**Week 3: Medium-effort improvements (20-50 lines each)**
8. Implement landmark quality filtering
9. Add length penalty to beam search
10. Merge singleton clusters post-processing

**Week 4: Integration & evaluation**
11. Combine best improvements
12. Run full dev set evaluation
13. Prepare ablation table for paper

**Expected result:** New baseline with **3-5% better joint error rate**, publishable as improved baseline.

---

## **Document Information**

- **Generated**: 2025-11-18
- **Repository**: https://github.com/aziz-hakiri/mcorec_baseline
- **Analysis Type**: Low-Hanging Fruit Improvements
- **Target**: Practical, incremental baseline improvements within existing framework
