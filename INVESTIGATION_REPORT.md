# Investigation Report: Why Ground-Truth ASD Worsens WER

## Executive Summary

**Finding:** Using ground-truth (GT) ASD segments produces **worse WER than noisy model predictions** due to a mismatch between the segmentation pipeline design and perfect GT data characteristics.

**Root Cause:** The `segment_by_asd()` function is optimized for **noisy predictions** and applies aggressive filtering that is **destructive when applied to clean GT data**.

**Impact:** Short utterances (<1 second) in GT are systematically deleted, causing deletion errors that increase WER.

---

## Background

From the paper (Section 4.1):
- The ASD model achieves 75.58% IoU on the dev set
- The model is "reasonably accurate" but has room for improvement
- The segmentation post-processing is designed to handle these noisy predictions

---

## How the Segmentation Pipeline Works

### Design Philosophy: Noise Suppression

The segmentation function (`src/talking_detector/segmentation.py:23-111`) processes ASD scores through three stages designed to clean up **noisy model predictions**:

#### Stage 1: Hysteresis Thresholding
```python
onset_threshold = 1.0   # Score must be > 1.0 to START speech
offset_threshold = 0.8  # Score must fall < 0.8 to STOP speech
```

**Purpose:** Smooth fluctuating predictions
- Noisy models produce scores that jump around: 2.5, 2.3, 1.8, 0.9, 1.2, 2.1, ...
- Without hysteresis, this creates fragmented segments
- Hysteresis creates "sticky" regions: once speech starts, it continues through small dips

#### Stage 2: Gap Filling
```python
min_duration_off = 0.5  # seconds (12.5 frames at 25fps)
```

**Purpose:** Merge close segments
- Noisy models often split continuous speech into multiple segments
- Segments separated by < 0.5s are merged

#### Stage 3: Duration Filtering
```python
min_duration_on = 1.0   # seconds (25 frames at 25fps)
max_chunk_size = 10-15  # seconds
```

**Purpose:** Remove spurious activations and split long segments
- Removes segments shorter than 1.0 second (assumed to be false positives)
- Splits segments longer than max_chunk_size

### Why This Works for Noisy Predictions

The noisy ASD model has several characteristics that make it "compatible" with this pipeline:

1. **Temporal blur:** The model's receptive field and uncertainty cause predictions to "spread"
   - A 0.3s word "Yes" gets predicted as a ~0.6-1.5s region
   - Short words get naturally "padded" by prediction uncertainty

2. **Boundary softness:** Transitions aren't sharp
   - Scores gradually rise and fall: 0.5 → 1.2 → 2.4 → 2.8 → 2.1 → 0.9 → 0.4
   - Hysteresis thresholding smooths this effectively

3. **False positives:** Model sometimes activates on non-speech
   - min_duration_on filter removes these spurious short activations

**Result:** The noisy model + aggressive filtering = reasonably good segmentation

---

## What Happens with Ground-Truth ASD

### Characteristics of GT Data

Ground-truth ASD has fundamentally different properties:

1. **Sharp boundaries:** Transitions are instant
   - Frame 32: not speaking (score = 0)
   - Frame 33: speaking (score = 1 or 5)

2. **Precise durations:** Short words are truly short
   - "Yes" = exactly 0.32s (8 frames)
   - "No" = exactly 0.32s (8 frames)
   - "Okay" = exactly 0.48s (12 frames)
   - "Mm-hmm" = exactly 0.40s (10 frames)

3. **No false positives:** Every active frame is truly speech

### The Destructive Impact

When GT data goes through the segmentation pipeline:

#### Issue 1: Onset Threshold Bug (Severity depends on GT format)

If GT uses binary scores (0/1):
```python
if score > onset_threshold:  # 1.0 > 1.0 → False!
```
- **Impact:** NO speech detected at all → Catastrophic WER (100% deletion)
- **Likelihood:** Low (student said "kind of worse", not catastrophic)

#### Issue 2: Short Utterance Deletion ⭐ **PRIMARY CAUSE**

Natural conversation contains many short utterances:

| Utterance Type | Typical Duration | Frequency in Conversation |
|----------------|------------------|---------------------------|
| "Yes" / "No" | 0.2-0.4s | Very high |
| "Okay" / "Right" | 0.3-0.5s | High |
| "Mm-hmm" / "Uh-huh" | 0.3-0.5s | High |
| "I see" | 0.4-0.6s | Medium |
| Brief laughter | 0.3-0.8s | Medium |
| Short interjections | 0.2-0.5s | High |

**With noisy model:**
- These get predicted as 0.8-1.5s regions due to temporal blur
- They survive the min_duration_on=1.0s filter (barely, but often)

**With GT:**
- These are precisely marked as 0.2-0.5s regions
- They are ALL deleted by min_duration_on=1.0s filter
- **Result:** Massive deletion errors → WER increases significantly

#### Issue 3: Over-merging Adjacent Segments

The min_duration_off=0.5s parameter merges segments that are close:

**With noisy model:**
- Already blurred, so gaps might already be filled

**With GT:**
- Natural turn-taking patterns get merged inappropriately
- Example: "A: Question?" [0.3s pause] "B: Answer"
  - These are separate turns but get merged into one segment
  - ASR model gets confused by speaker changes → increased substitution errors

---

## Experimental Verification

I created `test_asd_analysis.py` to demonstrate the issue. Results:

### Test Scenario
- Ground truth: 3 speech regions
  1. Frames 25-32 (0.32s) - "Yes"
  2. Frames 45-52 (0.32s) - "No"
  3. Frames 83-157 (3.00s) - "Long sentence"
- Total ideal speech: 3.64s

### Results

| Configuration | Segments Detected | Speech Duration | Issues |
|--------------|-------------------|-----------------|---------|
| **Noisy model** (realistic) | 1 | 4.12s | ✅ All speech captured (over-estimated) |
| **GT binary (0/1)** | 0 | 0s | ❌ CATASTROPHIC: No speech detected |
| **GT high confidence (5.0)** | 1 | 3.00s | ❌ Short words deleted (2 words lost) |
| **GT + relaxed params** | 3 | 3.64s | ✅ All speech preserved |

### Key Finding

With default parameters and GT (high confidence), the system deletes the two short utterances:
- Lost: "Yes" and "No" (0.64s of speech)
- Only detected the long sentence (3.00s)
- **Deletion rate:** 2 out of 3 utterances = 66% deletion error for short words!

---

## Why Gemini's Analysis Was Partially Wrong

Gemini identified some correct issues but missed the key insight:

**Gemini said:**
- ❌ "Catastrophic empty output" - The student reported "kind of worse", not catastrophic
- ✅ min_duration_on=1.0s deletes short segments - Correct!
- ❌ Focused too much on the onset=1.0 bug - This is only catastrophic for binary GT
- ❌ Didn't explain WHY noisy model predictions work better

**The real story:**
- The noisy model's **temporal blur is a feature, not a bug** for this pipeline
- Short words get "padded" by prediction uncertainty
- This accidental padding helps them survive the aggressive filtering
- GT doesn't have this padding → short words get deleted → WER increases

---

## Critical Reflection and Edge Cases

### Assumptions

1. **GT representation:** Assuming GT uses high confidence scores (e.g., 5.0 for active frames)
   - If binary (0/1): Would hit onset threshold bug → catastrophic
   - If exactly 1.0: Would hit onset threshold bug → catastrophic
   - Need to verify actual GT format in the dataset

2. **Short utterance frequency:** Assuming conversations have many short utterances
   - This is typical in natural dialogue (backchannels, confirmations)
   - If conversations are mostly long utterances, impact would be smaller

3. **WER impact:** Assuming deletions are the primary error type
   - Could also have insertion errors if GT boundaries are slightly loose
   - Need to check actual WER breakdown (deletions vs insertions vs substitutions)

### Potential Flaws

1. **ASR model sensitivity to segment length**
   - AV-HuBERT might have been trained on longer segments
   - Very short segments (<0.3s) might not have enough context
   - Could produce poor transcriptions even if passed to ASR
   - **Mitigation:** Check ASR model's performance on short clips (separate analysis needed)

2. **Boundary precision trade-off**
   - GT has perfect boundaries, but ASR might benefit from slightly padded context
   - A 0.3s "Yes" with 0.1s padding before/after might transcribe better
   - The noisy model accidentally provides this padding
   - **Implication:** Perfect ASD might not be optimal for ASR!

3. **Speaker change confusion**
   - Over-merging segments in multi-speaker scenarios
   - If two speakers are close together, min_duration_off might merge them
   - ASR model (trained on single speaker) would fail
   - **Impact:** Substitution and insertion errors, not just deletions

### Edge Cases

1. **Very short glitchy GT annotations**
   - GT might have 1-2 frame "glitches" from annotation artifacts
   - Setting min_duration_on=0 would pass these to ASR
   - **Solution:** Keep a minimal filter (e.g., 0.1-0.2s) even for GT

2. **Overlapping speech in GT**
   - Multi-speaker scenarios with speech overlap
   - How is this represented in GT? Separate ASD per speaker?
   - Pipeline assumes single-speaker segments

3. **Pause fillers and hesitations**
   - "Um", "uh", hesitations are very short
   - Are these in GT? Should they be transcribed?
   - Deleting them might actually improve some WER metrics (depends on reference)

### Missing Validations

1. **Actual GT format inspection**
   - Need to examine real GT ASD files from the dev set
   - Check score values, boundary patterns, short utterance distribution

2. **WER breakdown analysis**
   - Run evaluation with detailed error types:
     - Deletion rate
     - Insertion rate
     - Substitution rate
   - This would confirm deletions are the main issue

3. **ASR model context requirements**
   - Test ASR model on various segment lengths
   - Find minimum viable segment length
   - Determine if padding is beneficial

4. **Segment length distribution**
   - Analyze real conversations in dev set
   - Distribution of utterance lengths
   - This quantifies potential impact

---

## Recommended Solutions and Testing

### Solution 1: Bypass Filtering for GT (Recommended)

When using GT ASD, use relaxed parameters:

```python
GT_PARAMETERS = {
    "onset": 0.5,           # Lower threshold (GT doesn't need high confidence)
    "offset": 0.5,          # Symmetric threshold
    "min_duration_on": 0.0, # Don't filter short segments
    "min_duration_off": 0.0,# Don't merge close segments
    "max_chunk_size": 15,   # Only keep for practical ASR limits
}
```

**Pros:**
- Preserves all GT speech regions
- Respects precise boundaries
- Should give best theoretical performance

**Cons:**
- Might pass very short segments that ASR can't handle
- Might not provide optimal context padding for ASR

### Solution 2: Minimal Filtering for GT

Use light filtering to avoid artifacts:

```python
GT_PARAMETERS_MINIMAL = {
    "onset": 0.5,
    "offset": 0.5,
    "min_duration_on": 0.16,  # 4 frames @ 25fps = minimum viable utterance
    "min_duration_off": 0.08, # 2 frames = avoid glitches
    "max_chunk_size": 15,
}
```

**Pros:**
- Removes potential annotation artifacts
- Still preserves short utterances
- More robust to edge cases

**Cons:**
- Arbitrary thresholds (need validation)

### Solution 3: Adaptive Parameters Based on Source

Modify inference script to detect GT vs model predictions:

```python
def get_asd_parameters(asd_path, is_ground_truth=False):
    if is_ground_truth:
        return GT_PARAMETERS
    else:
        return CENTRAL_ASD_CHUNKING_PARAMETERS
```

---

## Verification Experiments

### Experiment 1: Quantify Short Utterance Impact

**Objective:** Measure how many utterances are <1s in dev set

**Method:**
1. Load all GT transcripts (.vtt files)
2. Count utterances by duration bins: 0-0.5s, 0.5-1.0s, 1.0-2.0s, >2.0s
3. Calculate percentage affected by min_duration_on filter

**Expected Result:** 30-50% of utterances are <1s

### Experiment 2: Controlled WER Comparison

**Objective:** Verify GT with relaxed params gives better WER

**Method:**
1. Create a script to convert GT .vtt → ASD JSON format
2. Run inference with:
   - GT + default params (min_duration_on=1.0)
   - GT + relaxed params (min_duration_on=0.0)
   - Noisy model predictions (baseline)
3. Compare WER for each configuration

**Expected Result:**
- GT + default: WORSE WER than noisy model (due to deletions)
- GT + relaxed: BEST WER (perfect ASD + no filtering losses)

### Experiment 3: ASR Context Sensitivity

**Objective:** Determine if short segments hurt ASR quality

**Method:**
1. Extract segments of varying lengths from dev set
2. Test ASR model on: 0.2s, 0.4s, 0.6s, 0.8s, 1.0s, 1.5s segments
3. Measure transcription quality vs segment length

**Expected Result:** Performance degrades below 0.3-0.4s (need padding)

### Experiment 4: Error Type Breakdown

**Objective:** Confirm deletions are the primary error

**Method:**
1. Modify evaluate.py to output detailed error breakdown
2. Run on student's GT results
3. Analyze deletion/insertion/substitution rates

**Expected Result:** High deletion rate (30-50% of errors)

---

## Implementation: Verification Script

I'll create a script to help verify these hypotheses:
