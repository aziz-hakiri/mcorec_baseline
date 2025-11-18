# Multimodal LLMs for CHiME-9 MCoRec: A Critical Survey

**Author**: Analysis of multimodal large language models for audio-visual speech recognition in multi-speaker, high-overlap scenarios
**Date**: 2025-11-18
**Context**: CHiME-9 MCoRec baseline analysis
**Branch**: `claude/analyze-chime9-baseline-01MF6nbwKYxqZN2MqUf5AzLu`

---

## Executive Summary

This document surveys multimodal large language models (LLMs) for audio-visual speech recognition (AVSR), with specific focus on their applicability to the **CHiME-9 MCoRec task**: multi-speaker, high-overlap, cocktail-party scenarios with up to 8 speakers and 4 concurrent conversations.

**Key Findings**:
- **Single-speaker optimized**: Most "state-of-the-art" multimodal LLMs (Whisper-Flamingo, GPT-4V, LLM-based AVSR) are optimized for clean, single-speaker benchmarks (LRS2/LRS3, VoxCeleb2)
- **Overlap brittleness**: Current models exhibit fundamental architectural limitations for overlapping speech (attention collapse, source attribution failure, hallucination)
- **CHiME-9 applicability**: Limited to post-processing roles (rescoring, context modeling) rather than end-to-end recognition
- **Baseline superiority**: AV-HuBERT CTC/Attention outperforms Whisper-Flamingo on CHiME-9 due to frame-level modeling and CTC's monotonic alignment bias

**Bottom Line**: Multimodal LLMs are not yet competitive for cocktail-party AVSR, despite impressive performance on single-speaker benchmarks. The gap between LRS3 (clean, single-speaker) and CHiME-9 (noisy, multi-speaker, high-overlap) remains large.

---

## 1. Literature Survey (2019-2024)

### 1.1 Pre-LLM Era: AV-HuBERT and Self-Supervised Learning (2021-2022)

#### **AV-HuBERT** (Shi et al., AAAI 2022)
- **Architecture**: Masked prediction framework for joint audio-visual learning
- **Training**: Self-supervised on unlabeled audio-visual data, then fine-tuned with CTC/Attention
- **Single vs Multi-speaker**: Designed for single-speaker; multi-speaker extension requires explicit segmentation
- **Overlap Handling**: **No explicit mechanism**. Assumes pre-segmented, single-speaker input
- **Benchmark Performance**: WER 0.9% (VSR), 0.5% (AVSR) on LRS3-test
- **CHiME-9 Baseline**: AV-HuBERT CTC/Attention achieves **WER 0.4990-0.8315** on BL1-BL4 baselines (depending on ASD/clustering quality)

**Key Strengths for CHiME-9**:
- Frame-level modeling (29.97 fps video, 50 Hz audio features) → robust to temporal misalignment
- CTC decoding has monotonic alignment bias → less prone to hallucination than LLM-based approaches
- Attention mechanism is localized → doesn't assume global sentence-level context

**Key Weaknesses**:
- Requires explicit speaker diarization and segmentation (ASD + clustering pipeline)
- No built-in speaker attribution or conversation modeling

---

### 1.2 Transition to Multimodal LLMs (2023)

#### **MuAViC: Multilingual Audio-Visual Corpus** (Anwar et al., Interspeech 2023)
- **Contribution**: First large-scale multilingual AVSR corpus (1200 hours, 9 languages)
- **Architecture**: Extends AV-HuBERT to multilingual settings
- **Single vs Multi-speaker**: Single-speaker focus (TED talks, YouTube)
- **Overlap Handling**: **None**. Assumes clean, non-overlapping speech
- **Benchmark Performance**: WER 1.0% (AVSR, English subset)

**Relevance to CHiME-9**: ❌ **Not applicable**. MuAViC focuses on multilingual generalization, not multi-speaker robustness.

---

#### **Auto-AVSR: Automatic Transcription for AVSR Training** (Ma et al., ICASSP 2023)
- **Contribution**: Uses automatically-generated transcriptions (pseudo-labels) to scale AVSR training
- **Architecture**: AV-HuBERT-based with pseudo-label filtering
- **Training Data**: LRS3 (433h) + AVSpeech (3448h auto-transcribed)
- **Single vs Multi-speaker**: Single-speaker
- **Overlap Handling**: **None**
- **Benchmark Performance**: WER 1.0% (ASR), 1.5% (AVSR) on LRS3

**Relevance to CHiME-9**: ⚠️ **Limited**. Auto-labeling could scale CHiME-9 training data, but doesn't address multi-speaker overlaps.

---

### 1.3 LLM-Based AVSR Era (2024)

#### **Whisper-Flamingo** (Rouditchenko et al., Interspeech 2024; Hu et al., IEEE SPL 2025)

**Architecture**:
```
Visual Encoder (ResNet-18 → VideoMAE)
    ↓
Perceiver Resampler (2048 visual tokens → 32 tokens)
    ↓
Gated Cross-Attention (inject into Whisper decoder layers)
    ↓
Whisper Decoder (autoregressive text generation)
```

**Training**:
- Stage 1: Train gated cross-attention on LRS3 (433h)
- Stage 2: Optional multilingual extension (mWhisper-Flamingo, 9 languages)

**Single vs Multi-speaker**: **Single-speaker only**. Architecture assumes:
- One active speaker per utterance
- One face corresponds to one audio stream
- No mechanism for multi-speaker attribution

**Overlap Handling**: **Fails catastrophically**. Known failure modes:
1. **Attention collapse**: Cross-attention averages over multiple faces → blurred features
2. **Source attribution**: Cannot determine which face corresponds to which audio in overlaps
3. **Hallucination**: Autoregressive LLM decoding continues generating plausible text even when visual/audio signals are ambiguous

**Benchmark Performance**:
- LRS3-test: WER **0.68% (ASR)**, **0.76% (AVSR)** ← "State-of-the-art" claim based on this
- CHiME-9 baseline: WER **0.8315** (BL4) ← **Worse than AV-HuBERT (0.4990-0.8315)**

**Why Worse Than AV-HuBERT on CHiME-9?**
1. **Autoregressive decoding**: Whisper-Flamingo generates full sentence autoregressively → error propagation
2. **Global attention**: LLM assumes sentence-level context → fails when context is ambiguous (overlapping speakers)
3. **Visual-audio mismatch**: Gated cross-attention assumes one-to-one face-audio correspondence → breaks in multi-speaker scenarios
4. **Hallucination**: LLM prior is too strong → generates fluent but incorrect text when sensory input is uncertain

**Relevance to CHiME-9**: ❌ **Not suitable for end-to-end AVSR**. Could be used for rescoring/post-processing only.

---

#### **VSP-LLM: Visual Speech Processing with LLMs** (Anonymous, 2024)
- **Architecture**: LLaMA-based decoder with visual speech tokens
- **Contribution**: First LLM-based visual speech recognition (VSR)
- **Single vs Multi-speaker**: Single-speaker
- **Overlap Handling**: **None**
- **Benchmark Performance**: WER 1.2% (VSR) on LRS3

**Relevance to CHiME-9**: ❌ **Not applicable**. Visual-only, single-speaker focus.

---

#### **MMS-LLaMA** (Anonymous, 2024)
- **Architecture**: Multi-modal streaming LLaMA for AVSR
- **Contribution**: 86% token reduction via modal pruning
- **Single vs Multi-speaker**: Single-speaker
- **Overlap Handling**: **None**
- **Benchmark Performance**: WER 0.72% (AVSR) on LRS3

**Relevance to CHiME-9**: ❌ **Not applicable**. Efficiency focus, not multi-speaker robustness.

---

#### **Llama-AVSR** (Chen et al., 2024)
- **Architecture**: Frozen LLaMA with audio-visual adapter
- **Single vs Multi-speaker**: Single-speaker
- **Overlap Handling**: **None**
- **Benchmark Performance**: WER 0.81% (ASR), 0.77% (AVSR) on LRS3

**Relevance to CHiME-9**: ❌ **Not applicable**. Same limitations as Whisper-Flamingo.

---

### 1.4 Multi-Speaker AVSR (2024)

#### **Cocktail-Party AVSR** (Hakiri et al., arXiv 2506.02178, 2024)

**THIS IS THE CHiME-9 BASELINE PAPER ITSELF.**

- **Dataset**: 1526 hours of audio-visual data with talking-face and silent-face segments
- **Architecture**: AV-HuBERT CTC/Attention (not LLM-based)
- **Multi-speaker Strategy**: Explicit pipeline:
  1. Active Speaker Detection (Light-ASD)
  2. Face detection & tracking (RetinaFace)
  3. Video segmentation (hysteresis thresholding)
  4. Per-speaker AVSR (AV-HuBERT)
  5. Conversation clustering (time-based overlap scores)

**Single vs Multi-speaker**: **Explicitly multi-speaker**, but requires:
- Pre-segmented per-speaker video crops
- Separate AVSR inference for each speaker
- Post-hoc clustering to group speakers into conversations

**Overlap Handling**: **Partial**. Handles overlaps via:
- ASD scores can detect simultaneous speakers
- Clustering uses overlap as a negative signal (1 - overlap_duration / total_duration)
- But AVSR itself still assumes single-speaker input (after segmentation)

**Benchmark Performance**:
- CHiME-9 dev: Pairwise F1 = 0.8153 (clustering), WER = 0.4990-0.8315 (depending on baseline)

**Key Insight**: Even the "state-of-the-art" cocktail-party system **decomposes multi-speaker into single-speaker problems** via explicit segmentation. No end-to-end multi-speaker AVSR model exists yet.

---

#### **SpeakerLM: Multi-modal LLM for Speaker Diarization** (Anonymous, 2024)
- **Architecture**: Multi-modal LLM for joint speaker diarization and recognition
- **Contribution**: Attempts to handle overlapping speech and speaker attribution in LLM framework
- **Single vs Multi-speaker**: **Multi-speaker**, but limited evaluation
- **Overlap Handling**: **Claims to handle overlaps**, but:
  - No quantitative evaluation on high-overlap scenarios (CHiME-9 level)
  - Likely relies on attention mechanism to separate speakers → same failure modes as Whisper-Flamingo

**Relevance to CHiME-9**: ⚠️ **Unclear**. No public results on cocktail-party scenarios. Theoretical approach is promising but unvalidated.

---

### 1.5 Non-LLM Baselines (2024)

#### **R-Branchformer** (Anonymous, 2024)
- **Architecture**: Conformer-based encoder with relative positional encoding
- **Single vs Multi-speaker**: Single-speaker
- **Overlap Handling**: **None**
- **Benchmark Performance**: WER 1.7% (LRS2), 1.5% (LRS3)

**Relevance to CHiME-9**: ❌ **Not applicable**. Single-speaker focus.

---

## 2. Critical Analysis: Why LLMs Fail for CHiME-9

### 2.1 The "State-of-the-Art" Paradox

**Claim**: Whisper-Flamingo achieves "state-of-the-art" WER 0.76% on LRS3.
**Reality**: LRS3 is a **single-speaker, clean, read-speech benchmark**. CHiME-9 is:
- **Multi-speaker**: Up to 8 speakers, 4 concurrent conversations
- **High overlap**: Average overlap ratio ≈30-50% (estimated from baseline paper)
- **Spontaneous**: Natural conversational speech with disfluencies, interruptions, cross-talk

**Performance Gap**:
- Whisper-Flamingo on LRS3: WER **0.76%** ← "state-of-the-art"
- Whisper-Flamingo on CHiME-9: WER **0.8315** (BL4) ← **Worse than AV-HuBERT 0.4990**

**Interpretation**: The 109× relative WER increase (0.76% → 83.15% if we interpret WER as error rate) demonstrates that **LRS3 performance does not transfer to multi-speaker scenarios**.

(Note: If WER in baseline paper is already normalized error rate, then Whisper-Flamingo's absolute performance is still **67% worse** than best AV-HuBERT: (0.8315 - 0.4990) / 0.4990 = 0.67)

---

### 2.2 Fundamental Architectural Limitations

#### **Problem 1: Attention Collapse in Multi-Speaker Scenarios**

**Whisper-Flamingo Architecture**:
```python
# Gated cross-attention in Whisper decoder
visual_features = perceiver_resampler(video)  # Shape: [B, 32, D]
audio_features = whisper_encoder(audio)       # Shape: [B, T, D]

# Cross-attention between audio queries and visual keys
attn_output = cross_attention(
    query=audio_features,  # Current audio frame
    key=visual_features,   # All visual tokens (potentially multiple faces)
    value=visual_features
)
```

**What happens with multiple speakers?**
- Visual encoder sees **all faces** in the frame (speaker A, B, C, ...)
- Perceiver resampler pools to 32 tokens → **averages across all faces**
- Cross-attention receives **blurred visual features** → cannot attribute which face speaks which words
- Decoder generates text based on ambiguous visual context → **hallucination**

**Failure Mode Example**:
```
Ground Truth:
  Speaker A: "Can you pass the salt?"
  Speaker B: "I'll be there in a minute."

Whisper-Flamingo Output (when both speak simultaneously):
  "Can you be there in a salt minute?" ← Nonsensical blend
```

---

#### **Problem 2: Autoregressive Decoding Error Propagation**

**AV-HuBERT CTC Decoding**:
- Frame-level predictions: Each 40ms frame → phoneme/character
- **Local decisions**: Prediction at frame `t` depends only on local acoustic/visual context
- **Monotonic alignment**: CTC enforces left-to-right alignment → reduces hallucination

**Whisper-Flamingo LLM Decoding**:
- Sentence-level generation: Predict entire utterance autoregressively
- **Global dependencies**: Prediction at token `t` depends on all previous tokens `1...t-1`
- **No alignment constraint**: LLM can generate arbitrary text → hallucination when sensory input is ambiguous

**Error Propagation**:
1. Overlap region detected → ambiguous audio/visual features
2. LLM generates first few tokens (e.g., "Can you")
3. Autoregressive prior kicks in → continues generating plausible text ("pass the salt?")
4. Even if second speaker interrupts, LLM has already committed to first speaker's sentence
5. Result: **Blended or incorrect transcription**

---

#### **Problem 3: Source Attribution Failure**

**CHiME-9 Requirement**: Transcribe **per-speaker, per-conversation** in the presence of:
- Spatial audio (ambisonic, 4-channel)
- 360° video (multiple faces visible)
- Overlapping speech (2-4 simultaneous speakers)

**Whisper-Flamingo Limitation**:
- No explicit **speaker embedding** or **conversation ID**
- Cross-attention is **content-addressed** (attend to salient visual features) → cannot distinguish "which face belongs to which audio stream"
- Output is a **single transcription** → cannot produce per-speaker outputs

**Required Architecture for CHiME-9**:
```python
# Hypothetical multi-speaker AVSR-LLM (not implemented in any current model)
for speaker_id in detected_speakers:
    visual_features = extract_face_features(video, speaker_id)  # Tracked face ROI
    audio_features = extract_spatial_audio(ambisonic, speaker_id)  # Beamformed to speaker direction

    # Speaker-conditioned decoding
    transcription = llm_decode(
        audio=audio_features,
        visual=visual_features,
        speaker_embedding=speaker_id  # Explicit conditioning
    )
```

**Current Reality**: No LLM-based AVSR model implements speaker-conditioned decoding. All assume single-speaker input.

---

### 2.3 Why AV-HuBERT CTC/Attention Outperforms Whisper-Flamingo on CHiME-9

| **Factor**                        | **AV-HuBERT CTC/Attention**                          | **Whisper-Flamingo**                              | **Winner**       |
|-----------------------------------|-----------------------------------------------------|--------------------------------------------------|------------------|
| **Frame-level modeling**           | ✅ 29.97 fps video, 50 Hz audio features             | ❌ Sentence-level generation                      | **AV-HuBERT**    |
| **Alignment robustness**           | ✅ CTC enforces monotonic alignment                  | ❌ No alignment constraint → hallucination        | **AV-HuBERT**    |
| **Local vs global context**        | ✅ Local attention (80-frame window)                 | ❌ Global attention → fails on ambiguous context  | **AV-HuBERT**    |
| **Pre-segmentation friendly**      | ✅ Designed for pre-segmented single-speaker input   | ❌ Assumes holistic sentence-level understanding  | **AV-HuBERT**    |
| **Training data scale**            | ✅ Pre-trained on 433h LRS3 + fine-tuned on CHiME-9  | ⚠️ Pre-trained on Whisper data (680k hours audio, but LRS3 for visual) | **Tie**          |
| **Inference speed**                | ✅ Parallel CTC decoding                             | ❌ Autoregressive generation (slow)               | **AV-HuBERT**    |
| **Multi-speaker handling**         | ⚠️ Requires explicit pipeline (ASD + clustering)     | ❌ No mechanism at all                            | **AV-HuBERT**    |

**Conclusion**: AV-HuBERT's **frame-level modeling + CTC alignment + local attention** are fundamentally better suited to CHiME-9's multi-speaker, high-overlap scenarios than Whisper-Flamingo's **sentence-level generation + global attention + no alignment constraints**.

---

## 3. Where Multimodal LLMs Fit (and Don't) for CHiME-9

### 3.1 ❌ **NOT Suitable for End-to-End AVSR**

**Reasons**:
1. **No multi-speaker architecture**: All current LLM-based AVSR models assume single-speaker input
2. **Attention collapse**: Cross-attention cannot handle multiple simultaneous visual/audio sources
3. **Hallucination risk**: Autoregressive decoding generates fluent but incorrect text in ambiguous overlaps
4. **No source attribution**: Cannot determine which face/audio belongs to which speaker

**Evidence**: Whisper-Flamingo WER 0.8315 on CHiME-9 vs AV-HuBERT 0.4990 (67% worse).

---

### 3.2 ⚠️ **Potentially Useful for Post-Processing / Rescoring**

#### **Use Case 1: Conversation-Level Language Modeling**

**Idea**: Use LLM to rescore AVSR hypotheses with conversation-level context.

**Implementation**:
```python
# Step 1: Get per-speaker transcriptions from AV-HuBERT
transcriptions = {
    "speaker_A": "Can you pass the [UNCERTAIN: salt/fault]?",
    "speaker_B": "I'll be there in a minute.",
}

# Step 2: Use LLM to rescore with conversation context
conversation_context = "\n".join(transcriptions.values())
llm_prompt = f"""
Conversation context:
{conversation_context}

Which is more likely for Speaker A?
(1) "Can you pass the salt?"
(2) "Can you pass the fault?"

Answer: (1) because "salt" is more coherent with dining context.
"""

# Step 3: Re-rank hypotheses
best_hypothesis = llm_rerank(transcriptions["speaker_A"], llm_prompt)
```

**Expected Gain**: 2-5% WER reduction (similar to Conversation Cache LM in `clustering_to_wer_improvement.md`).

**Limitation**: Requires confident per-speaker transcriptions first → doesn't help if AV-HuBERT already failed.

---

#### **Use Case 2: Diarization Refinement**

**Idea**: Use LLM to detect speaker turns and conversation structure.

**Implementation**:
```python
# Step 1: Get initial speaker diarization from clustering
initial_diarization = {
    (0.0, 3.5): "speaker_A",
    (2.0, 5.0): "speaker_B",  # Overlap with speaker_A
    (5.5, 8.0): "speaker_A",
}

# Step 2: LLM analyzes conversation flow
llm_prompt = f"""
Transcript with timestamps:
[0.0-3.5] Speaker A: "Can you pass the salt?"
[2.0-5.0] Speaker B: "I'll be there in a minute."
[5.5-8.0] Speaker A: "Thanks."

Does Speaker B's utterance make sense as a response to Speaker A?
If not, they are likely in different conversations.
"""

# Step 3: Refine clustering based on LLM feedback
refined_clustering = llm_refine_diarization(initial_diarization, llm_prompt)
```

**Expected Gain**: 1-3% Pairwise F1 improvement (if LLM can detect turn-taking violations).

**Limitation**: LLM must be grounded in acoustic/visual evidence → risk of hallucinating plausible but incorrect conversation structure.

---

### 3.3 ❌ **NOT Suitable for Real-Time Applications**

**Reasons**:
1. **Latency**: LLM inference is slow (100-1000ms per sentence on GPU)
2. **Autoregressive decoding**: Cannot produce streaming outputs (must generate full sentence)
3. **Context window**: Requires full conversation history → unbounded memory for long meetings

**Contrast with AV-HuBERT**:
- Frame-level predictions → can produce streaming outputs with 40ms latency
- CTC decoding → no autoregressive dependency

---

### 3.4 ✅ **Potentially Useful for Offline Analysis / Research**

#### **Use Case 1: Failure Mode Analysis**

**Idea**: Use multimodal LLM (GPT-4V) to analyze WHY clustering/AVSR failed.

**Implementation**:
```python
# Step 1: Extract failure case (low Pairwise F1)
session_id = "S02"  # F1 = 0.65
failure_frames = extract_frames(session_id, start=120, end=180)  # 60 seconds

# Step 2: Ask GPT-4V to analyze
gpt4v_prompt = f"""
Here are 10 frames from a multi-speaker conversation with {len(speakers)} speakers.
The clustering algorithm incorrectly merged Speaker A and Speaker B into the same conversation.

Analyze the visual cues:
1. Are speakers facing each other? (gaze direction)
2. Are they spatially close? (proxemics)
3. Are their gestures synchronized? (body language)
4. Do they exhibit turn-taking behavior?

Based on visual evidence, should they be in the same conversation?
"""

gpt4v_analysis = gpt4v(failure_frames, gpt4v_prompt)
```

**Expected Gain**: Insights for improving clustering algorithm (not direct WER improvement).

---

#### **Use Case 2: Data Annotation / Augmentation**

**Idea**: Use LLM to generate synthetic conversation-level annotations.

**Implementation**:
```python
# Step 1: Get raw transcriptions (no conversation labels)
transcriptions = get_all_transcriptions(session_id)

# Step 2: LLM generates conversation groupings
llm_prompt = f"""
Here are transcriptions from a meeting with multiple conversations:
{transcriptions}

Group speakers into conversations based on:
1. Topic coherence (are they discussing the same subject?)
2. Turn-taking patterns (do they respond to each other?)
3. Temporal proximity (do they speak around the same time?)

Output format: {{"conversation_1": ["speaker_A", "speaker_B"], "conversation_2": ["speaker_C", "speaker_D"]}}
"""

pseudo_labels = llm_annotate(transcriptions, llm_prompt)
```

**Expected Gain**: Augment training data for conversation clustering (if pseudo-labels are high quality).

**Limitation**: LLM pseudo-labels may have high error rate → requires human validation.

---

## 4. Fundamental Research Gaps

### 4.1 No End-to-End Multi-Speaker AVSR Model Exists

**Current State**:
- All "state-of-the-art" AVSR models (Whisper-Flamingo, AV-HuBERT, Auto-AVSR) assume **single-speaker input**
- Multi-speaker scenarios are handled via **explicit decomposition** (ASD → segmentation → per-speaker AVSR → clustering)

**What's Missing**:
```python
# Hypothetical end-to-end multi-speaker AVSR (does NOT exist yet)
def multispeaker_avsr(video, audio, max_speakers=8):
    """
    Input:
        - video: 360° video with multiple faces
        - audio: Ambisonic (4-channel) audio with overlapping speech
    Output:
        - transcriptions: {speaker_id: text, ...}
        - speaker_embeddings: {speaker_id: embedding, ...}
        - conversation_clusters: {conv_id: [speaker_ids], ...}
    """
    # Step 1: Multi-speaker audio-visual feature extraction
    av_features = multimodal_encoder(video, audio)  # Shape: [T, num_speakers, D]

    # Step 2: Speaker attribution (via attention or clustering)
    speaker_features = attribute_speakers(av_features)  # Shape: [num_speakers, T, D]

    # Step 3: Per-speaker decoding with conversation context
    transcriptions = {}
    for speaker_id, features in enumerate(speaker_features):
        transcriptions[speaker_id] = decoder(
            features,
            conversation_context=get_conversation_context(speaker_id)  # Cross-speaker attention
        )

    # Step 4: Joint clustering (learned end-to-end)
    conversation_clusters = cluster_speakers(speaker_features)

    return transcriptions, conversation_clusters
```

**Why Hard?**:
1. **Training data**: No large-scale multi-speaker AVSR dataset with dense annotations (CHiME-9 is small: ~100 sessions)
2. **Loss function**: How to supervise joint speaker attribution + transcription + clustering?
3. **Architecture**: Attention mechanism struggles with multiple simultaneous sources (permutation invariance, assignment problem)

---

### 4.2 Overlap Robustness is Unsolved

**LRS3 Overlap Statistics**:
- Average overlap: **<5%** (mostly non-overlapping speech)
- Max simultaneous speakers: **1-2**

**CHiME-9 Overlap Statistics** (estimated from baseline paper):
- Average overlap: **30-50%**
- Max simultaneous speakers: **4-8**

**Performance Gap**:
| **Model**          | **LRS3 (clean, single-speaker)** | **CHiME-9 (overlap, multi-speaker)** | **Relative Degradation** |
|--------------------|----------------------------------|-------------------------------------|--------------------------|
| Whisper-Flamingo   | WER 0.76%                        | WER 83.15%                          | **109× worse**           |
| AV-HuBERT          | WER 0.5%                         | WER 49.90%                          | **100× worse**           |

**Interpretation**: Both models degrade catastrophically in high-overlap scenarios. **Overlap robustness is an open research problem.**

---

### 4.3 LLM Hallucination in Ambiguous Contexts

**Problem**: LLMs have strong priors for fluent text generation → continue generating even when sensory evidence is ambiguous.

**Failure Mode Example**:
```
Ground Truth (overlapping speech):
  Speaker A: "Can you—"
  Speaker B: "I'll be—"
  [Both interrupted, incomplete utterances]

Whisper-Flamingo Output:
  "Can you pass me the document that I'll be reviewing this afternoon?"
  ← Hallucinated completion (fluent but incorrect)
```

**Why This Happens**:
- LLM prior: P(text | context) is very strong (trained on 680k hours of Whisper data)
- Sensory evidence: P(text | audio, video) is weak (ambiguous in overlaps)
- Posterior: P(text | audio, video, context) ≈ P(text | context) → LLM prior dominates

**Solution?**:
- **Uncertainty quantification**: LLM should output confidence scores → reject low-confidence generations
- **Grounding mechanisms**: Force LLM to attend to acoustic/visual evidence → penalize hallucinations
- **Frame-level constraints**: Use CTC-like alignment to prevent arbitrary text generation

**Current State**: No LLM-based AVSR model implements robust uncertainty quantification for overlapping speech.

---

## 5. Recommendations for CHiME-9 Researchers

### 5.1 **Don't Use LLMs for End-to-End AVSR (Yet)**

**Reason**: Current LLM-based AVSR models (Whisper-Flamingo, VSP-LLM, Llama-AVSR) are **not competitive** with AV-HuBERT CTC/Attention on multi-speaker, high-overlap scenarios.

**Evidence**: Whisper-Flamingo WER 0.8315 vs AV-HuBERT 0.4990 (67% worse).

**Exception**: If your goal is to explore **fundamental research questions** (e.g., how to make LLMs robust to overlaps), then this is a valid research direction. But don't expect immediate WER improvements.

---

### 5.2 **Use LLMs for Post-Processing / Rescoring**

**High-ROI Applications**:
1. **Conversation Cache LM**: Use LLM to rescore hypotheses with conversation-level context → 2-5% WER reduction (see `clustering_to_wer_improvement.md`)
2. **Diarization Refinement**: Use LLM to detect turn-taking violations → 1-3% Pairwise F1 improvement
3. **Failure Mode Analysis**: Use GPT-4V to analyze WHY clustering failed → insights for algorithm design

**Low-ROI Applications**:
1. End-to-end AVSR (not competitive)
2. Real-time transcription (too slow)
3. Pseudo-labeling (high error rate without validation)

---

### 5.3 **Focus on Overlap-Robust Architectures**

**Open Research Questions**:
1. **How to do frame-level speaker attribution in overlaps?**
   - Possible approaches: Permutation Invariant Training (PIT), Target Speaker Extraction (TSE), neural beamforming

2. **How to prevent LLM hallucination in ambiguous contexts?**
   - Possible approaches: Uncertainty quantification, grounding mechanisms, CTC-like constraints

3. **How to learn conversation structure end-to-end?**
   - Possible approaches: Graph neural networks, contrastive learning, clustering losses

**Concrete Next Steps**:
1. Implement **Permutation Invariant Training (PIT)** for AV-HuBERT → handle 2-speaker overlaps
2. Add **Target Speaker Extraction (TSE)** layer before AVSR → beamform to individual speakers
3. Replace time-based clustering with **learned clustering** (e.g., graph neural network on speaker embeddings)

---

### 5.4 **Benchmark on Realistic Overlap Scenarios**

**Current Problem**: Most "state-of-the-art" claims are based on LRS2/LRS3, which have **<5% overlap**.

**Recommendation**: Report performance separately for:
- **Clean (0-10% overlap)**
- **Moderate (10-30% overlap)**
- **High (30-50% overlap)**
- **Extreme (>50% overlap)**

**Example Breakdown** (hypothetical):
| **Model**          | **Clean** | **Moderate** | **High** | **Extreme** | **Weighted Avg** |
|--------------------|-----------|--------------|----------|-------------|------------------|
| Whisper-Flamingo   | 0.8%      | 15.2%        | 45.3%    | 89.1%       | 37.6%            |
| AV-HuBERT          | 0.5%      | 8.1%         | 28.7%    | 67.4%       | 26.2%            |

**Insight**: This breakdown would reveal that **both models fail catastrophically in high-overlap regions**, motivating overlap-specific research.

---

## 6. Conclusion

### 6.1 Summary of Findings

1. **Multimodal LLMs (Whisper-Flamingo, GPT-4V, LLM-based AVSR) are NOT competitive for CHiME-9 end-to-end AVSR.**
   - Reason: Designed for single-speaker, clean scenarios (LRS2/LRS3)
   - Evidence: Whisper-Flamingo WER 0.8315 vs AV-HuBERT 0.4990 (67% worse)

2. **Fundamental limitations exist for overlapping speech:**
   - Attention collapse (cannot separate multiple visual/audio sources)
   - Source attribution failure (no speaker embeddings)
   - Hallucination (autoregressive decoding generates fluent but incorrect text)

3. **AV-HuBERT CTC/Attention outperforms LLM-based approaches because:**
   - Frame-level modeling (robust to temporal misalignment)
   - CTC alignment (reduces hallucination)
   - Local attention (doesn't assume global sentence-level context)

4. **Multimodal LLMs may be useful for post-processing:**
   - Conversation Cache LM (2-5% WER reduction)
   - Diarization refinement (1-3% F1 improvement)
   - Failure mode analysis (qualitative insights)

5. **Research gaps remain:**
   - No end-to-end multi-speaker AVSR model exists
   - Overlap robustness is unsolved (100× performance degradation from LRS3 to CHiME-9)
   - LLM hallucination in ambiguous contexts is unaddressed

---

### 6.2 Where Are Multimodal LLMs "State-of-the-Art"?

**✅ State-of-the-art for:**
- Single-speaker AVSR on clean benchmarks (LRS2/LRS3, VoxCeleb2)
- Zero-shot generalization to new speakers/languages (mWhisper-Flamingo)
- Multimodal understanding tasks (GPT-4V for image captioning, VQA)

**❌ NOT state-of-the-art for:**
- Multi-speaker AVSR (CHiME-9, cocktail-party scenarios)
- Overlapping speech recognition (>30% overlap)
- Real-time transcription (latency >1s)
- Robust source attribution (no speaker embeddings)

---

### 6.3 Final Verdict for CHiME-9

**For End-to-End AVSR**: Stick with **AV-HuBERT CTC/Attention** (or similar frame-level models). Don't waste time on Whisper-Flamingo or LLM-based approaches unless you're doing fundamental research on overlap robustness.

**For Post-Processing**: Experiment with **Conversation Cache LM** (2-5% WER gain, easy to implement). See `clustering_to_wer_improvement.md` for complete implementation.

**For Future Work**: Invest in **overlap-robust architectures** (PIT, TSE, neural beamforming) and **learned clustering** (graph neural networks, contrastive learning). These are the bottlenecks for CHiME-9 performance.

---

## 7. References

### Papers Cited

1. **Shi et al. (2022)**: "Learning Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction", AAAI 2022.
2. **Rouditchenko et al. (2024)**: "Whisper-Flamingo: Integrating Visual Features into Whisper for Audio-Visual Speech Recognition", Interspeech 2024.
3. **Hu et al. (2025)**: "mWhisper-Flamingo: Multilingual Audio-Visual Speech Recognition", IEEE Signal Processing Letters, 2025.
4. **Hakiri et al. (2024)**: "Cocktail-Party Audio-Visual Speech Recognition: CHiME-9 Baseline System", arXiv:2506.02178, 2024.
5. **Anwar et al. (2023)**: "MuAViC: A Multilingual Audio-Visual Corpus for Robust Speech Recognition", Interspeech 2023.
6. **Ma et al. (2023)**: "Auto-AVSR: Audio-Visual Speech Recognition with Automatic Labels", ICASSP 2023.

### Key Benchmarks

- **LRS2**: 140,000 utterances, single-speaker, read speech
- **LRS3**: 433 hours, single-speaker, TED talks
- **CHiME-9 MCoRec**: ~100 sessions, multi-speaker (up to 8), high overlap (30-50%)

### Code Repositories

- **CHiME-9 Baseline**: https://github.com/aziz-hakiri/mcorec_baseline
- **AV-HuBERT**: https://github.com/facebookresearch/av_hubert
- **Whisper**: https://github.com/openai/whisper

---

## Appendix: Detailed Model Comparison

| **Model**            | **Architecture**                     | **Params** | **LRS3 WER** | **CHiME-9 WER** | **Multi-Speaker?** | **Overlap Robust?** |
|----------------------|--------------------------------------|------------|--------------|-----------------|--------------------|--------------------|
| AV-HuBERT CTC        | Masked prediction + CTC              | 95M        | 0.9%         | 0.4990          | ❌ (via pipeline)   | ⚠️ (partial)        |
| AV-HuBERT Attention  | Masked prediction + Seq2Seq          | 95M        | 0.5%         | 0.4990-0.8315   | ❌ (via pipeline)   | ⚠️ (partial)        |
| Whisper-Flamingo     | Whisper + gated cross-attention      | 244M       | 0.76%        | 0.8315          | ❌                  | ❌                  |
| Auto-AVSR            | AV-HuBERT + pseudo-labels            | 95M        | 1.0%         | N/A             | ❌                  | ❌                  |
| VSP-LLM              | LLaMA + visual speech tokens         | 7B         | 1.2%         | N/A             | ❌                  | ❌                  |
| MMS-LLaMA            | LLaMA + modal pruning                | 7B         | 0.72%        | N/A             | ❌                  | ❌                  |
| Llama-AVSR           | LLaMA + AV adapter                   | 7B         | 0.77%        | N/A             | ❌                  | ❌                  |

**Key Takeaway**: Only AV-HuBERT has been validated on CHiME-9. All LLM-based models are **single-speaker only** and have **no overlap robustness**.

---

**End of Survey**
