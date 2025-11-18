# **Methods Description: CHiME-9 MCoRec Baseline System**

## **Overview**

The CHiME-9 Task 1 baseline system addresses Multi-Modal Context-aware Recognition (MCoRec) in environments with multiple concurrent conversations. The system processes a single 360° video and audio recording to both **transcribe each speaker's speech** and **identify which speakers belong to the same conversation**. The baseline implements a pipeline consisting of five core components: active speaker detection, face landmark detection and mouth cropping, video segmentation and chunking, audio-visual speech recognition, and time-based conversation clustering.

---

## **1. Active Speaker Detection (ASD)**

### **Purpose**
Identifies temporal regions where each speaker is actively speaking, enabling the AVSR system to process only speech-active segments and providing temporal features for conversation clustering.

### **Model**
The baseline employs the **Light-ASD** model [1], a lightweight active speaker detection system that combines audio and visual modalities.

### **Input**
- Face crop video (96×96 pixel face region extracted from 360° video)
- Audio stream sampled at 16 kHz

### **Processing** (`script/asd.py`)
1. **Audio features**: 13-dimensional MFCC features computed with 25ms window and 10ms stride
2. **Video features**: Grayscale face crops resized to 224×224 pixels, center-cropped to 112×112 pixels
3. **Multi-scale evaluation**: ASD scores computed across multiple temporal windows (1, 2, 3, 4, 5, and 6 seconds) and averaged
4. **Pretrained weights**: `model-bin/finetuning_TalkSet.model`

### **Output**
Frame-wise active speaker detection scores stored in JSON format (`track_xx_asd.json`), where each frame index maps to a continuous score indicating speech activity likelihood.

---

## **2. Face Landmark Detection and Mouth Cropping**

### **Purpose**
Extracts precise mouth regions from face crop videos, providing visual input for audio-visual speech recognition models.

### **Models**
- **RetinaFace** [2]: Face detector based on single-stage dense face localization
- **FAN (Face Alignment Network)** [3]: 2D facial landmark detector for identifying facial keypoints

### **Input**
- Face crop video (96×96 pixels) from 360° recording
- Synchronized audio stream

### **Processing** (`script/lip_crop.py`)
1. Face detection using RetinaFace on each video frame
2. Facial landmark detection using FAN to identify mouth region keypoints
3. Geometric transformation to crop and align mouth region to 96×96 pixels
4. Audio-video synchronization maintained throughout processing

The implementation follows the preprocessing pipeline from the Auto-AVSR repository [4].

### **Output**
Mouth crop video with audio (`track_xx_lip.av.mp4`), containing precisely aligned mouth region videos suitable for AVSR model input.

---

## **3. Video Segmentation and Chunking**

### **Purpose**
Segments long mouth crop videos into manageable chunks (≤15 seconds) based on speech activity, enabling efficient batch processing for AVSR inference.

### **Algorithm** (`src/talking_detector/segmentation.py`)

The segmentation employs a **three-stage hysteresis-based approach**:

#### **Stage 1: Speech Region Detection**
- **Hysteresis thresholding** with onset threshold (1.0) and offset threshold (0.8)
- Identifies continuous speech regions where ASD scores exceed thresholds
- Prevents rapid on/off switching during speech boundaries

#### **Stage 2: Gap Filling**
- **Minimum off-duration**: 0.5 seconds
- Merges speech regions separated by short non-speech gaps
- Prevents fragmentation of continuous utterances

#### **Stage 3: Duration Filtering and Splitting**
- **Minimum on-duration**: 1.0 seconds (removes short noise segments)
- **Maximum chunk size**: 15 seconds (configurable via `--max_length` parameter in `inference.py`)
- Long continuous speech segments are divided into equal-length chunks to fit within maximum duration constraint

### **Input**
- Mouth crop video
- ASD scores (frame-wise)

### **Output**
List of video segments with start/end timestamps (in seconds), each representing a speech-active region suitable for AVSR processing.

---

## **4. Audio-Visual Speech Recognition (AVSR)**

### **Purpose**
Transcribes speech from segmented mouth crop videos by jointly modeling audio and visual modalities.

### **Input**
- Segmented mouth crop videos (≤15 seconds per segment)
- Corresponding audio streams (16 kHz sampling rate)
- Temporal alignment information (start/end timestamps)

### **Processing** (`script/inference.py`)

The baseline provides **four AVSR model configurations** (BL1-BL4), all sharing common components:

#### **Common Components**
- **Tokenizer**: SentencePiece unigram tokenizer with 5,000 vocabulary units
- **Decoding**: Beam search with beam size = 3
- **Input format**: 96×96 pixel mouth crop videos at 25 fps, 16 kHz audio

#### **Model Architectures**

**BL1: AV-HuBERT CTC/Attention** [5]
- **Encoder**: Pre-trained AV-HuBERT Large model (`nguyenvulebinh/avhubert_encoder_large_noise_pt_noise_ft_433h`)
- **Decoder**: Transformer decoder with joint CTC/Attention training
- **Training data**: LRS2, VoxCeleb2, AVYT, AVYT-mix (~1.9M samples)
- **Implementation**: `src/avhubert_avsr/avhubert_avsr_model.py`
- **Checkpoint**: `./model-bin/avsr_cocktail`

**BL2: MuAViC-EN** [6]
- **Encoder**: AV-HuBERT encoder adapted for multilingual audio-visual corpus
- **Decoder**: Transformer decoder from Hugging Face's Speech2Text architecture
- **Training data**: MuAViC English subset
- **Implementation**: `src/avhubert_muavic/avhubert2text.py`
- **Checkpoint**: `nguyenvulebinh/AV-HuBERT-MuAViC-en` (auto-downloaded from Hugging Face)

**BL3: Auto-AVSR** [4]
- **Encoder**: Separate visual encoder (3D CNN) and audio encoder (Conformer)
- **Fusion**: Concatenation-based fusion of audio and visual features
- **Decoder**: Conformer-based CTC/Attention decoder
- **Training data**: Trained on LRS2, LRS3, VoxCeleb2, AVSpeech datasets
- **Implementation**: `src/auto_avsr/avsr_model.py`
- **Checkpoint**: `./model-bin/auto_avsr/avsr_trlrwlrs2lrs3vox2avsp_base.pth`

**BL4: AV-HuBERT CTC/Attention (MCoRec Fine-tuned)** [5]
- Same architecture as BL1
- **Additional fine-tuning**: MCoRec training set (89.3k segments)
- **Training script**: `script/train.py` (multi-GPU distributed training)
- **Convergence**: ~200,000 steps with batch size 24
- **Checkpoint**: `./model-bin/avsr_cocktail_mcorec_finetune`

### **Output**
Time-aligned transcriptions in WebVTT format (`spk_X.vtt`), containing:
- Start and end timestamps for each segment
- Transcribed text with normalization applied
- Speaker-specific transcription files

---

## **5. Time-based Conversation Clustering**

### **Purpose**
Groups speakers into conversation clusters based on temporal speaking patterns, leveraging the observation that speakers in the same conversation tend to speak sequentially (low overlap), while speakers in different conversations often speak simultaneously (high overlap).

### **Input**
Active speaker detection scores for all target speakers (extracted via component 1)

### **Processing** (`src/cluster/conv_spks.py`)

The clustering algorithm consists of **four stages**:

#### **Stage 1: Speaker Activity Extraction**
- Applies ASD-based segmentation (component 3) to each speaker's detection scores
- Extracts time segments `[(start₁, end₁), (start₂, end₂), ...]` where each speaker is actively talking
- Aligns segments to evaluation intervals (UEM: Unmapped Evaluation Map)

#### **Stage 2: Pairwise Conversation Score Calculation**
For each speaker pair `(i, j)`:
1. **Overlap duration**: Total time both speakers speak simultaneously
   ```
   overlap = Σ max(0, min(end_i, end_j) - max(start_i, start_j))
   ```
2. **Non-overlap duration**: Total speaking time without overlap
   ```
   non_overlap = duration_i + duration_j - 2×overlap
   ```
3. **Conversation likelihood score**:
   ```
   score(i,j) = 1 - (overlap / (overlap + non_overlap))
   ```
   - **Higher score** (approaching 1): Less overlap → likely same conversation
   - **Lower score** (approaching 0): More overlap → likely different conversations

#### **Stage 3: Distance Matrix Construction**
Convert conversation scores to distances for hierarchical clustering:
```
distance(i,j) = 1 - score(i,j)
```

#### **Stage 4: Agglomerative Clustering**
- **Algorithm**: Hierarchical agglomerative clustering with complete linkage
- **Metric**: Pre-computed distance matrix
- **Threshold**: 0.7 (default distance threshold: 1 - 0.7 = 0.3)
- **Max clusters**: 4 conversations (constrained by challenge rules)
- **Implementation**: scikit-learn's `AgglomerativeClustering`

### **Output**
Speaker-to-cluster mapping in JSON format (`speaker_to_cluster.json`):
```json
{
    "spk_0": 0,
    "spk_1": 0,
    "spk_2": 1,
    "spk_3": 1
}
```

---

## **Evaluation Metrics** (`script/evaluate.py`)

The baseline system is evaluated using three complementary metrics:

### **1. Individual Speaker WER (Word Error Rate)**
- **Computation**: jiwer library with WebVTT alignment
- **Normalization**: English text normalization + disfluency removal
- **Scope**: Evaluated within speaker-specific evaluation intervals (UEM)

### **2. Conversation Clustering F1 (Pairwise)**
- **Metric**: Pairwise F1 score comparing all speaker pairs
- **Formula**:
  ```
  For each pair (i,j):
    - TP: Both in same cluster (predicted) and same cluster (ground truth)
    - FP: In same cluster (predicted) but different clusters (ground truth)
    - FN: In different clusters (predicted) but same cluster (ground truth)

  F1 = 2×TP / (2×TP + FP + FN)
  ```

### **3. Joint ASR-Clustering Error Rate** (Primary Metric)
- **Formula**:
  ```
  Error(speaker) = 0.5 × WER(speaker) + 0.5 × (1 - F1(speaker))
  ```
- **Per-speaker F1**: Pairwise F1 computed only for pairs involving that speaker
- **Aggregation**: Average across all speakers

---

## **Baseline System Results (Development Set)**

All baseline systems use **identical** pipeline components (ASD, mouth cropping, segmentation, clustering) and differ **only** in the AVSR model architecture:

| System | AVSR Model | MCoRec Fine-tuned | Conversation Clustering F1 | Speaker WER | Joint Error Rate |
|--------|------------|-------------------|----------------------------|-------------|------------------|
| **BL1** | AV-HuBERT CTC/Attention [5] | No | **0.8153** | 0.5536 | 0.3821 |
| **BL2** | MuAViC-EN [6] | No | **0.8153** | 0.7180 | 0.4643 |
| **BL3** | Auto-AVSR [4] | No | **0.8153** | 0.8315 | 0.5211 |
| **BL4** | AV-HuBERT CTC/Attention [5] | **Yes** | **0.8153** | **0.4990** | **0.3548** |

### **Key Observations**
1. **Clustering performance**: All baselines achieve identical conversation clustering F1 (0.8153) because they use the same time-based clustering algorithm operating on pre-computed ASD scores.

2. **ASR performance**: BL4 (fine-tuned on MCoRec) achieves the lowest WER (0.4990) and best joint error rate (0.3548), demonstrating the benefit of domain adaptation.

3. **Model comparison**: Among non-fine-tuned models, BL1 (AV-HuBERT CTC/Attention) outperforms BL2 (MuAViC-EN) and BL3 (Auto-AVSR) on this challenging cocktail-party scenario.

---

## **Implementation Details**

### **Scripts and Key Files**
- **`script/asd.py`**: Active speaker detection processing
- **`script/lip_crop.py`**: Face landmark detection and mouth cropping
- **`script/inference.py`**: Main inference pipeline (segmentation, AVSR, clustering)
- **`script/evaluate.py`**: Evaluation metrics computation
- **`script/train.py`**: AVSR model fine-tuning
- **`src/talking_detector/segmentation.py`**: Hysteresis-based video segmentation
- **`src/cluster/conv_spks.py`**: Time-based conversation clustering algorithm

### **System Requirements**
- **GPU**: NVIDIA GPU with CUDA 12+ and ≥24GB VRAM
- **Processing time**: ~2 hours for complete dev set on single NVIDIA Titan RTX 24GB
- **Python**: 3.11 with PyTorch, torchaudio, torchvision, scikit-learn, jiwer

### **Key Parameters**
- **Video segmentation**: `max_length=15` seconds (configurable)
- **Beam search**: `beam_size=3`
- **ASD thresholds**: onset=1.0, offset=0.8
- **Clustering threshold**: 0.7
- **Frame rate**: 25 fps (video), 16 kHz (audio)

---

## **References**

[1] Liao et al. (2023). "A Light Weight Model for Active Speaker Detection." *arXiv preprint arXiv:2303.04439*.

[2] Deng et al. (2020). "RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild." *CVPR 2020*.

[3] Bulat & Tzimiropoulos (2017). "How far are we from solving the 2D & 3D Face Alignment problem?" *ICCV 2017*.

[4] Ma et al. (2023). "Auto-AVSR: Audio-Visual Speech Recognition with Automatic Labels." *ICASSP 2023*, arXiv:2303.14307.

[5] Nguyen et al. (2024). "Cocktail-Party Audio-Visual Speech Recognition." *arXiv preprint arXiv:2506.02178*.

[6] Anwar et al. (2023). "MuAViC: A Multilingual Audio-Visual Corpus for Robust Speech Recognition and Robust Speech-to-Text Translation." *arXiv preprint arXiv:2303.00628*.

---

## **Document Information**

- **Generated**: 2025-11-18
- **Repository**: https://github.com/aziz-hakiri/mcorec_baseline
- **Challenge**: CHiME-9 Task 1 - Multi-Modal Context-aware Recognition (MCoRec)
- **Analysis Type**: Baseline System Methods Description
