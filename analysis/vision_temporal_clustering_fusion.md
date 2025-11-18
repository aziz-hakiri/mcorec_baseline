# **Vision-Temporal Clustering Fusion: Integration Strategies**

## **Overview**

This document explores how an independent **vision-based gaze clustering system** (achieving pairwise F1 ≈ 0.9) can be integrated with the baseline's **time-based overlap clustering** (F1 ≈ 0.8153) to create a more robust multi-modal conversation clustering system.

**Key assumptions:**
- Vision system produces `speaker_to_cluster.json` per session (same format as baseline)
- Vision system uses gaze patterns, spatial proximity, body orientation, face direction, etc.
- Vision system is independent (no access to audio/temporal features)
- Vision clustering F1 ≈ 0.9 (significantly better than time-based F1 ≈ 0.8153)

---

## **Part 1: Concrete Integration Mechanisms**

### **1.1 As Constraints for Time-Based Clustering**

#### **Mechanism 1a: Must-Link / Cannot-Link Constraints**

Convert vision clustering into pairwise constraints for agglomerative clustering:

```python
def get_constraints_from_vision(vision_clusters):
    """
    Extract must-link and cannot-link constraints

    Must-link: Speakers in same vision cluster should stay together
    Cannot-link: Speakers in different vision clusters should stay separate
    """
    must_link = []
    cannot_link = []

    speakers = list(vision_clusters.keys())
    for i in range(len(speakers)):
        for j in range(i+1, len(speakers)):
            if vision_clusters[speakers[i]] == vision_clusters[speakers[j]]:
                must_link.append((speakers[i], speakers[j]))
            else:
                cannot_link.append((speakers[i], speakers[j]))

    return must_link, cannot_link


def constrained_clustering(distances, speaker_ids, vision_clusters,
                          must_link_weight=0.1, cannot_link_weight=0.5):
    """
    Modify distance matrix based on vision constraints

    - Decrease distance for must-link pairs (encourage merging)
    - Increase distance for cannot-link pairs (discourage merging)
    """
    constrained_distances = distances.copy()
    must_link, cannot_link = get_constraints_from_vision(vision_clusters)

    speaker_to_idx = {spk: i for i, spk in enumerate(speaker_ids)}

    # Apply must-link constraints (decrease distance)
    for spk_i, spk_j in must_link:
        i, j = speaker_to_idx[spk_i], speaker_to_idx[spk_j]
        constrained_distances[i, j] *= (1 - must_link_weight)  # e.g., 0.3 → 0.27
        constrained_distances[j, i] *= (1 - must_link_weight)

    # Apply cannot-link constraints (increase distance)
    for spk_i, spk_j in cannot_link:
        i, j = speaker_to_idx[spk_i], speaker_to_idx[spk_j]
        constrained_distances[i, j] = min(constrained_distances[i, j] + cannot_link_weight, 1.0)
        constrained_distances[j, i] = min(constrained_distances[j, i] + cannot_link_weight, 1.0)

    # Cluster with modified distances
    clustering = AgglomerativeClustering(
        distance_threshold=0.3,
        metric='precomputed',
        linkage='complete'
    )
    return clustering.fit_predict(constrained_distances)
```

**Pros:**
- Soft constraints (doesn't force perfect agreement)
- Can correct vision errors using temporal evidence
- Tunable constraint strength

**Cons:**
- Still depends on distance threshold
- May violate constraints if temporal evidence is strong
- Hyperparameters (weights) need tuning

---

#### **Mechanism 1b: Initialize Clusters from Vision**

Use vision clustering as starting point, refine with time-based merging:

```python
def refine_vision_clusters_with_temporal(vision_clusters, temporal_scores,
                                         merge_threshold=0.85):
    """
    Start with vision clusters, merge only if temporal score is very high

    Conservative merging: Only fix vision errors where temporal evidence is strong
    """
    refined_clusters = vision_clusters.copy()

    # Consider merging vision clusters if temporal overlap is very low
    vision_cluster_ids = set(vision_clusters.values())

    for cluster_a in vision_cluster_ids:
        for cluster_b in vision_cluster_ids:
            if cluster_a >= cluster_b:
                continue

            # Get speakers in each cluster
            spks_a = [spk for spk, cid in vision_clusters.items() if cid == cluster_a]
            spks_b = [spk for spk, cid in vision_clusters.items() if cid == cluster_b]

            # Check average temporal score between clusters
            scores = []
            for spk_i in spks_a:
                for spk_j in spks_b:
                    scores.append(temporal_scores[spk_i, spk_j])

            avg_score = np.mean(scores)

            # If very high temporal score (very low overlap), consider merging
            if avg_score > merge_threshold:
                # Merge cluster_b into cluster_a
                for spk in spks_b:
                    refined_clusters[spk] = cluster_a

    return refined_clusters
```

**Pros:**
- Preserves vision clustering structure
- Only corrects clear vision errors
- Conservative (low risk of introducing new errors)

**Cons:**
- Cannot fix vision under-clustering (too many separate clusters)
- Merge-only (no split capability)
- May miss subtle improvements

---

### **1.2 As Regularizer for Clustering Decisions**

#### **Mechanism 2a: Vision-Temporal Agreement Score**

Define a joint objective that balances temporal clustering quality with vision agreement:

```python
def compute_joint_clustering_score(pred_clusters, temporal_scores, vision_clusters,
                                   alpha=0.7):
    """
    Joint score = alpha * temporal_quality + (1-alpha) * vision_agreement

    alpha: Weight for temporal vs. vision
           alpha=1.0 → pure temporal
           alpha=0.0 → pure vision
           alpha=0.7 → temporal-focused with vision regularization
    """
    # Temporal quality: Average conversation score within predicted clusters
    temporal_quality = compute_within_cluster_scores(pred_clusters, temporal_scores)

    # Vision agreement: Normalized mutual information or F1 between pred and vision
    vision_agreement = pairwise_f1_score(
        [vision_clusters[spk] for spk in pred_clusters.keys()],
        [pred_clusters[spk] for spk in pred_clusters.keys()]
    )

    joint_score = alpha * temporal_quality + (1 - alpha) * vision_agreement
    return joint_score


def search_best_clustering_with_vision_regularization(temporal_scores, vision_clusters,
                                                      alpha=0.7):
    """
    Search over clustering hyperparameters to maximize joint score
    """
    best_score = -1
    best_clusters = None

    # Search over thresholds
    for threshold in np.arange(0.5, 0.9, 0.05):
        # Cluster with this threshold
        pred_clusters = cluster_speakers(temporal_scores, list(vision_clusters.keys()),
                                        threshold=threshold)

        # Compute joint score
        score = compute_joint_clustering_score(pred_clusters, temporal_scores,
                                               vision_clusters, alpha)

        if score > best_score:
            best_score = score
            best_clusters = pred_clusters

    return best_clusters, best_score
```

**Pros:**
- Principled combination of both modalities
- Tunable balance via alpha parameter
- Can explore different temporal thresholds guided by vision

**Cons:**
- Computationally expensive (search over hyperparameters)
- Alpha tuning required
- May still make errors if both modalities agree on wrong clustering

---

#### **Mechanism 2b: Disallow High-Disagreement Decisions**

Explicitly veto clustering decisions that strongly contradict vision:

```python
def cluster_with_vision_veto(temporal_distances, speaker_ids, vision_clusters,
                            max_disagreement=0.2):
    """
    Perform temporal clustering but veto decisions that disagree too much with vision

    max_disagreement: Maximum allowed pairwise F1 disagreement
                      (e.g., 0.2 means final F1 vs vision must be >= 0.8)
    """
    # Initial temporal clustering
    pred_clusters = cluster_speakers(temporal_distances, speaker_ids, threshold=0.7)

    # Compute disagreement with vision
    f1 = pairwise_f1_score(
        [vision_clusters[spk] for spk in speaker_ids],
        [pred_clusters[spk] for spk in speaker_ids]
    )
    disagreement = 1 - f1

    if disagreement <= max_disagreement:
        return pred_clusters  # Acceptable disagreement

    # Too much disagreement, iterate to fix
    # Strategy: Identify most disagreeing pairs and force vision clustering
    false_merges, false_splits = identify_disagreements(pred_clusters, vision_clusters)

    # Fix false merges (split to match vision)
    for spk_i, spk_j in false_merges:
        # Force these speakers into different clusters
        # (Implementation: modify distance matrix and re-cluster)
        pass

    # Fix false splits (merge to match vision)
    for spk_i, spk_j in false_splits:
        # Force these speakers into same cluster
        pass

    return refined_clusters
```

**Pros:**
- Hard constraint on maximum error
- Prevents catastrophic disagreements
- Transparent decision (when vision is trusted more)

**Cons:**
- May over-rely on vision even when temporal evidence is strong
- Threshold (max_disagreement) is arbitrary
- Can lead to inconsistent clustering (mix of temporal and vision logic)

---

### **1.3 As Features for Upstream Modules**

#### **Mechanism 3a: Conversation-ID Features for AVSR**

Use vision clustering to provide conversation context to AVSR models:

```python
class AVSRWithConversationContext(BaseInferenceModel):
    """
    AVSR model that receives conversation cluster ID as input feature
    """

    def inference(self, videos, audios, conversation_id=None, **kwargs):
        """
        conversation_id: Integer indicating which conversation this segment belongs to
        """
        # Encode conversation ID as embedding
        if conversation_id is not None:
            conv_embedding = self.conversation_encoder(torch.tensor([conversation_id]))
            # Concatenate with audiovisual features
            audiovisual_feat = torch.cat([audiovisual_feat, conv_embedding.expand(T, -1)], dim=-1)

        # Standard AVSR inference
        nbest_hyps = self.beam_search(audiovisual_feat)
        predicted = self.text_transform.post_process(nbest_hyps[0]["yseq"])
        return predicted


def inference_with_conversation_context(session_dir, vision_clusters):
    """
    Run AVSR inference with conversation context from vision clustering
    """
    for speaker_name, speaker_data in metadata.items():
        # Get conversation ID from vision clustering
        conversation_id = vision_clusters[speaker_name]

        # Run AVSR for this speaker with conversation context
        for track in speaker_data['central']['crops']:
            video_path = track['lip']
            hypotheses = model.infer_video(video_path, conversation_id=conversation_id)
```

**Pros:**
- AVSR can learn conversation-specific patterns (jargon, speaking style)
- Implicitly models conversation context
- May improve WER if conversations have distinct linguistic characteristics

**Cons:**
- Requires retraining AVSR models with conversation ID features
- Benefit unclear (conversations may not have distinct linguistic patterns)
- Moderate implementation effort

---

#### **Mechanism 3b: Speaker Group Embeddings for ASD**

Provide conversation cluster information to ASD module:

```python
def asd_with_conversation_context(video, audio, conversation_id, num_conversations):
    """
    ASD model aware of which conversation the speaker belongs to

    Could help distinguish:
    - Cross-conversation overlap (different conv IDs → more likely simultaneous)
    - Within-conversation overlap (same conv ID → more likely turn-taking)
    """
    conv_onehot = F.one_hot(torch.tensor(conversation_id), num_conversations)

    # Include conversation ID in ASD model input
    # (Requires model retraining)
    asd_score = asd_model(video, audio, conv_onehot)
    return asd_score
```

**Pros:**
- Could improve ASD for multi-conversation scenarios
- Conversation-specific calibration

**Cons:**
- Requires ASD model retraining
- Circular dependency (ASD affects clustering, clustering affects ASD)
- Benefit uncertain

---

## **Part 2: Which Pipeline Components Benefit Most?**

### **Component Impact Analysis**

| Component | Integration Benefit | Likelihood of Improvement | Implementation Effort |
|-----------|---------------------|---------------------------|----------------------|
| **1. Active Speaker Detection** | Low | 10-20% | High (retraining) |
| **2. Face/Landmark/Mouth Crop** | None | 0% | N/A |
| **3. Segmentation & Chunking** | Low | 5-10% | Medium |
| **4. AVSR Model** | Medium | 10-30% | High (retraining) |
| **5. Conversation Clustering** | **HIGH** | **50-100%** | **Low-Medium** |

---

### **Detailed Analysis**

#### **5. Conversation Clustering** ⭐ **HIGHEST BENEFIT**

**Why this benefits most:**
- Vision clustering already achieves F1=0.9 (vs. temporal F1=0.8153)
- Direct replacement or fusion is straightforward
- No retraining required (integration at inference time)
- Addresses the core weakness of time-based clustering

**Concrete benefits:**
- **Immediate:** Use vision clustering directly → F1 = 0.9 (10% absolute improvement)
- **Conservative:** Constrained temporal clustering → F1 ≈ 0.85-0.88
- **Optimal:** Soft fusion → F1 ≈ 0.92-0.95 (if modalities are complementary)

**Failure modes addressed:**
- Turn-taking across conversations: Vision sees they face different groups
- Interruption-heavy: Vision sees they face same group despite overlap
- Speaker bridging: Vision disambiguates by gaze direction

**Implementation path:**
```python
# Option 1: Direct replacement (1 line of code)
clusters = vision_clusters  # Instead of cluster_speakers(temporal_scores)

# Option 2: Soft fusion (20 lines of code)
joint_distances = alpha * temporal_distances + (1-alpha) * vision_distances
clusters = cluster_speakers(joint_distances)

# Option 3: Constrained clustering (50 lines of code)
constrained_distances = apply_vision_constraints(temporal_distances, vision_clusters)
clusters = cluster_speakers(constrained_distances)
```

---

#### **4. AVSR Model** - Medium Benefit

**Potential benefit:**
If conversations have distinct linguistic characteristics:
- **Conversation-specific vocabulary:** Medical conv vs. sports conv
- **Conversation-specific speaking styles:** Formal vs. casual
- **Conversation-specific acoustic conditions:** Different room locations

**How to integrate:**
```python
# Approach 1: Conversation-conditioned AVSR
# Add conversation ID as input feature (requires retraining)
output = avsr_model(video, audio, conversation_id)

# Approach 2: Post-processing with conversation-level language model
# Rescore hypotheses based on conversation context
best_hyp = rescore_with_conversation_lm(nbest_hyps, conversation_id)

# Approach 3: Conversation-specific fine-tuning
# Train separate AVSR heads for each conversation type (if types are recurring)
```

**Expected improvement:**
- **WER reduction:** 2-5% relative (if linguistic patterns differ)
- **Joint error reduction:** 1-2% absolute

**Caveats:**
- Requires conversation-labeled training data
- Benefit unclear for casual conversations (may not have distinct patterns)
- High implementation effort (model retraining)

---

#### **3. Segmentation & Chunking** - Low Benefit

**Possible integration:**
Use vision clustering to refine speaking turn detection:

```python
def conversation_aware_segmentation(asd_scores, conversation_id, other_speakers_convs):
    """
    Adjust ASD thresholds based on conversation context

    Hypothesis: Within same conversation, use lower onset threshold (catch backchannels)
               Across conversations, use higher threshold (avoid false positives)
    """
    if speaker_shares_conversation_with_others(conversation_id, other_speakers_convs):
        # Same conversation: more lenient (expect turn-taking)
        onset = 0.9  # Lower than default 1.0
    else:
        # Different conversation: more strict (avoid cross-talk)
        onset = 1.2  # Higher than default

    segments = segment_by_asd(asd_scores, {'onset': onset})
    return segments
```

**Expected improvement:**
- Marginal (1-2% WER improvement at best)
- Requires per-conversation calibration

---

#### **1. Active Speaker Detection** - Low Benefit

**Possible integration:**
- Conversation-conditioned ASD (as described in 1.3b)
- Cross-conversation overlap suppression

**Challenges:**
- ASD is pretrained on generic data
- Retraining complex and may not generalize
- Circular dependency (ASD → clustering → ASD)

**Expected improvement:**
- Unclear, potentially 5-10% ASD accuracy improvement
- Indirect benefit to clustering

---

### **Summary: Where to Focus Integration Efforts**

**Tier 1 (High ROI):**
- ✅ **Conversation clustering:** Direct fusion of vision and temporal (10-15% F1 improvement)

**Tier 2 (Medium ROI, if resources available):**
- ⚠️ **AVSR post-processing:** Conversation-level language model rescoring (2-5% WER improvement)

**Tier 3 (Low ROI, research exploration):**
- ⏸️ **ASD conversation context:** Conversation-conditioned active speaker detection
- ⏸️ **Segmentation refinement:** Conversation-aware threshold tuning

**Recommendation:** Start with **Tier 1 (clustering fusion)**, evaluate improvement, then consider Tier 2 if needed.

---

## **Part 3: High-Level Integration Schemes**

### **Scheme 1: Hard Prior (Vision-Dominant)**

#### **Description**
Use vision clustering as the **final cluster labels**. Temporal scores serve only as diagnostics or confidence indicators.

#### **Implementation**
```python
def hard_prior_clustering(vision_clusters, temporal_scores):
    """
    Vision clustering is the ground truth; temporal scores used for diagnostics
    """
    final_clusters = vision_clusters.copy()

    # Optional: Flag low-confidence assignments
    for spk, cluster_id in final_clusters.items():
        # Check if temporal evidence supports this assignment
        cluster_members = [s for s, c in final_clusters.items() if c == cluster_id]
        temporal_support = compute_temporal_support(spk, cluster_members, temporal_scores)

        if temporal_support < 0.5:  # Low temporal agreement
            print(f"Warning: {spk} assigned to cluster {cluster_id} by vision, "
                  f"but low temporal support ({temporal_support:.2f})")

    return final_clusters


def compute_temporal_support(speaker, cluster_members, temporal_scores):
    """
    Average temporal score between speaker and cluster members
    High score = low overlap = supports same conversation
    """
    scores = [temporal_scores[speaker, other] for other in cluster_members if other != speaker]
    return np.mean(scores) if scores else 0.0
```

#### **Pros**
- ✅ **Simplest implementation** (1 line: use vision clustering directly)
- ✅ **Best clustering performance** (F1 ≈ 0.9, guaranteed)
- ✅ **No hyperparameters** to tune
- ✅ **Interpretable:** Vision is the primary signal, temporal is secondary

#### **Cons**
- ❌ **Ignores temporal information** (wastes available signal)
- ❌ **Cannot correct vision errors** using temporal evidence
- ❌ **Vision failures propagate** directly to final output
- ❌ **No fusion benefit:** Doesn't leverage complementary nature of modalities

#### **When to Use**
- Vision system is **highly reliable** (F1 > 0.95)
- Temporal clustering is **unreliable** (F1 < 0.7)
- Simplicity and interpretability are critical
- No labeled dev data for tuning fusion parameters

#### **Failure Analysis Needed**
- **Identify vision failure modes:** Which sessions does vision get wrong?
- **Check temporal disagreement:** When does temporal evidence strongly contradict vision?
- **Error analysis:** Are vision errors correctable by temporal information?

**Diagnostic script:**
```python
def analyze_vision_temporal_disagreement(vision_clusters, temporal_clusters):
    """
    Identify sessions where vision and temporal strongly disagree
    """
    vision_f1 = pairwise_f1_score(ground_truth, vision_clusters)
    temporal_f1 = pairwise_f1_score(ground_truth, temporal_clusters)

    # Cases where temporal is better
    if temporal_f1 > vision_f1:
        print(f"⚠️ Vision F1={vision_f1:.3f}, Temporal F1={temporal_f1:.3f}")
        print(f"   Temporal is better! Hard prior would degrade performance.")

    # Analyze pairwise disagreements
    disagreements = get_pairwise_disagreements(vision_clusters, temporal_clusters)
    print(f"Pairs where vision and temporal disagree: {len(disagreements)}")
```

---

### **Scheme 2: Soft Prior (Weighted Fusion)**

#### **Description**
Combine vision-based similarity and temporal overlap similarity into a **joint distance matrix**, then cluster.

#### **Implementation**
```python
def soft_prior_clustering(temporal_scores, vision_clusters, speaker_ids, alpha=0.6):
    """
    Joint distance = alpha * temporal_distance + (1-alpha) * vision_distance

    alpha: Weight for temporal modality
           alpha=1.0 → pure temporal (baseline)
           alpha=0.0 → pure vision
           alpha=0.6 → balanced fusion
    """
    # Convert temporal overlap scores to distances
    temporal_distances = 1 - temporal_scores  # [0, 1]

    # Convert vision clustering to pairwise distances
    vision_distances = compute_vision_distances(vision_clusters, speaker_ids)

    # Weighted fusion
    joint_distances = alpha * temporal_distances + (1 - alpha) * vision_distances

    # Cluster on joint distances
    clustering = AgglomerativeClustering(
        distance_threshold=0.3,  # May need re-tuning
        metric='precomputed',
        linkage='complete'
    )
    cluster_labels = clustering.fit_predict(joint_distances)

    # Convert to speaker_to_cluster format
    final_clusters = {spk: int(label) for spk, label in zip(speaker_ids, cluster_labels)}
    return final_clusters


def compute_vision_distances(vision_clusters, speaker_ids):
    """
    Convert vision cluster assignments to pairwise distance matrix

    Distance = 0 if same cluster, 1 if different cluster
    (Can be softened for uncertainty)
    """
    n = len(speaker_ids)
    distances = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                distances[i, j] = 0.0
            elif vision_clusters[speaker_ids[i]] == vision_clusters[speaker_ids[j]]:
                distances[i, j] = 0.0  # Same cluster in vision
            else:
                distances[i, j] = 1.0  # Different cluster in vision

    return distances
```

#### **Advanced: Uncertainty-Weighted Fusion**

```python
def soft_prior_with_uncertainty(temporal_scores, vision_clusters, vision_confidence,
                               speaker_ids, alpha=0.6):
    """
    Weight fusion by confidence in each modality

    vision_confidence: Per-pair confidence from vision system (e.g., gaze stability)
    temporal_confidence: Derived from ASD score variance
    """
    temporal_distances = 1 - temporal_scores
    vision_distances = compute_vision_distances(vision_clusters, speaker_ids)

    # Compute temporal confidence (inverse of score variance)
    temporal_confidence = compute_temporal_confidence(temporal_scores)

    # Adaptive weighting per pair
    joint_distances = np.zeros_like(temporal_distances)
    for i in range(len(speaker_ids)):
        for j in range(len(speaker_ids)):
            # Normalize confidences
            total_conf = temporal_confidence[i,j] + vision_confidence[i,j]
            w_temporal = temporal_confidence[i,j] / total_conf
            w_vision = vision_confidence[i,j] / total_conf

            joint_distances[i,j] = (w_temporal * temporal_distances[i,j] +
                                   w_vision * vision_distances[i,j])

    # Cluster
    return cluster_on_distances(joint_distances)
```

#### **Pros**
- ✅ **Leverages both modalities:** Fusion can outperform either alone
- ✅ **Complementary strengths:** Vision handles turn-taking, temporal handles interruptions
- ✅ **Tunable balance:** Alpha parameter controls modality weighting
- ✅ **Robust to single-modality failures:** If one modality is wrong, other can compensate

#### **Cons**
- ❌ **Hyperparameter tuning required:** Need to find optimal alpha
- ❌ **Distance threshold may need adjustment:** Joint distances have different scale
- ❌ **Hard to interpret:** What does distance=0.4 mean in joint space?
- ❌ **Requires vision distance conversion:** May lose information

#### **When to Use**
- Both modalities have comparable reliability (both F1 > 0.8)
- Modalities make **complementary errors** (vision fails where temporal succeeds, and vice versa)
- Labeled dev data available for alpha tuning
- Want to maximize clustering performance

#### **Failure Analysis Needed**
- **Complementarity analysis:** Do modalities make independent errors?
- **Alpha sensitivity:** How does performance vary with alpha?
- **Error analysis:** Which errors does fusion fix vs. introduce?

**Complementarity check:**
```python
def check_complementarity(vision_clusters, temporal_clusters, ground_truth):
    """
    Check if vision and temporal make complementary errors
    """
    vision_errors = get_error_pairs(vision_clusters, ground_truth)
    temporal_errors = get_error_pairs(temporal_clusters, ground_truth)

    # Errors unique to each modality
    vision_only_errors = vision_errors - temporal_errors
    temporal_only_errors = temporal_errors - vision_errors
    shared_errors = vision_errors & temporal_errors

    print(f"Vision-only errors: {len(vision_only_errors)}")
    print(f"Temporal-only errors: {len(temporal_only_errors)}")
    print(f"Shared errors: {len(shared_errors)}")

    complementarity = (len(vision_only_errors) + len(temporal_only_errors)) / len(vision_errors | temporal_errors)
    print(f"Complementarity score: {complementarity:.2%}")

    if complementarity > 0.5:
        print("✅ High complementarity: Fusion likely to help")
    else:
        print("⚠️ Low complementarity: Fusion benefit unclear")
```

**Alpha tuning:**
```python
def tune_fusion_alpha(vision_clusters_dev, temporal_scores_dev, ground_truth_dev):
    """
    Grid search for optimal alpha on dev set
    """
    best_alpha = None
    best_f1 = 0

    for alpha in np.arange(0.0, 1.05, 0.05):
        # Fuse with this alpha
        fused_clusters = soft_prior_clustering(temporal_scores_dev, vision_clusters_dev,
                                              alpha=alpha)

        # Evaluate
        f1 = pairwise_f1_score(ground_truth_dev, fused_clusters)

        print(f"Alpha={alpha:.2f}: F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_alpha = alpha

    print(f"\nBest alpha: {best_alpha:.2f}, F1={best_f1:.4f}")
    return best_alpha
```

---

### **Scheme 3: Two-Stage (Vision-then-Temporal Refinement)**

#### **Description**
1. **Stage 1 (Coarse):** Cluster using vision (high recall, may have errors)
2. **Stage 2 (Refinement):** Within each vision cluster, use temporal clustering to:
   - **Split** if temporal evidence shows distinct conversations
   - **Merge** adjacent vision clusters if temporal evidence is strong

#### **Implementation**

```python
def two_stage_clustering(vision_clusters, temporal_scores, speaker_ids,
                        split_threshold=0.6, merge_threshold=0.85):
    """
    Stage 1: Vision clustering (coarse)
    Stage 2: Temporal refinement (fine-grained)

    split_threshold: If temporal score < 0.6 within vision cluster → split
    merge_threshold: If temporal score > 0.85 across vision clusters → merge
    """
    refined_clusters = vision_clusters.copy()

    # Stage 2a: Check for splits within vision clusters
    for vision_cluster_id in set(vision_clusters.values()):
        cluster_members = [spk for spk, cid in vision_clusters.items()
                          if cid == vision_cluster_id]

        if len(cluster_members) < 2:
            continue  # Can't split singleton

        # Extract temporal scores for this cluster
        member_indices = [speaker_ids.index(spk) for spk in cluster_members]
        submatrix = temporal_scores[np.ix_(member_indices, member_indices)]

        # Check if subgroup has low temporal scores (high overlap → different convs)
        min_score = np.min(submatrix[submatrix > 0])  # Exclude diagonal

        if min_score < split_threshold:
            # Split this vision cluster using temporal clustering
            sub_distances = 1 - submatrix
            sub_clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.4,
                metric='precomputed',
                linkage='complete'
            )
            sub_labels = sub_clustering.fit_predict(sub_distances)

            # Assign new cluster IDs
            max_cluster_id = max(refined_clusters.values())
            for idx, sub_label in enumerate(sub_labels):
                if sub_label > 0:  # Keep first subgroup in original cluster
                    spk = cluster_members[idx]
                    refined_clusters[spk] = max_cluster_id + sub_label

            print(f"Split vision cluster {vision_cluster_id} into "
                  f"{len(set(sub_labels))} sub-clusters (low temporal scores)")

    # Stage 2b: Check for merges across vision clusters
    vision_cluster_ids = sorted(set(refined_clusters.values()))

    for i in range(len(vision_cluster_ids)):
        for j in range(i+1, len(vision_cluster_ids)):
            cluster_a = vision_cluster_ids[i]
            cluster_b = vision_cluster_ids[j]

            members_a = [spk for spk, cid in refined_clusters.items() if cid == cluster_a]
            members_b = [spk for spk, cid in refined_clusters.items() if cid == cluster_b]

            # Average temporal score between clusters
            scores = []
            for spk_a in members_a:
                for spk_b in members_b:
                    idx_a = speaker_ids.index(spk_a)
                    idx_b = speaker_ids.index(spk_b)
                    scores.append(temporal_scores[idx_a, idx_b])

            avg_score = np.mean(scores)

            if avg_score > merge_threshold:
                # Merge cluster_b into cluster_a
                for spk in members_b:
                    refined_clusters[spk] = cluster_a

                print(f"Merged vision clusters {cluster_a} and {cluster_b} "
                      f"(high temporal score: {avg_score:.3f})")

    return refined_clusters
```

#### **Pros**
- ✅ **Best of both worlds:** Vision provides structure, temporal refines
- ✅ **Hierarchical:** Matches natural clustering process (coarse-to-fine)
- ✅ **Interpretable:** Each stage has clear purpose
- ✅ **Conservative:** Only changes vision clustering when temporal evidence is strong
- ✅ **Can fix both types of vision errors:** Over-clustering (via merges) and under-clustering (via splits)

#### **Cons**
- ❌ **Two thresholds to tune:** split_threshold and merge_threshold
- ❌ **Asymmetric:** Treats vision as primary, temporal as secondary
- ❌ **May not converge:** Splitting and merging could conflict
- ❌ **Complexity:** More moving parts than hard or soft priors

#### **When to Use**
- Vision clustering is **mostly correct** but has occasional errors
- Temporal clustering can **reliably identify errors** (high precision, may have lower recall)
- Want **conservative refinement** (only change when confident)
- Interpretability is important (can explain why each change was made)

#### **Failure Analysis Needed**
- **Vision error patterns:** Does vision tend to over-cluster or under-cluster?
- **Temporal refinement accuracy:** What % of splits/merges are correct?
- **Threshold sensitivity:** How do split/merge thresholds affect results?

**Threshold tuning:**
```python
def tune_two_stage_thresholds(vision_clusters_dev, temporal_scores_dev, ground_truth_dev):
    """
    Grid search for optimal split and merge thresholds
    """
    best_f1 = 0
    best_params = None

    for split_thresh in [0.5, 0.6, 0.65, 0.7]:
        for merge_thresh in [0.8, 0.85, 0.9, 0.95]:
            if merge_thresh <= split_thresh:
                continue  # Invalid (would cause conflicts)

            # Run two-stage clustering
            refined = two_stage_clustering(vision_clusters_dev, temporal_scores_dev,
                                          split_threshold=split_thresh,
                                          merge_threshold=merge_thresh)

            # Evaluate
            f1 = pairwise_f1_score(ground_truth_dev, refined)

            print(f"Split={split_thresh:.2f}, Merge={merge_thresh:.2f}: F1={f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_params = (split_thresh, merge_thresh)

    print(f"\nBest params: split={best_params[0]:.2f}, merge={best_params[1]:.2f}, F1={best_f1:.4f}")
    return best_params
```

**Refinement analysis:**
```python
def analyze_refinements(vision_clusters, refined_clusters, ground_truth):
    """
    Analyze which splits/merges were correct
    """
    # Identify what changed
    splits = []
    merges = []

    for spk_i in vision_clusters.keys():
        for spk_j in vision_clusters.keys():
            if spk_i >= spk_j:
                continue

            vision_same = (vision_clusters[spk_i] == vision_clusters[spk_j])
            refined_same = (refined_clusters[spk_i] == refined_clusters[spk_j])
            gt_same = (ground_truth[spk_i] == ground_truth[spk_j])

            if vision_same and not refined_same:
                # Split occurred
                is_correct = not gt_same
                splits.append((spk_i, spk_j, is_correct))

            elif not vision_same and refined_same:
                # Merge occurred
                is_correct = gt_same
                merges.append((spk_i, spk_j, is_correct))

    # Statistics
    split_correct = sum(1 for s in splits if s[2])
    merge_correct = sum(1 for m in merges if m[2])

    print(f"Splits: {len(splits)} total, {split_correct} correct ({100*split_correct/len(splits):.1f}%)")
    print(f"Merges: {len(merges)} total, {merge_correct} correct ({100*merge_correct/len(merges):.1f}%)")

    # Overall impact
    vision_f1 = pairwise_f1_score(ground_truth, vision_clusters)
    refined_f1 = pairwise_f1_score(ground_truth, refined_clusters)

    print(f"\nVision F1: {vision_f1:.4f}")
    print(f"Refined F1: {refined_f1:.4f}")
    print(f"Improvement: {refined_f1 - vision_f1:+.4f}")
```

---

## **Part 4: Scheme Comparison and Recommendations**

### **Performance Expectations**

| Scheme | Expected F1 | Baseline Improvement | Implementation Effort | Tuning Required |
|--------|-------------|----------------------|----------------------|-----------------|
| **Hard Prior** | 0.90 | +0.08 | **Low** (5 lines) | None |
| **Soft Prior** | 0.92-0.95 | +0.10-0.13 | **Medium** (50 lines) | Alpha (1 param) |
| **Two-Stage** | 0.91-0.93 | +0.09-0.11 | **Medium** (100 lines) | Split/Merge (2 params) |
| **Baseline (temporal only)** | 0.8153 | - | - | - |

### **Decision Matrix**

#### **Choose Hard Prior if:**
- ✅ Vision system is very reliable (F1 > 0.95)
- ✅ Need simple, interpretable baseline
- ✅ No labeled dev data for tuning
- ✅ Implementation time is critical

#### **Choose Soft Prior if:**
- ✅ Want to maximize performance
- ✅ Both modalities have comparable quality (0.8 < F1 < 0.95)
- ✅ Have labeled dev data for alpha tuning
- ✅ Modalities make complementary errors

#### **Choose Two-Stage if:**
- ✅ Vision is good but not perfect (F1 ≈ 0.85-0.92)
- ✅ Want interpretable refinements
- ✅ Need to explain why clustering changed
- ✅ Conservative approach preferred (only change when confident)

---

### **Recommended Integration Path**

**Phase 1: Quick Win (Week 1)**
```python
# Implement Hard Prior
final_clusters = vision_clusters  # 1 line!

# Evaluate
f1 = pairwise_f1_score(ground_truth, final_clusters)
print(f"Hard prior F1: {f1:.4f}")  # Expected: ~0.90

# Analyze disagreements
analyze_vision_temporal_disagreement(vision_clusters, temporal_clusters)
```

**Phase 2: Complementarity Analysis (Week 2)**
```python
# Check if modalities make complementary errors
check_complementarity(vision_clusters, temporal_clusters, ground_truth)

# If complementarity > 0.5, proceed to fusion
# If complementarity < 0.3, stick with hard prior
```

**Phase 3: Fusion (Week 3-4, if justified)**
```python
# If complementarity is high, implement soft prior
best_alpha = tune_fusion_alpha(vision_dev, temporal_dev, gt_dev)
fused_clusters = soft_prior_clustering(temporal_scores, vision_clusters,
                                      alpha=best_alpha)

# Or implement two-stage
best_params = tune_two_stage_thresholds(vision_dev, temporal_dev, gt_dev)
refined_clusters = two_stage_clustering(vision_clusters, temporal_scores,
                                       split_threshold=best_params[0],
                                       merge_threshold=best_params[1])

# Compare
print(f"Hard prior F1: {hard_f1:.4f}")
print(f"Soft prior F1: {soft_f1:.4f}")
print(f"Two-stage F1: {two_stage_f1:.4f}")
```

---

### **Integration with Full Pipeline**

Once clustering is improved, propagate benefits to joint error rate:

```python
def compute_joint_error_with_vision_clustering(session_dir, vision_clusters):
    """
    Use vision-based clustering to improve joint ASR-clustering error
    """
    # Run AVSR (unchanged)
    speaker_wer = run_avsr_and_compute_wer(session_dir)

    # Use vision clustering instead of temporal
    # (Or fused clustering)
    final_clusters = vision_clusters  # or soft_prior_clustering(...)

    # Evaluate clustering
    gt_clusters = load_clustering(f"{session_dir}/labels/speaker_to_cluster.json")
    speaker_clustering_f1 = get_speaker_clustering_f1_score(final_clusters, gt_clusters)

    # Compute joint error
    joint_error = {}
    for speaker in speaker_wer.keys():
        joint_error[speaker] = 0.5 * speaker_wer[speaker] + 0.5 * (1 - speaker_clustering_f1[speaker])

    return joint_error
```

**Expected impact on joint error rate:**

| Clustering F1 | Joint Error (if WER=0.50) | Joint Error (if WER=0.55) |
|---------------|---------------------------|---------------------------|
| 0.8153 (baseline) | 0.3424 | 0.3674 |
| 0.90 (hard prior) | 0.3000 | 0.3250 |
| 0.93 (soft prior) | 0.2850 | 0.3100 |

**Improvement:** 4-6% absolute reduction in joint error rate

---

## **Summary and Key Takeaways**

### **Main Insights**

1. **Conversation clustering is the highest-ROI integration point**
   - Direct impact on primary metric (pairwise F1)
   - Low implementation effort
   - Guaranteed improvement (vision F1 > baseline F1)

2. **Three viable integration schemes:**
   - **Hard prior:** Simplest, immediate ~10% F1 improvement
   - **Soft prior:** Best performance if modalities are complementary
   - **Two-stage:** Best interpretability and conservative refinement

3. **Complementarity is key to fusion benefit:**
   - If vision and temporal make **independent errors** → fusion helps significantly
   - If they make **same errors** → fusion provides marginal benefit
   - **Must measure complementarity** before investing in complex fusion

4. **AVSR integration is secondary:**
   - Requires model retraining (high effort)
   - Benefit depends on whether conversations have distinct linguistic patterns
   - Recommend focusing on clustering first, AVSR later if needed

### **Action Items**

**Immediate (Week 1):**
- [ ] Implement hard prior (use vision clustering directly)
- [ ] Measure F1 improvement on dev set
- [ ] Analyze vision-temporal disagreements

**Short-term (Week 2-3):**
- [ ] Check modality complementarity
- [ ] If complementary, implement soft prior or two-stage
- [ ] Tune fusion parameters on dev set

**Long-term (Month 2+, if justified):**
- [ ] Explore AVSR with conversation context
- [ ] Conversation-specific language model rescoring
- [ ] ASD with conversation awareness

---

## **Document Information**

- **Generated**: 2025-11-18
- **Repository**: https://github.com/aziz-hakiri/mcorec_baseline
- **Analysis Type**: Multi-Modal Clustering Fusion Strategies
- **Purpose**: Conceptual integration of vision-based and time-based clustering
