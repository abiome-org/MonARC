# Training Pipeline and Loss Formulations

Date: 2026-08-30  
Status: Training Protocol Specification  
Repository: [abiome-org/MonARC](https://github.com/abiome-org/MonARC)  

---

## 1. Strictly Sequential Three-Stage Training Pipeline

MonARC enforces a strictly staged, decoupled training pipeline. End-to-end backpropagation across the entire system is architecturally prohibited.

**v1 does not walk CONUS, does not pretrain a foundation model, and does not require a custom flight campaign.** Inference coverage is **Colorado**. Cost law: [`docs/cost.md`](./cost.md).

```
+===================================================================================================+
| STAGE 1: SAMPLED CODEBOOK + COLORADO INFERENCE                                                     |
| Train: Fusion Stem theta_geo + Fusion MLP theta_fuse + FSQ Projection theta_fsq                     |
|        on a *sampled* diverse tile set (multiple biomes; tiny versus CONUS)                        |
| Infer: Frozen stack on Colorado (Jefferson County / Front Range may be a first slice)           |
| Frozen: Pretrained DINOv2 RGB backbone. No foundation-model pretrain.                               |
| Export: FSQ codes + inverted metric index for Colorado. Not a CONUS fp16 / Zarr grid.           |
+===================================================================================================+
                                                  |
                                                  v  (Freeze Stage 1 Weights & Codebook)
+===================================================================================================+
| STAGE 2: PERSPECTIVE ENCODER CROSS-VIEW ALIGNMENT & CONFIDENCE CALIBRATION                         |
| Data: Public thin pairs only — University-1652, DenseUAV, SUES-200, OrthoLoC                     |
| Active Parameters: Perspective Cross-View Adapter psi_persp + Confidence Head psi_conf           |
| Frozen: Stage 1 Ingestion Network, DINOv2 Backbone, FSQ Codebook Quantizer                        |
| Objective: InfoNCE cross-view alignment to Phi_map + calibrated confidence                        |
| Out of v1: Custom flight-log campaigns. Photorealistic renderer / 3DGS / GISNav / Unreal         |
|            unless alignment on those four public sets actually fails.                             |
+===================================================================================================+
                                                  |
                                                  v  (Freeze Stage 2 Perception Weights)
+===================================================================================================+
| STAGE 3: ACTIVE PERCEPTION POLICY (HUNTER) OPTIMIZATION IN FRUSTUM GYM                            |
| Data: Abstract 2.5D Frustum Gym with Randomized Landmark Fields & Code Collisions                |
| Active Parameters: Hunter Mode-Attention Transformer Policy phi_hunter                            |
| Frozen: Entire Perception & State Estimation Stack (Stages 1 & 2)                                 |
| Compute: CPU; millions of episodes on a laptop or workstation. No MSFS / Unreal / Unity.        |
| Optimization: Trajectory rollouts via MPPI/CEM on expected entropy drop -> Supervised Distillation|
+===================================================================================================+
```

---

## 2. Stage 1: Sampled Codebook Training and Colorado Inference

Stage 1 is **not** a continental feature-field pretrain. It has two sequential jobs:

1. **Train** the fusion stem and FSQ on a sampled diverse tile set (multiple biomes). The sample is small relative to CONUS. Do not iterate every NAIP tile in the United States.
2. **Infer** the frozen encoder + FSQ on **Colorado**. A Jefferson County slice may run first. Persist FSQ codes and the inverted metric index. Landmarks are emergent extrema, not every DINOv2 token and not hardcoded geology/landcover classes. Do not write a dense CONUS fp16 / Zarr field.

DINOv2 remains frozen. Do not train a new foundation visual backbone.

### 2.1 Forward Formulation
Let \( \mathbf{I}_{\mathrm{rgb}} \in \mathbb{R}^{B \times 3 \times H \times W} \) be the orthophoto patch, \( \mathbf{D} \in \mathbb{R}^{B \times 1 \times H \times W} \) be the aligned 3DEP elevation raster, and \( \mathbf{V} \in \mathbb{R}^{B \times C_g \times H \times W} \) be the rasterized vector geometry.

1. **RGB Feature Extraction (Frozen)**:
   \[
   \mathbf{f}_{\mathrm{rgb}} = \mathrm{DINOv2}(\mathbf{I}_{\mathrm{rgb}}) \in \mathbb{R}^{B \times D_v \times H' \times W'} \quad (D_v = 768, \, H'=H/14, \, W'=W/14)
   \]
2. **Elevation & Vector Geometry Fusion (Trainable)**:
   \[
   \mathbf{f}_{\mathrm{geo}} = \mathrm{Stem}_{\theta_{\mathrm{geo}}}([\mathbf{D}, \mathbf{V}]) \in \mathbb{R}^{B \times D_g \times H' \times W'}
   \]
3. **Multi-Modal Feature Synthesis**:
   \[
   \mathbf{z}_{\mathrm{fused}} = \mathrm{MLP}_{\theta_{\mathrm{fuse}}}([\mathbf{f}_{\mathrm{rgb}}, \mathbf{f}_{\mathrm{geo}}]) \in \mathbb{R}^{B \times D_f \times H' \times W'}
   \]
4. **Finite Scalar Quantization (Deterministic)**:
   Given quantization level tuple \( L = (l_1, l_2, \dots, l_d) \) yielding \( K = \prod_{i=1}^d l_i \) discrete codes, the continuous representation \( \mathbf{z}_{\mathrm{fused}} \) is projected to \( \mathbb{R}^d \) and quantized via FSQ:
   \[
   \hat{\mathbf{z}} = \mathrm{FSQ}(\mathbf{z}_{\mathrm{fused}})
   \]

### 2.2 Loss Formulation for Stage 1
Stage 1 is optimized using a dual-objective loss:
\[
\mathcal{L}_{\mathrm{Stage1}} = \mathcal{L}_{\mathrm{recon}} + \lambda_{\mathrm{smooth}} \mathcal{L}_{\mathrm{spatial}}
\]
- **Reconstruction Loss**: Reconstructing multi-scale patch appearance and geometric normals from quantized codes:
  \[
  \mathcal{L}_{\mathrm{recon}} = \|\mathbf{f}_{\mathrm{rgb}} - \mathrm{Decoder}_{\theta_{\mathrm{dec}}}(\hat{\mathbf{z}})\|_2^2 + \|\nabla \mathbf{D} - \mathrm{NormDecoder}_{\theta_{\mathrm{norm}}}(\hat{\mathbf{z}})\|_2^2
  \]
- **Spatial Smoothness Regularizer**: Prevents high-frequency checkerboard quantization artifacts across adjacent spatial patches:
  \[
  \mathcal{L}_{\mathrm{spatial}} = \sum_{i,j} \|\hat{\mathbf{z}}_{i+1,j} - \hat{\mathbf{z}}_{i,j}\|_2^2 + \|\hat{\mathbf{z}}_{i,j+1} - \hat{\mathbf{z}}_{i,j}\|_2^2
  \]

Query field \( \Phi_{\mathrm{map}} \) used in Stage 2 is the **Colorado** representation: interpolated from FSQ codes / a working-set grid, not a continental dense store.

---

## 3. Stage 2: Perspective Encoder Alignment & Confidence Calibration

v1 Stage 2 trains **only** on University-1652, DenseUAV, SUES-200, and OrthoLoC. Do not require proprietary flight logs. Do not stand up GISNav, 3DGS, Unreal, or a photorealistic renderer unless those four sets actually fail to support alignment.

### 3.1 Cross-View Contrastive Alignment (InfoNCE)
Given an oblique drone camera frame \( \mathbf{I}_{\mathrm{persp}} \) with calibrated camera intrinsics \( \mathbf{K} \) and true 6-DoF pose \( T^* \), each perspective patch \( p \) has a ray intersection \( \mathbf{x}_p^* = (x, y, z)_p^* \) on the 3DEP ground surface (or the bench's supplied DSM).

1. Compute perspective token \( \mathbf{z}_p = g_{\psi_{\mathrm{persp}}}(\mathrm{DINOv2}(\mathbf{I}_{\mathrm{persp}})_p) \).
2. Query the aerial field \( \Phi_{\mathrm{map}}(\mathbf{x}_p^*) \) (Colorado interpolated field or the bench's paired geodata field).
3. Compute the multi-negative InfoNCE contrastive loss:
   \[
   \mathcal{L}_{\mathrm{align}} = -\frac{1}{N_p} \sum_{p=1}^{N_p} \log \frac{\exp\left( \langle \mathbf{z}_p, \Phi_{\mathrm{map}}(\mathbf{x}_p^*) \rangle / \tau \right)}{\exp\left( \langle \mathbf{z}_p, \Phi_{\mathrm{map}}(\mathbf{x}_p^*) \rangle / \tau \right) + \sum_{k=1}^{N_{\mathrm{neg}}} \exp\left( \langle \mathbf{z}_p, \Phi_{\mathrm{map}}(\mathbf{x}_{p,k}^{\mathrm{neg}}) \rangle / \tau \right)}
   \]
   where negative spatial locations \( \mathbf{x}_{p,k}^{\mathrm{neg}} \) are sampled using a hard-negative mining strategy within local spatial neighborhoods.

### 3.2 Confidence Calibration Loss
The confidence head \( c_{\psi_{\mathrm{conf}}}(\mathbf{z}_p) \) predicts whether the correspondence \( (c_p, \mathbf{u}_p) \) will yield a reprojection error below a geometric inlier threshold hyperparameter \( \epsilon_{\mathrm{reproj}} \) (a training target definition swept during development, e.g., \( \epsilon_{\mathrm{reproj}} \approx 5.0 \text{ px} \)) under the ground-truth pose:
\[
y_p = \mathbb{I}\left( \|\mathbf{u}_p - \pi(T^*, \mathbf{x}_p^*)\|_2 \le \epsilon_{\mathrm{reproj}} \right)
\]
The confidence parameter is trained via binary cross-entropy with temperature scaling:
\[
\mathcal{L}_{\mathrm{conf}} = -\sum_{p=1}^{N_p} \left[ y_p \log c_p + (1 - y_p) \log(1 - c_p) \right]
\]

Total Stage 2 Loss:
\[
\mathcal{L}_{\mathrm{Stage2}} = \mathcal{L}_{\mathrm{align}} + \lambda_{\mathrm{conf}} \mathcal{L}_{\mathrm{conf}}
\]

---

## 4. Stage 3: Hunter Policy Optimization in Frustum Gym

Stage 3 stays a **CPU** frustum gym. Millions of episodes on a laptop or workstation. Microsoft Flight Simulator, Unreal Engine, Unity, and other game-engine visual gyms remain forbidden.

### 4.1 Information-Gain MPPI Trajectory Optimizer
In the abstract 2.5D frustum gym, the expert policy generates control sequences \( \mathbf{U} = (a_0, a_1, \dots, a_{H-1}) \) across horizon \( H \) using Model Predictive Path Integral (MPPI) control:
1. Sample candidate control trajectories:
   \[
   a_t^{(m)} \sim \mathcal{N}(\mu_t^{(k)}, \Sigma_t)
   \]
2. Evaluate trajectory reward based strictly on pose posterior Shannon entropy reduction:
   \[
   R(\tau^{(m)}) = \sum_{t=0}^{H-1} \left( H(p(T_t^{(m)})) - H(p(T_{t+1}^{(m)})) \right) - \lambda_{\mathrm{act}} \|a_t^{(m)}\|_2^2 + \mathbb{I}(\mathrm{Converged}) \cdot R_{\mathrm{term}}
   \]
   where \( R_{\mathrm{term}} \) is a bounded constant reward for achieving unimodal posterior concentration (\( H(p(T)) < H_{\mathrm{threshold}} \)).
3. Update distribution mean:
   \[
   \mu_t^{(k+1)} = \frac{\sum_{m=1}^M \exp\left( \frac{1}{\lambda} R(\tau^{(m)}) \right) a_t^{(m)}}{\sum_{m=1}^M \exp\left( \frac{1}{\lambda} R(\tau^{(m)}) \right)}
   \]

### 4.2 Supervised Behavioral Cloning into Tiny Transformer
The converged MPPI control actions \( a_0^* \) are recorded across randomized gym episodes. The Hunter transformer policy \( \pi_\phi(a \mid s) \) is trained via supervised regression:
\[
\mathcal{L}_{\mathrm{Hunter}} = \mathbb{E}_{(s, a^*)}\left[ \| \pi_\phi(s) - a^* \|_2^2 \right]
\]
where input state \( s = [ H(p(T_t)), \Delta \boldsymbol{\mu}_{\mathrm{modes}}, \mathbf{S}_{\mathrm{rim}} ] \).

---

## 5. Parameter Freeze Schedule

| Training Stage | DINOv2 RGB Backbone | Fusion Stem | FSQ Quantizer | Perspective Adapter | Where-Am-I Perceiver | Hunter Policy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 1: Sampled codebook + Colorado infer** | **FROZEN** | TRAINABLE | TRAINABLE | N/A | N/A | N/A |
| **Stage 2: Public-bench cross-view** | **FROZEN** | **FROZEN** | **FROZEN** | TRAINABLE | TRAINABLE | N/A |
| **Stage 3: CPU frustum gym Hunter**| **FROZEN** | **FROZEN** | **FROZEN** | **FROZEN** | **FROZEN** | TRAINABLE |

This freeze schedule prevents catastrophic forgetting of the visual codebook and guarantees that the Colorado geodata index remains stationary after Stage 1 inference.

---

## 6. First executable increment

`monarc dry-run` runs a few CPU steps of Stage 1 (fusion + FSQ projection; frozen DINO-contract stub). `monarc extract` + `monarc train-fsq` is the GPU path: frozen DINOv2-B/14 (`dinov2_vitb14`) on RGB chips, then FSQ on cached features with periodic checkpoints. Stage 2 public-UAV alignment uses the University-1652 loader when a local tree is present (optional zip URL download). Stage 3 Hunter/MPPI is not trained in this increment. The v0 pose path is matcher + PnP/LM (`monarc.localization.dpnp`), not Perceiver regression.
