# Training Pipeline and Loss Formulations

Date: 2026-08-28  
Status: Training Protocol Specification  
Repository: [abiome-org/MonARC](https://github.com/abiome-org/MonARC)  

---

## 1. Strictly Sequential Three-Stage Training Pipeline

MonARC enforces a strictly staged, decoupled training pipeline. End-to-end backpropagation across the entire system is architecturally prohibited.

```
+===================================================================================================+
| STAGE 1: 2D MASS CODEBOOK & CONTINUOUS FEATURE FIELD PRETRAINING                                  |
| Data: Continental 2D/2.5D Mass Geodata (NAIP + 3DEP DSM + Overture/OSM Rasters)                  |
| Active Parameters: Fusion Stem theta_geo + Fusion MLP theta_fuse + FSQ Projection theta_fsq       |
| Frozen: Pretrained DINOv2 Foundation Backbone (RGB)                                               |
| Objective: Multi-scale spatial feature consistency + deterministic FSQ codebook quantization     |
+===================================================================================================+
                                                  |
                                                  v  (Freeze Stage 1 Weights & Codebook)
+===================================================================================================+
| STAGE 2: PERSPECTIVE ENCODER CROSS-VIEW ALIGNMENT & CONFIDENCE CALIBRATION                         |
| Data: Thin Perspective Pairs (University-1652, DenseUAV, SUES-200, OrthoLoC, Real Flight Logs)    |
| Active Parameters: Perspective Cross-View Adapter psi_persp + Confidence Head psi_conf           |
| Frozen: Stage 1 Ingestion Network, DINOv2 Backbone, FSQ Codebook Quantizer                        |
| Objective: InfoNCE cross-view feature alignment to continuous field Phi_map + calibrated conf     |
+===================================================================================================+
                                                  |
                                                  v  (Freeze Stage 2 Perception Weights)
+===================================================================================================+
| STAGE 3: ACTIVE PERCEPTION POLICY (HUNTER) OPTIMIZATION IN FRUSTUM GYM                            |
| Data: Abstract 2.5D Frustum Gym with Randomized Landmark Fields & Code Collisions                |
| Active Parameters: Hunter Mode-Attention Transformer Policy phi_hunter                            |
| Frozen: Entire Perception & State Estimation Stack (Stages 1 & 2)                                 |
| Optimization: Trajectory rollouts via MPPI/CEM on expected entropy drop -> Supervised Distillation|
+===================================================================================================+
```

---

## 2. Stage 1: Continuous Field and FSQ Codebook Training

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
   Given quantization level tuple \( L = (8, 8, 8, 5, 5) \) yielding \( K = 12,800 \) discrete codes, the continuous representation \( \mathbf{z}_{\mathrm{fused}} \) is projected to \( \mathbb{R}^5 \) and quantized via FSQ:
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

---

## 3. Stage 2: Perspective Encoder Alignment & Confidence Calibration

### 3.1 Cross-View Contrastive Alignment (InfoNCE)
Given an oblique drone camera frame \( \mathbf{I}_{\mathrm{persp}} \) with calibrated camera intrinsics \( \mathbf{K} \) and true 6-DoF pose \( T^* \), each perspective patch \( p \) has a ray intersection \( \mathbf{x}_p^* = (x, y, z)_p^* \) on the 3DEP ground surface.

1. Compute perspective token \( \mathbf{z}_p = g_{\psi_{\mathrm{persp}}}(\mathrm{DINOv2}(\mathbf{I}_{\mathrm{persp}})_p) \).
2. Query the continuous aerial field \( \Phi_{\mathrm{map}}(\mathbf{x}_p^*) \).
3. Compute the multi-negative InfoNCE contrastive loss:
   \[
   \mathcal{L}_{\mathrm{align}} = -\frac{1}{N_p} \sum_{p=1}^{N_p} \log \frac{\exp\left( \langle \mathbf{z}_p, \Phi_{\mathrm{map}}(\mathbf{x}_p^*) \rangle / \tau \right)}{\exp\left( \langle \mathbf{z}_p, \Phi_{\mathrm{map}}(\mathbf{x}_p^*) \rangle / \tau \right) + \sum_{k=1}^{N_{\mathrm{neg}}} \exp\left( \langle \mathbf{z}_p, \Phi_{\mathrm{map}}(\mathbf{x}_{p,k}^{\mathrm{neg}}) \rangle / \tau \right)}
   \]
   where negative spatial locations \( \mathbf{x}_{p,k}^{\mathrm{neg}} \) are sampled using a hard-negative mining strategy within a 5 km radius.

### 3.2 Confidence Calibration Loss
The confidence head \( c_{\psi_{\mathrm{conf}}}(\mathbf{z}_p) \) predicts whether the correspondence \( (c_p, \mathbf{u}_p) \) will yield a reprojection error below a geometric inlier threshold \( \epsilon_{\mathrm{reproj}} = 5.0 \text{ px} \) under the ground-truth pose:
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

### 4.1 Information-Gain MPPI Trajectory Optimizer
In the abstract 2.5D frustum gym, the expert policy generates control sequences \( \mathbf{U} = (a_0, a_1, \dots, a_{H-1}) \) across horizon \( H = 12 \) using Model Predictive Path Integral (MPPI) control:
1. Sample \( M = 2048 \) candidate control trajectories:
   \[
   a_t^{(m)} \sim \mathcal{N}(\mu_t^{(k)}, \Sigma_t)
   \]
2. Evaluate trajectory reward based strictly on pose posterior Shannon entropy reduction:
   \[
   R(\tau^{(m)}) = \sum_{t=0}^{H-1} \left( H(p(T_t^{(m)})) - H(p(T_{t+1}^{(m)})) \right) - \lambda_{\mathrm{act}} \|a_t^{(m)}\|_2^2 + \mathbb{I}(\mathrm{Converged}) \cdot R_{\mathrm{term}}
   \]
   where \( R_{\mathrm{term}} \) is a bounded constant reward (\( \le 2.0 \)) for achieving unimodal posterior concentration (\( H(p(T)) < H_{\mathrm{threshold}} \)).
3. Update distribution mean:
   \[
   \mu_t^{(k+1)} = \frac{\sum_{m=1}^M \exp\left( \frac{1}{\lambda} R(\tau^{(m)}) \right) a_t^{(m)}}{\sum_{m=1}^M \exp\left( \frac{1}{\lambda} R(\tau^{(m)}) \right)}
   \]

### 4.2 Supervised Behavioral Cloning into Tiny Transformer
The converged MPPI control actions \( a_0^* \) are recorded across 1,000,000 randomized gym episodes. The Hunter transformer policy \( \pi_\phi(a \mid s) \) is trained via supervised regression:
\[
\mathcal{L}_{\mathrm{Hunter}} = \mathbb{E}_{(s, a^*)}\left[ \| \pi_\phi(s) - a^* \|_2^2 \right]
\]
where input state \( s = [ H(p(T_t)), \Delta \boldsymbol{\mu}_{\mathrm{modes}}, \mathbf{S}_{\mathrm{rim}} ] \).

---

## 5. Parameter Freeze Schedule

| Training Stage | DINOv2 RGB Backbone | Fusion Stem | FSQ Quantizer | Perspective Adapter | Where-Am-I Perceiver | Hunter Policy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 1: 2D Mass Field** | **FROZEN** | TRAINABLE | TRAINABLE | N/A | N/A | N/A |
| **Stage 2: Cross-View Align** | **FROZEN** | **FROZEN** | **FROZEN** | TRAINABLE | TRAINABLE | N/A |
| **Stage 3: Frustum Gym Hunter**| **FROZEN** | **FROZEN** | **FROZEN** | **FROZEN** | **FROZEN** | TRAINABLE |

This freeze schedule prevents catastrophic forgetting of the foundational visual codebook and guarantees that the offline geodata index remains completely stationary.
