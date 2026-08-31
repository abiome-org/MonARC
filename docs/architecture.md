# System Architecture: The Four Decoupled Subsystems

Date: 2026-08-30  
Status: Architectural Baseline  
Repository: [abiome-org/MonARC](https://github.com/abiome-org/MonARC)  

---

## 1. Architectural Invariant: Decoupled Decomposition

MonARC enforces a four-piece decoupled architecture. Collapsing these subsystems into a single end-to-end Vision-Language-Action (VLA) model or allowing active policy components to consume raw camera pixels is strictly prohibited.

v1 **does not change this decomposition**. It compresses Subsystem 1's geographic and storage scope to the **state of Colorado**. Jefferson County / Front Range may be a first slice, not the product boundary. CONUS is v2, gated on Colorado working. Cost law: [`docs/cost.md`](./cost.md).

```
+====================================================================================================+
|                                  SUBSYSTEM 1: MAP & CODEBOOK                                       |
|  Offline Ingestion:                                                                                |
|    NAIP RGB [B, 3, H, W] ---------> [Frozen DINOv2 Backbone] -------> f_rgb [B, D_v, H/14, W/14]   |
|                                                                              |                     |
|    3DEP DSM [B, 1, H, W] \                                                   v                     |
|    OSM/Overture [B, C_g, H, W] ---> [Lightweight Fusion Stem] ------> [Channel Fusion MLP]         |
|                                                                              |                     |
|                                                                              v                     |
|                                                                     [FSQ Quantizer Head]           |
|                                                                              |                     |
|                                      +---------------------------------------+                     |
|                                      v                                       v                     |
|                     Optional on-demand interpolated field   Inverted Metric Index (v1 export)|
|                     Phi_map: R^2 -> R^D                       code -> {xyz in R^3,               |
|                     (working-set grid; never a dense                    constellation: {code_j,   |
|                      CONUS fp16 / Zarr store)                                  d_ij, b_ij}}     |
+====================================================================================================+
                                                                               |
                                      +----------------------------------------+
                                      | (Exported Colorado S2 shards; no CONUS map onboard)
                                      v
+====================================================================================================+
|                               SUBSYSTEM 2: PERSPECTIVE ENCODER                                     |
|  Onboard Perception (The ONLY module where perspective pixels are spent):                          |
|    Live Camera Frame I_t [B, 3, H_c, W_c]                                                          |
|       |                                                                                            |
|       v                                                                                            |
|    [Frozen Vision Backbone / VGGT Sequence Geometry Tokens]                                        |
|       |                                                                                            |
|       v                                                                                            |
|    [Contrastive BEV Alignment Head]  <-- Aligned to Phi_map (Colorado interpolated field)          |
|       |                                                                                            |
|       v                                                                                            |
|    [FSQ Discretization & Temperature-Calibrated Confidence Head]                                   |
|       |                                                                                            |
|       v                                                                                            |
|    Landmark Set: S_t = { (code_i, u_i, v_i, c_i) }_{i=1}^{N_t}                                     |
|      code_i in {0, ..., K-1}, (u_i, v_i) in R^2, c_i in [0, 1]                                     |
+====================================================================================================+
                                      |
                                      v
+====================================================================================================+
|                               SUBSYSTEM 3: WHERE-AM-I HEAD                                         |
|  Onboard State Estimation (Images NEVER enter this module):                                        |
|    Inputs: S_t (Landmark Set), p(T_{t-1}) (Prior Pose Distribution), Local Map Index               |
|       |                                                                                            |
|       +--> (If Lost-in-Space): Corridor Retrieval (MegaLoc / code n-grams) -> Tile Seed        |
|       +--> Geometric Hypothesis Generation: Differentiable PnP (dPnP) Particle Initializer        |
|       +--> Metric Constellation Verification: Resolve code aliasing via relative bearings/distances|
|       |                                                                                            |
|       v                                                                                            |
|    [Perceiver Set Transformer]                                                                     |
|       * Cross-attends landmark correspondence tokens into SE(3) latent pose query tokens           |
|       * Outputs particle log-weights w_k and Lie algebra innovation offsets delta_xi in se(3)      |
|       |                                                                                            |
|       v                                                                                            |
|    SE(3) Pose Posterior: p(T_t | I_{1:t}, M) = sum_{k=1}^K w_k * N_SE(3)(T_{t,k}, Sigma_k)        |
+====================================================================================================+
                                      |
                                      v
+====================================================================================================+
|                               SUBSYSTEM 4: HUNTER ACTIVE POLICY                                    |
|  Onboard Active Perception (Zero pixel inputs; operates on distribution tokens):                   |
|    Inputs: Posterior Entropy H(p(T_t)), Mode Vectors {mu_k, w_k}, Frustum Rim Codes S_rim          |
|       |                                                                                            |
|       v                                                                                            |
|    [Compact Mode-Attention Transformer]                                                            |
|       * Trained offline via MPPI/CEM on expected entropy reduction in abstract 2.5D Frustum Gym   |
|       * Distilled via Behavioral Cloning into tiny transformer                                     |
|       |                                                                                            |
|       v                                                                                            |
|    Action a_t: [delta_yaw, delta_pitch, delta_v_x, delta_v_y, delta_v_z] in R^5                      |
+====================================================================================================+
```

---

## 2. Subsystem 1: Map Representation and Discrete Metric Codebook

### 2.1 The GLACE Dilemma in Large-Scale Geo-Localization
Traditional Scene Coordinate Regression (SCR) methods (e.g., DSAC*, ACE) learn direct neural mappings from visual patches to absolute 3D world coordinates \( \mathbf{x} \in \mathbb{R}^3 \). As proven by Wang et al. (GLACE, CVPR 2024), scaling direct coordinate regression toward continental extents encounters a fundamental dilemma. v1 does **not** attempt continental SCR; it indexes **Colorado**. The dilemma still applies inside repetitive Front Range suburbs, plains, and montane terrain, so the same structural split is required:
- **Invariance Requirement**: The model must be invariant to viewpoint, illumination, seasonal changes, and sensor noise for the same geographic landmark.
- **Discrimination Requirement**: The model must simultaneously discriminate between distinct geographic locations that exhibit near-identical visual appearances (e.g., suburban road grids, repetitive farmland, identical commercial buildings).

MonARC resolves the GLACE dilemma through structural separation:
1. **Visual Quantization**: Visual features are quantized into a discrete, finite codebook via Finite Scalar Quantization (FSQ). Visual codes capture local appearance patterns without attempting to encode global spatial coordinates.
2. **Metric Constellation Indexing**: Spatial uniqueness is enforced geometrically rather than visually. The inverted index stores co-visible *metric constellations* (exact relative 3D distance vectors and angular bearings between adjacent landmarks). An ambiguous landmark code is disambiguated by the metric geometry of its co-visible cluster, not by memorizing world coordinates in neural network weights.

### 2.2 Dual-Access Map Representation
The map is a single mathematical object \( \mathcal{M} \) accessed through two views. **v1 persists the inverted index for Colorado.** The continuous view is interpolated/on-demand or a working-set grid — not a dense CONUS fp16 / Zarr store.

- **Aerial Feature Field \( \Phi_{\mathrm{map}}(\mathbf{p}) \)**: Mapping from geodetic surface coordinates \( \mathbf{p} = (u_{\mathrm{geo}}, v_{\mathrm{geo}}) \in \mathbb{R}^2 \) to continuous features \( \mathbf{z} \in \mathbb{R}^{D_v} \). In v1 this is reconstructed from FSQ codes or a working-set interpolated grid over Colorado. A continental dense field is v2+, not a v1 artifact.
- **Inverted Metric Landmark Index \( \mathcal{I}_{\mathrm{map}} \)** (**v1 export**): An inverted table mapping discrete landmark code \( c \in \{0, \dots, K-1\} \) to occurrences \( \{ (\mathbf{x}_m, \mathcal{C}_m) \} \), where \( \mathbf{x}_m = (x, y, z)_m \in \mathbb{R}^3 \) in local UTM/EPSG coordinates and \( \mathcal{C}_m \) defines the local co-visibility constellation:
  \[
  \mathcal{C}_m = \left\{ \left( c_j, \Delta \mathbf{x}_{mj}, \theta_{mj} \right) \mid j \in \mathrm{Neighbors}(m), \, \|\Delta \mathbf{x}_{mj}\|_2 \le R_{\mathrm{covis}} \right\}
  \]
  where \( \Delta \mathbf{x}_{mj} = \mathbf{x}_j - \mathbf{x}_m \) is the relative 3D displacement vector and \( \theta_{mj} \) is the metric azimuth bearing.

Landmarks stored in \( \mathcal{I}_{\mathrm{map}} \) are **emergent extrema** in the fused field (corners, junctions, roof geometry, terrain texture peaks). Do not index every DINOv2 token.

```
                      METRIC CONSTELLATION STRUCTURE
                      
                          Landmark B [Code 402]
                                 ^
                                 |  Delta x_AB = (+12.4m, +48.2m, +3.1m)
                                 |  Bearing theta_AB = 075.5 deg
                                 |
     Landmark A [Code 109] ------+------> Landmark C [Code 881]
     (Center of Constellation)   |        Delta x_AC = (+64.0m, -08.1m, -0.4m)
                                 |        Bearing theta_AC = 172.8 deg
                                 v
                          Landmark D [Code 055]
                          Delta x_AD = (-22.1m, -35.0m, +1.2m)
                          Bearing theta_AD = 237.7 deg
```

### 2.3 Channel Ingestion and Fusion Stem
Frozen visual foundation backbones (e.g., DINOv2-ViT-B/14) are strictly 3-channel RGB models. Concatenating 1-channel digital elevation models (USGS 3DEP DSM) and multi-channel rasterized vector geometries (Overture Maps / OSM building and road masks) directly to the RGB tensor violates the pretrained input domain.

MonARC enforces the following ingestion pipeline. v1 inputs are the Colorado clip (a Jefferson County slice may run first), range-read in `us-west-2`. Do not concatenate 6-channel tensors into frozen DINOv2. Do not inject geology or landcover class channels as a semantic taxonomy.

```
+-----------------------------------------------------------------------------+
| INGESTION TENSOR SPECIFICATIONS                                             |
|                                                                             |
| Input 1: Orthophoto RGB       T_rgb in R^{B x 3 x H x W}                    |
|          (v1: NAIP visualization COG, one vintage, ~0.6 m)              |
| Input 2: 3DEP Elevation DSM   T_dsm in R^{B x 1 x H x W}                    |
| Input 3: Vector Masks Raster  T_vec in R^{B x C_g x H x W}                  |
|          (Channel 0: Road Centerlines, Channel 1: Building Footprints,      |
|           Channel 2: Water Boundaries; binary geometric masks only.           |
|           Not NLCD, geology, or landcover class IDs.)                        |
+-----------------------------------------------------------------------------+
```

```
                          INGESTION FLOW
                          
  T_rgb [B, 3, H, W] ---------------------> [Frozen DINOv2 Backbone]
                                                    |
                                                    v
                                            f_rgb [B, 768, H/14, W/14]
                                                    |
  T_dsm [B, 1, H, W]   \                            |
                        +-> [Trainable Fusion Stem] |
  T_vec [B, C_g, H, W] /    (Conv2D 3x3 -> Norm)    |
                                    |               |
                                    v               v
                            f_geo [B, 128, H/14, W/14]
                                    \               /
                                     \             /
                                      v           v
                                 [Cross-Attention / Fusion MLP]
                                              |
                                              v
                                      f_fused [B, 256, H/14, W/14]
                                              |
                                              v
                                   [FSQ Quantizer Head]
                                              |
                                              v
                                   Discrete Codes [B, H/14, W/14]
```

### 2.4 Quantization via Finite Scalar Quantization (FSQ)
Standard Vector Quantization (VQ-VAE) suffers from codebook collapse and requires delicate codebook reset heuristics. MonARC adopts Finite Scalar Quantization (FSQ; Mentzer et al., 2023). FSQ bounds each feature dimension to a discrete set of levels \( L = (l_1, l_2, \dots, l_d) \), yielding a fixed codebook of size \( K = \prod_{i=1}^d l_i \). There is no learned embedding table.

v1 Stage-1 / `train-fsq` default is Mentzer et al. 10-bit \( L = (8, 5, 5, 5) \), \( K = 1000 \). The CPU dry-run may use a smaller \( L = (5, 5, 5) \) (\( K = 125 \)) for synthetic chips only. Even \( l_i \) use the paper offset so all \( l_i \) bins are reachable. A reconstruction-only objective can still collapse the projection onto a handful of joint codes (observed: `unique_codes: 4` on 128 Golden–Morrison chips with \( L = (5, 5, 5) \)); Stage 1 therefore includes a differentiable usage term on soft code occupancy (see [`docs/training.md`](./training.md) §2.2). Unique-code counts that are tiny relative to \( \min(n_{\mathrm{chips}}, K) \) are treated as collapse, not as a map.

\[
\hat{z}_i = \mathrm{round}\bigl( (l_i-1)/2 \cdot \tanh(z_i + s_i) - o_i \bigr)
\]
with \( o_i = 0 \) for odd \( l_i \) and \( o_i = 1/2 \) for even \( l_i \), \( s_i = \tan(o_i / \mathrm{half}\text{-}l_i) \).

---

## 3. Subsystem 2: Perspective Encoder

The Perspective Encoder is the sole onboard module where raw camera pixels are processed.

```
                      PERSPECTIVE ENCODER FLOW
                      
  Live Camera Frame I_t [B, 3, H_c, W_c]
          |
          v
  [Frozen Vision Backbone (DINOv2-ViT-S/14 or Sequence VGGT Tokens)]
          |
          v
  Patch Tokens f_persp in R^{B x N_p x D_v}
          |
          v
  [Trainable Cross-View Projection Head] (Contrastively aligned to Phi_map)
          |
          v
  Aligned Tokens z_persp in R^{B x N_p x D_fsq}
          |
          +-----------------------------------+
          |                                   |
          v                                   v
  [FSQ Discretization]             [Confidence Calibration MLP]
          |                                   |
          v                                   v
  Landmark Codes c_i in {0..K-1}    Confidence c_i = sigma(w^T z_i + b) in [0, 1]
          \                                   /
           +----------------+----------------+
                            |
                            v
  Output Set: S_t = { (c_i, u_i, v_i, c_i) }_{i=1}^{N_t} (sparse extrema; N_t ~ 32 to 128)
```

### 3.1 Contrastive Alignment to Ortho Feature Field
Before discretization, perspective patch features \( \mathbf{z}_{\mathrm{persp}} \) are projected through a lightweight adapter network \( g_\theta \) and contrastively aligned to the ortho feature field \( \Phi_{\mathrm{map}} \) (v1: Colorado interpolated field or public-bench paired geodata) using an InfoNCE objective:
\[
\mathcal{L}_{\mathrm{align}} = -\log \frac{\exp\left( \langle g_\theta(\mathbf{z}_{\mathrm{persp}}), \Phi_{\mathrm{map}}(\mathbf{p}^*) \rangle / \tau \right)}{\sum_{\mathbf{p}'} \exp\left( \langle g_\theta(\mathbf{z}_{\mathrm{persp}}), \Phi_{\mathrm{map}}(\mathbf{p}') \rangle / \tau \right)}
\]
where \( \mathbf{p}^* \) is the ground-truth geodetic footprint of the perspective ray intersected with the 3DEP terrain surface.

### 3.2 Confidence Calibration
Vector quantization commitment error is not a calibrated localization confidence. MonARC trains an auxiliary confidence head \( c(\mathbf{z}) \in [0, 1] \) calibrated via temperature scaling and supervised by correspondence geometric consistency:
\[
c_i = \sigma\left( \frac{\mathbf{w}_c^\top \mathbf{z}_i + b_c}{T_{\mathrm{calib}}} \right)
\]
Correspondences with \( c_i < \tau_{\mathrm{conf}} \) (a filter hyperparameter swept during development) are filtered prior to state estimation.

---

## 4. Subsystem 3: Where-Am-I State Estimation Head

The Where-Am-I head operates strictly on discrete landmark sets and geometric state distributions. Raw camera imagery is never passed to this module.

```
                      WHERE-AM-I ESTIMATION HEAD FLOW
                      
  Inputs:
    1. Observed Landmark Set: S_t = { (c_i, u_i, v_i, c_i) }_{i=1}^{N_t}
    2. Prior Pose Distribution: p(T_{t-1})
    3. Inverted Map Index: I_map (or local S2 shard)
    
                                  |
                                  v
  [1. RETRIEVE-THEN-LOCALIZE PIPELINE]
    * Lost-in-Space Mode: MegaLoc / code n-gram hash over the *Colorado* index -> Top-K tiles
    * Tracking Mode: Query constrained to local S2 cell (radius ~ 500m) inside loaded shards
                                  |
                                  v
  [2. GEOMETRIC HYPOTHESIS GENERATION (dPnP INITIALIZER)]
    * Query I_map for candidate 3D coordinates: x_i in R^3 for code c_i
    * Formulate 2D-3D correspondence pairs: ( (u_i, v_i) <-> (x_i, y_i, z_i) )
    * Differentiable PnP generates initial particle mode hypotheses: { T^{(k)}_init }
                                  |
                                  v
  [3. METRIC CONSTELLATION VERIFICATION]
    * Compute pairwise bearing and distance errors across observed co-visible graph
    * Reject global code collisions that violate rigid SE(3) constellation geometry
                                  |
                                  v
  [4. PERCEIVER SET TRANSFORMER]
    * Cross-Attention: Latent SE(3) pose queries cross-attend to verified ties
    * Self-Attention: Refine inter-mode correlations
    * Emission: Particle log-weights w_k and Lie algebra deltas delta_xi_k in se(3)
                                  |
                                  v
  [5. SE(3) POSE POSTERIOR UPDATE]
    * T_{t,k} = T^{(k)}_init * exp(delta_xi_k)
    * Pose Posterior: p(T_t | S_t) = sum_{k=1}^K w_k * N_SE(3)(T_{t,k}, Sigma_k)
```

### 4.1 Tensor Specifications for Perceiver Head

```
+-----------------------------------------------------------------------------+
| PERCEIVER SET TRANSFORMER TENSOR SIGNATURES                                 |
|                                                                             |
| Input 1 (Observation Tokens):                                               |
|   T_obs in R^{B x N_t x D_obs}                                              |
|   where D_obs = 3 (xyz) + 2 (uv) + 1 (conf) + D_code (code embedding)      |
|                                                                             |
| Input 2 (Latent Pose Query Tokens):                                         |
|   T_pose in R^{B x K_modes x D_pose}                                        |
|   initialized from dPnP modes and previous posterior particles              |
|                                                                             |
| Output 1 (Mode Weights):                                                    |
|   w_k in R^{B x K_modes}, with sum_{k=1}^{K_modes} w_k = 1                  |
|                                                                             |
| Output 2 (Lie Algebra Innovation Deltas):                                   |
|   delta_xi_k in R^{B x K_modes x 6}  (se(3) tangent space corrections)      |
|                                                                             |
| Output 3 (Mode Covariance Diagnostics):                                     |
|   Sigma_k in R^{B x K_modes x 6 x 6} (Positive Definite covariance)        |
+-----------------------------------------------------------------------------+
```

### 4.2 Mathematical Update on SE(3)
State estimation is formulated directly on the Special Euclidean group \( \mathrm{SE(3)} = \mathrm{SO(3)} \ltimes \mathbb{R}^3 \). Let \( T \in \mathrm{SE(3)} \) be represented as a \( 4 \times 4 \) transformation matrix:
\[
T = \begin{bmatrix} R & \mathbf{t} \\ \mathbf{0}^\top & 1 \end{bmatrix}, \quad R \in \mathrm{SO(3)}, \, \mathbf{t} \in \mathbb{R}^3
\]
Tangent space updates apply the exponential map \( \exp: \mathfrak{se}(3) \to \mathrm{SE(3)} \):
\[
\hat{T}_k = T_k \cdot \exp\left( \delta \boldsymbol{\xi}_k^\wedge \right)
\]
where \( \delta \boldsymbol{\xi} = [\mathbf{v}^\top, \boldsymbol{\omega}^\top]^\top \in \mathbb{R}^6 \).

---

## 5. Subsystem 4: Hunter Active Perception Policy

The Hunter policy controls the vehicle's gaze and trajectory to actively minimize localization uncertainty.

```
                      HUNTER ACTIVE POLICY FLOW
                      
  Inputs (Zero raw pixels):
    1. Posterior Shannon Entropy: H(p(T_t)) = - sum_k w_k log w_k + 0.5 * log |Sigma_k|
    2. Dominant Mode Dispersion: delta_mu = { mu_k - mu_1 }_{k=2}^K in R^{(K-1) x 6}
    3. Frustum Rim Codes: S_rim = { (c_r, theta_r, phi_r) } (Codes visible on boundary)
    
                                  |
                                  v
  [MODE-ATTENTION TRANSFORMER POLICY (Compact Actor)]
    * Self-attention over mode dispersion tokens and rim-candidate tokens
    * Generates information-gradient trajectory adjustments
                                  |
                                  v
  Action Vector a_t in R^5:
    [ delta_yaw, delta_pitch, delta_v_x, delta_v_y, delta_v_z ]
    * Gaze Adjustment: Re-point camera gimbal to bring high-saliency rim codes into center
    * Flight Step: Navigate toward geometrically rich landmark constellations
```

### 5.1 Training Protocol: MPPI Expert to Transformer Distillation
The Hunter is trained entirely within an abstract 2.5D frustum gym on a **CPU** (laptop or workstation; millions of episodes). Game-engine visual gyms remain forbidden:
1. **Expert Generation via Model Predictive Path Integral (MPPI)**:
   - Roll out candidate trajectory rollouts over horizon \( H \).
   - Reward function is strictly the expected drop in pose posterior Shannon entropy:
     \[
     R(s_t, a_t) = \mathbb{E}\left[ H(p(T_t)) - H(p(T_{t+1})) \right] + \lambda_{\mathrm{smooth}} \|a_t - a_{t-1}\|_2^2
     \]
   - A minor terminal reward \( R_{\mathrm{unique}} \) is added for locking onto an unambiguous metric constellation, but bounded to prevent detour attractor traps.
2. **Behavioral Cloning**:
   - Supervised distillation trains the lightweight transformer policy \( \pi_\theta(a_t \mid s_t) \) to clone the converged MPPI trajectory distribution across randomized terrain tiles and code collision conditions.

---

## 6. Onboard vs. Offline Compute Split

```
+---------------------------------------------------------------------------------------+
| OFFLINE (v1: us-west-2 next to open-data buckets; Colorado)                            |
| Tasks:                                                                                |
|   1. Range-read NAIP visualization COGs (one vintage) + Colorado 3DEP + Overture clip |
|   2. Train fusion stem + FSQ on a sampled diverse tile set (frozen DINOv2)          |
|   3. Infer FSQ codes on Colorado; emit inverted metric index (LMDB/S2)             |
|   4. Optional on-demand interpolated field — never a dense CONUS fp16 / Zarr     |
|   5. Partition the *Colorado* index into S2 Level-12 shards                         |
| Compute: Frozen DINOv2 + small trainable stem on Colorado / sample tiles.           |
|          Not a continental GPU factory. See docs/cost.md.                           |
+---------------------------------------------------------------------------------------+
                                           |
                    (Pre-flight upload of Colorado mission shards)
                                           v
+---------------------------------------------------------------------------------------+
| ONBOARD UAV FLIGHT PAYLOAD                                                            |
| Hardware Class: One Jetson-class System-on-Module                                    |
| Real-Time Execution Loop:                                                             |
|   1. Perspective Encoder: Thin Frozen ViT-S/B Backbone + FSQ Head                     |
|   2. S2-Sharded Inverted Index Query over Colorado Mission Working Set                   |
|   3. Perceiver Where-Am-I Estimation Head (Cross-Attention over Landmark Ties)        |
|   4. Hunter Mode-Attention Transformer Policy (Distribution Tokens -> Action)         |
| Architecture Property:                                                                |
|   The aircraft never carries a CONUS / global map. Working set = Colorado mission     |
|   shards. The offline object is the Colorado inverted index, not a continental field. |
+---------------------------------------------------------------------------------------+
```

---

## 7. First Executable Path (v0 tensor contracts)

This increment implements a CPU-runnable subset of subsystems 1–3. It does not implement Hunter/MPPI and does not make a Perceiver pose regressor the pose solver.

```
RGB [B, 3, H, W], H and W divisible by 14
    -> FrozenDinoBackbone (DINOv2-B/14 contract: patch 14, D=768, frozen)
    -> f_rgb [B, 768, H/14, W/14]

DSM [B, 1, H, W] + vectors [B, 4, H, W]
    -> FusionStem (trainable, never concatenated into the DINO input)
    -> f_geo [B, 128, H/14, W/14]

concat(f_rgb, f_geo) -> ChannelFusion -> f_fused [B, 256, H/14, W/14]
    -> FSQHead (default L=(8,5,5,5), K=1000; no VQ-VAE embedding)
    -> codes [B, H/14, W/14] integer, xyz [N, 3] ENU meters

Lost-in-space retrieve: bag-of-codes (+ optional adjacent n-grams)
Pose: code matcher -> 2D-3D ties -> DLT PnP + Levenberg-Marquardt on se(3)
Persist: codes.npy, xyz.npy, meta.json (no GeoTIFF, no naip-source, no R2 rasters)
```

Official DINOv2-B/14 (`torch.hub` `facebookresearch/dinov2` entry `dinov2_vitb14`, or Hugging Face `facebook/dinov2-base`) loads only when `mode="vitb14"` / CUDA `auto` with a local cache or `allow_download=True`. Tests, `dry-run`, and CPU `auto` use the frozen patch-14 768-d stub and do not download weights.

`monarc ingest-aoi` (default `--source planetary-computer`) writes a Golden–Morrison rehearsal manifest of Planetary Computer NAIP HREFs with anonymous SAS plus public 3DEP records and a chip-window plan. `monarc extract` reads a directory of RGB chips (optional DSM, xyz sidecar) and writes `features.npy` + `xyz.npy` (optional `codes.npy` if an FSQ checkpoint is supplied). Range-read COG chips only; do not copy full GeoTIFFs to R2. `monarc train-fsq` trains fusion+FSQ on that cache on GPU and checkpoints `stage1_last.pt`. `monarc eval-retrieve` loads those arrays on CPU (no network), holds out a spatial box of chips, and reports bag-of-codes Recall@1/5 plus rank-1 xyz error. Rasters are not a persist object. v1 coverage remains Colorado; Golden–Morrison is the rehearsal slice. AWS `naip-visualization` ingest is an explicit source flag, not the first-slice default.

The Perceiver set transformer remains specified in §4 as a later Where-Am-I path. v0 pose is matcher + geometry. Hunter (subsystem 4) is unspecified in code in this increment.
