# Product Specification: GPS-Denied Visual Localization for UAVs

Date: 2026-08-29  
Status: Active Specification  
Repository: [abiome-org/MonARC](https://github.com/abiome-org/MonARC)  

---

## 1. Operating Domain and Platform Definition

MonARC is engineered specifically for autonomous unmanned aerial systems (UAS / UAVs) operating in communication-denied, GPS-denied environments.

v1 geographic scope is **one operational corridor** (default: Jefferson County / Colorado Front Range), not CONUS. Cost law: [`docs/cost.md`](./cost.md).

```
                    ALTITUDE REGIMES & OPERATING ENVELOPE

  Altitude (AGL)
       ^
300m+  |  High Altitude / Satellite Imagery Domain (Pure Ortho, Coarse GSD)
       |
150m   +-- [ TOP OF MONARC OPERATIONAL ENVELOPE ] ------------------------+
       |   * Predominantly nadir-to-oblique perspectives (pitch 0° to 45°) |
       |   * Flat-terrain approximations hold across patch scales         |
       |   * 3DEP DSM provides true geometric ground elevation            |
       |   * Building footprints and road networks visually distinguishable|
80m    +-- [ BOTTOM OF MONARC OPERATIONAL ENVELOPE ] ---------------------+
       |
15m    |  Low Altitude / Facade Regime (~50 ft)
(50ft) |  * Severe 3D occlusions, vertical building facades dominate
       |  * Inverse Perspective Mapping (IPM) breaks down
       |  * OUT OF DEFAULT PRODUCT SCOPE (Automotive/Street VPR Domain)
 0m    +-------------------------------------------------------------------
```

### 1.1 Target Platform Characteristics
- **Vehicle Type**: Multirotor, fixed-wing, or VTOL aerial drones.
- **Operating Altitude**: 80 m to 150 m Above Ground Level (AGL).
- **Camera Configuration**: Monocular or stereo global-shutter camera with calibrated intrinsics, mounted on a fixed downward-canted or 2-axis stabilized gimbal (nadir to 45° oblique).
- **Primary Sensors**: Visual camera stream + tactical/industrial Inertial Measurement Unit (IMU) for inter-frame dead-reckoning propagation.
- **Excluded Primary Signals**: GNSS (GPS, GLONASS, Galileo, BeiDou), RTK base stations, cellular towers, magnetic compass (unreliable due to electromagnetic motor interference or geographic distortion).

### 1.2 Boundary with Ground / Street VPR
Ground vehicles (automobiles, delivery robots) operate at 1.5–2.5 m AGL where visual scenes are dominated by vertical building facades, dynamic pedestrian traffic, street furniture, and extreme perspective fore-shortening. At 80–150 m AGL, the visual manifold is dominated by top-down surface topology, roof contours, field boundaries, road intersections, and tree canopy distributions. MonARC's visual codebook is explicitly trained on orthographic geodata fused with elevation and vector geometry. Mixing street-level ground features into the aerial codebook degrades representation efficiency and introduces code aliasing.

---

## 2. Problem Formulation

MonARC addresses two fundamental localization regimes under zero communication:

```
+-----------------------------------------------------------------------------+
| 1. COLD-START / "LOST-IN-SPACE" LOCALIZATION                                |
| Initial Prior: p(T_0) = Uniform(SO(3) x R^3) across designated *corridor* (not CONUS) |
| Input: Single camera frame I_t + Geo-indexed landmark field M               |
| Process: Coarse Corridor Retrieval (MegaLoc/n-gram) -> Constellation Match   |
| Output: Multimodal Pose Posterior p(T_t | I_t, M) with concentrated modes   |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
| 2. CONTINUOUS 6-DoF TRAJECTORY TRACKING                                      |
| Prior: p(T_t | I_{1:t-1}) maintained via SE(3) Particle / Mixture Filter    |
| Input: Stream I_t, IMU kinematics Delta T_imu, Local S2 Map Shard M_local   |
| Process: Perspective Encoding -> Set-Transformer Update -> Pose Posterior   |
| Output: Unimodal / Low-Entropy SE(3) Pose Estimate T_t in WGS84/UTM Datum   |
+-----------------------------------------------------------------------------+
```

### 2.1 Lost-in-Space (Cold-Start Relocalization)
When the vehicle is initialized without prior state estimates (e.g., dropped from a mothercraft, booted mid-flight, or recovering from total sensor outage):
- The initial pose distribution \( p(T_0) \) is uniform across the designated **mission corridor**. v1: Jefferson County / Colorado Front Range. It is **not** uniform over CONUS.
- The system must retrieve candidate spatial tiles using visual descriptors (MegaLoc-class or code n-gram hashes) over the **corridor index** and resolve ambiguous matches using local metric constellations.
- The output is an explicit SE(3) pose posterior:
  \[
  p(T_t \mid I_t, \mathcal{M}) = \sum_{k=1}^K w_k \, \mathcal{N}_{\mathrm{SE(3)}}(\mu_k, \Sigma_k)
  \]
  where \( w_k \) are mixture weights, \( \mu_k \in \mathrm{SE(3)} \), and \( \Sigma_k \in \mathbb{R}^{6 \times 6} \).

### 2.2 Continuous 6-DoF Tracking
Once the pose posterior mass concentrates around a unimodal cluster:
- The system queries a spatially constrained S2 map shard corresponding to the local bounding volume **inside the loaded corridor shards**.
- High-frequency IMU integration propagates particles forward in time; the visual Where-Am-I head computes particle weight updates and Lie algebra \( \mathfrak{se}(3) \) innovation deltas.
- Metric scale is strictly anchored by the Digital Surface Model (DSM) stored in the landmark field, eliminating monocular scale drift.

---

## 3. Product Success Criteria and Turing-Test Analog

In GPS-denied autonomous aviation, subjective perceptual quality and LLM-as-judge heuristics are invalid metrics. Success is defined strictly by physical and geometric performance against verified RTK-GPS ground truth across spatially held-out flight trajectories:

1. **GPS-Denied Metric Position Estimation**: Accurate horizontal Euclidean position recovery evaluated against dual-frequency RTK-GPS across varied terrain topologies (urban, suburban, rural, agrarian).
2. **True Geometric Altitude Recovery**: Accurate vertical elevation recovery evaluated against LiDAR Digital Surface Models.
3. **Attitude and Heading Alignment**: Metric heading (yaw) and orientation estimation aligned to true geodetic North.
4. **Cold-Start Convergence**: Rapid reduction of pose posterior entropy from an uninformative uniform prior to a concentrated unimodal state.
5. **Zero-Drift Long-Distance Endurance**: Bounded localization error maintained across long flight paths through periodic map-conditioned landmark resets, preventing open-loop dead-reckoning divergence.
6. **Zero-Shot Generalization on New Flight Paths**: The system must localize over flight paths that were never previously flown, relying strictly on pre-ingested public geodata without requiring prior aerial reconnaissance or map retraining. v1 evaluates this inside the designated corridor and on public UAV benches, not as a CONUS flight claim.

All quantitative evaluation protocols, reporting bins, and diagnostic benchmarks are defined in [`docs/evaluation.md`](./evaluation.md).

---

## 4. Failure Modes and Edge Case Taxonomy

Operational deployments must anticipate and mitigate seven primary failure modes:

### 4.1 Repetitive Agrarian / Monoculture Collisions
- **Symptom**: Flying over continuous pivot-irrigated cornfields, wheat fields, or uniform vineyards generates visual codes with high entropy and near-identical spatial arrangements across kilometers.
- **Mitigation**: The system detects high posterior entropy across candidate poses, prevents premature particle filter collapse, and signals the Hunter policy to execute an exploratory trajectory toward field boundaries, drainage ditches, access roads, or farmstead structures.

### 4.2 Seasonal and Environmental Shifts
- **Symptom**: The reference map (e.g., summer NAIP imagery with full leaf canopy) differs drastically from winter flight conditions (snow cover, defoliated deciduous trees, frozen water bodies).
- **Mitigation**: Visual feature extraction via DINOv2 provides invariance to high-frequency color and lighting variations. The fusion stem incorporates invariant elevation contours (USGS 3DEP) and vector road centerline topology (Overture Maps) which remain static across seasons.

### 4.3 Low-Light and Twilight Operations
- **Symptom**: Diminished signal-to-noise ratio in camera imagery at dawn, dusk, or heavy overcast reduces visual keypoint salience.
- **Mitigation**: Temperature-scaled confidence calibrators attenuate low-confidence code emissions. The Where-Am-I head dynamically weights geometry-heavy correspondences over low-light texture matches.

### 4.4 Featureless Expanses (Water and Desert)
- **Symptom**: Open water, smooth sand dunes, or dense cloud layers yield zero valid visual landmark codes.
- **Mitigation**: The system detects zero valid visual ties, suspends visual pose updates, falls back to pure inertial propagation, and updates the posterior covariance to reflect dead-reckoning dispersion until shoreline or topological landmarks enter the camera frustum.

### 4.5 Temporal Infrastructure Drift
- **Symptom**: New residential subdivisions, highway construction, or building demolitions completed after the satellite/aerial survey date produce conflicting correspondences.
- **Mitigation**: The Perceiver set transformer performs robust outlier rejection through learned cross-attention, treating unmapped structures as discordant noise while locking onto persistent invariant terrain and road geometry.

### 4.6 The Water-Tower Attractor Trap
- **Symptom**: An active navigation policy rewarded purely for seeing "unique" landmarks deviates excessively from mission flight paths to circle isolated structures (e.g., water towers, tall antennas).
- **Mitigation**: The Hunter policy's reward function is strictly formulated as expected entropy reduction (\( \Delta H \)) over the vehicle pose posterior. Unique landmarks provide no surplus reward unless they reduce trajectory uncertainty along the flight path.

### 4.7 Low-Altitude IPM Breakdown (50 ft AGL)
- **Symptom**: At low altitudes, 3D vertical structures (walls, poles, trees) violate the planar homography assumption of Inverse Perspective Mapping (IPM), causing projected features to shear severely.
- **Mitigation**: Operational envelope enforcement. Altitudes below 80 m AGL are flagged as out-of-spec for planar cross-view alignment, transitioning the platform to local visual odometry or safe landing behaviors.
