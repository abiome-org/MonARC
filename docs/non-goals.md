# Non-Goals, Architectural Boundaries, and Anti-Patterns

Date: 2026-08-28  
Status: Out-of-Scope Architectural Registry  
Repository: [abiome-org/MonARC](https://github.com/abiome-org/MonARC)  

---

## 1. Explicit Non-Goals

To prevent scope creep and maintain strict architectural boundaries, the following capabilities and paradigms are explicitly designated as non-goals for MonARC:

```
+---------------------------------------------------------------------------------------+
|                                  MONARC SYSTEM SCOPE                                  |
|                                                                                       |
|  [ IN SCOPE ]                                   [ EXPLICIT NON-GOALS ]                |
|  - 6-DoF GPS-denied global localization         - Autopilot motor control (PX4 clone) |
|  - 80m to 150m AGL nadir-oblique flight         - Close-quarters obstacle avoidance   |
|  - Map-conditioned SE(3) posterior estimation   - Unconstrained GeoGuessr guessing    |
|  - Emergent metric landmark constellations      - Street-level automotive VPR         |
|  - Federal geodata ingestion (NAIP/3DEP/OSM)    - Human semantic POI / business names |
|  - Abstract 2.5D frustum gym RL                 - Heavyweight game engine sims (FSX)  |
|  - 4-way decoupled modular subsystems           - Monolithic end-to-end VLA models    |
|  - Discrete FSQ visual codebooks                - Continental 50-ft drone grid sweeps |
+---------------------------------------------------------------------------------------+
```

---

## 2. Detailed Breakdown of Excluded Paradigms

### 2.1 Autopilot Flight Controller Clone (Not PX4 or ArduPilot)
- **Boundary**: MonARC does not implement low-level PID attitude loops, Electronic Speed Controller (ESC) pulse-width modulation (PWM), or fail-safe battery return-to-launch logic.
- **Interface Contract**: MonARC produces an absolute SE(3) pose posterior stream and trajectory correction vectors transmitted via standard MAVLink messages (`VISION_POSITION_ESTIMATE`, `ATT_POS_MOCAP`) or ROS 2 topics to an external autopilot (e.g., PX4 or ArduPilot).

### 2.2 Close-Quarters Obstacle Avoidance (Not Skydio Autonomy)
- **Boundary**: MonARC is a macro-scale global navigation and relocalization system designed for 80 m to 150 m AGL. It does not compute dense micro-voxel occupancy grids for weaving between tree branches, power lines, or building overhangs.
- **Rationale**: Obstacle avoidance operates in high-frequency local ego-frames (0.5 m to 5 m scale), whereas MonARC operates in the georeferenced global frame.

### 2.3 Primary Visual-Inertial Odometry (VIO) Replacement
- **Boundary**: MonARC does not replace high-frequency local Visual Odometry (VO) or Visual-Inertial Odometry (VIO) algorithms (e.g., OpenVINS, VINS-Mono).
- **Function**: MonARC serves as the drift-free global geo-referencing anchor that bounds accumulating VIO dead-reckoning integration drift.

### 2.4 Unconstrained Visual Guessing (Not GeoGuessr)
- **Boundary**: MonARC is strictly map-conditioned. It never attempts open-world visual guessing or hallucinating coordinates from an un-mapped image without reference geodata.
- **Constraint**: If an area has not been ingested into the continuous feature field and inverted metric index, MonARC reports indeterminate global uncertainty rather than emitting heuristic guesses.

### 2.5 Street-Level Automotive Place Recognition (Not Ground VPR)
- **Boundary**: Ground-level automotive VPR systems operate at 1.5 m AGL amidst vertical facades, pedestrians, and dynamic traffic. MonARC's visual codebook is tuned exclusively for top-down and nadir-oblique aerial perspectives (80–150 m AGL).
- **Anti-Pattern**: Do not import ground-view street datasets (e.g., Pittsburgh, Tokyo 24/7, RobotCar) into the MonARC aerial codebook.

### 2.6 Human Semantic POI Taxonomies
- **Boundary**: MonARC strictly ignores human semantic labels, business names, restaurant categories, Wikipedia entities, and municipal boundaries.
- **Invariant**: Visual landmarks are emergent mathematical extrema in multi-modal feature space (roof vertices, forest-clearing boundaries, road intersections, elevation ridges). Relying on human semantic POIs introduces brittle, non-metric dependencies.

### 2.7 Spatial Hexagonal / Binned Cell Maps (Not H3 as Representation)
- **Boundary**: H3 hexagons, geohash strings, and administrative bounding boxes are prohibited as primary landmark representations.
- **Invariant**: Landmarks exist at continuous metric coordinates \( (x, y, z) \in \mathbb{R}^3 \). Spatial hashing (such as S2) is used strictly as an acceleration shard for local runtime cache queries during continuous tracking, never as the landmark's identity.

### 2.8 Game Engine and Flight Simulator Policy Gyms (Not MSFS or Unreal)
- **Boundary**: Training active vision policies inside heavyweight 3D flight simulators (Microsoft Flight Simulator, Unreal Engine, Unity) is prohibited.
- **Rationale**: 3D rendering engines introduce massive sim-to-real visual domain gaps and bottleneck policy rollout throughput to < 100 FPS. MonARC policies are trained at > 10,000 FPS in mathematical 2.5D frustum gyms operating directly over geodata arrays.

### 2.9 Monolithic Vision-Language-Action (VLA) Architecture
- **Boundary**: MonARC explicitly rejects single end-to-end transformers that map camera pixels directly to actuator actions.
- **Invariant**: The four subsystems (Map Ingestion, Perspective Encoder, Where-Am-I Head, Hunter Policy) must remain strictly decoupled through typed tensor contracts. The Hunter policy never observes raw pixels.

### 2.10 Physical 50-Foot Continental Photographic Sweeps
- **Boundary**: MonARC does not require or recommend flying drone camera grids at 50 ft AGL across the United States.
- **Strategy**: The map is synthesized entirely from pre-existing open federal geodata (NAIP 0.5m orthophotos, USGS 3DEP LiDAR DSMs) and open vector geometry (Overture Maps).

### 2.11 Unstructured Web Video Scraping
- **Boundary**: Scraping video sharing platforms (e.g., YouTube drone compilations) is banned due to unknown camera intrinsics, unreliable GPS geotags, compression artifacts, and licensing liabilities.

### 2.12 Fabricated Performance Thresholds
- **Boundary**: MonARC documentation, benchmarks, and pull requests must never claim unverified numeric gates (e.g., "99.8% precision everywhere"). All reported figures must derive from reproducible evaluation scripts and published dataset splits.
