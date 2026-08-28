# MonARC: Visual GPS-Denied Localization via Emergent Metric Landmarks

Date: 2026-08-28  
Status: Specification and Architecture Baseline  
Repository: [abiome-org/MonARC](https://github.com/abiome-org/MonARC)  
License: MIT  

---

## 1. Product Overview

MonARC is a map-conditioned, camera-only visual localization system designed for unmanned aerial vehicles (UAVs) operating in GPS-denied environments at altitudes of 80 to 150 meters above ground level (AGL). Given an onboard camera stream and an offline geo-indexed visual landmark field, MonARC solves both the cold-start "lost-in-space" problem (zero initial pose prior) and continuous 6-DoF trajectory tracking by estimating an SE(3) pose posterior without active RF signals, magnetic compass trust, or pre-flown perspective visual databases. The name derives from *monarch* (biological long-distance navigation over emergent visual cues) and *Automatic Retrieval Course*.

```
+---------------------------------------------------------------------------------------+
|                                    OFFLINE INGESTION                                  |
|  NAIP Ortho (RGB) + USGS 3DEP (DSM) + Overture/OSM (Geometry Channels)                 |
|       |                     |                                                         |
|  [Frozen DINOv2]      [Fusion Stem]                                                   |
|       \                     /                                                         |
|        +-----> [FSQ Quantizer] -----> Continuous Feature Field & Inverted Metric Index|
+---------------------------------------------------------------------------------------+
                                                                |
+---------------------------------------------------------------+-----------------------+
|                                    ONBOARD RUNTIME            v                       |
|  Live Drone Camera Frame (80-150m AGL)                                                |
|       |                                                                               |
|  [Perspective Encoder]                                                                |
|       |                                                                               |
|       +--> {code, pixel_uv, confidence}                                               |
|                 |                                                                     |
|                 v                                                                     |
|  [Where-Am-I Head] <------- Local Metric Constellation Sub-Index (S2 Sharded)         |
|       |                                                                               |
|       +--> SE(3) Pose Posterior (Particles / Mixture Modes)                           |
|                 |                                                                     |
|                 v                                                                     |
|  [Hunter Transformer Policy]                                                          |
|       |                                                                               |
|       +--> Look-at / Yaw / Frustum Step Command (Information Gain Optimization)       |
+---------------------------------------------------------------------------------------+
```

---

## 2. Locked Architecture

MonARC enforces a strict four-subsystem decoupled architecture. Under no circumstance is the policy network permitted to observe raw pixels, nor is the system collapsed into an end-to-end Vision-Language-Action (VLA) model:

1. **Map Representation and Codebook**: A unified dual-access geodata object. Ingestion combines orthophotography (RGB) through a frozen foundation vision encoder (DINOv2) with high-resolution digital surface models (USGS 3DEP DSM) and vector geometry rasters (OSM/Overture building/road masks) processed through a lightweight fusion stem. Continuous representations are quantized via Finite Scalar Quantization (FSQ) into a continuous aerial feature field and an inverted index mapping landmark codes to 3D metric coordinates and co-visible metric constellation bearings.
2. **Perspective Encoder**: An onboard perception module executing a frozen vision backbone (or geometric tokens from a sequence transformer) with a lightweight projection head that aligns oblique perspectives to the orthographic metric feature space before FSQ discretization. Outputs sparse sets of `{code, pixel_uv, confidence}` tuples. Perspective pixels are consumed only here.
3. **Where-Am-I Estimation Head**: A set transformer (Perceiver-style architecture) taking sparse landmark correspondences and the prior pose distribution to regress log-weights and \( \mathfrak{se}(3) \) corrections over an SE(3) pose posterior. Global retrieval (MegaLoc-class or code n-grams) seeds the lost-in-space mode, differentiable PnP initializes particle clusters using true DSM metric heights, and metric constellation geometry breaks global code aliasing.
4. **Hunter Active Perception Policy**: A compact transformer policy operating entirely on pose posterior entropy, mode dispersions, and rim landmark codes. Trained offline via Model Predictive Path Integral (MPPI) / Cross-Entropy Method (CEM) on expected information gain within an idealized camera frustum gym and cloned into an onboard actor. Emits gaze and flight steering commands to actively reduce localization entropy.

---

## 3. Data Law

MonARC forbids grid-based aerial photographic sweeps of the continental United States at 50 ft AGL and bans flight simulator visual scrapers (such as MSFS or Unreal Engine) for policy training. The landmark field is constructed exclusively from open federal geodata (NAIP, USGS 3DEP) and open vector geometry (Overture Maps, OpenStreetMap). Real perspective pairs for cross-view alignment are sourced from rigorously geo-referenced public UAV benchmarks (University-1652, DenseUAV, SUES-200, OrthoLoC) and calibrated flight logs. Active vision policies are trained exclusively inside abstract frustum environments against noisy landmark fields.

---

## 4. Documentation Index

Detailed specifications, mathematical derivations, operating constraints, and engineering protocols are organized in the following sections:

- [`AGENTS.md`](./AGENTS.md): Strict engineering laws, development invariants, codebase navigation, and modification protocols for autonomous agents and contributors.
- [`docs/product.md`](./docs/product.md): Operational domain definition (80–150 m AGL), problem formulations, Turing-test localization criteria, and edge-case failure modes.
- [`docs/architecture.md`](./docs/architecture.md): Complete subsystem breakdowns, tensor-level input/output signatures, fusion stems, GLACE dilemma resolution, and metric constellation schemas.
- [`docs/data.md`](./docs/data.md): Data hierarchy (mass geodata vs. thin perspective pairs vs. abstract gym), dataset sources, licensing, and Aflora data factory ingestion pipelines.
- [`docs/training.md`](./docs/training.md): Three-stage sequential training schedule, loss functions, confidence calibration formulations, and freeze requirements.
- [`docs/evaluation.md`](./docs/evaluation.md): Protocol specifications, spatial/seasonal holdouts, metric reporting standards, and rejection of fabricated performance gates.
- [`docs/onboard.md`](./docs/onboard.md): Embedded edge compute specifications, SWaP-C budgets, execution timing constraints, and memory models.
- [`docs/literature.md`](./docs/literature.md): Annotated bibliography of foundational visual localization, coordinate regression, and active vision literature with verified links.
- [`docs/non-goals.md`](./docs/non-goals.md): Explicit architectural exclusions, anti-patterns, and out-of-scope capabilities.

---

## 5. Non-Goals Summary

MonARC is explicitly not:
- An autopilot flight controller or PX4 replacement.
- A close-quarters obstacle avoidance system (such as Skydio Autonomy).
- A primary Visual-Inertial Odometry (VIO) package.
- An unconstrained visual guessing heuristic (GeoGuessr) without map conditioning.
- A street-level visual place recognition pipeline designed for automotive ground views.
- A semantic taxonomy reliant on human labels, business categories, or administrative boundaries.
- An end-to-end monolithic Vision-Language-Action (VLA) network.

---

## 6. License

MonARC is released under the terms of the [MIT License](./LICENSE).
