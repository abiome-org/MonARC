# MonARC: Visual GPS-Denied Localization via Emergent Metric Landmarks

Date: 2026-08-30  
Status: Specification plus first executable CPU path (subsystems 1–3 dry-run)  
Repository: [abiome-org/MonARC](https://github.com/abiome-org/MonARC)  
License: MIT  

---

## 1. Product Overview

MonARC is a map-conditioned, camera-only visual localization system designed for unmanned aerial vehicles (UAVs) operating in GPS-denied environments at altitudes of 80 to 150 meters above ground level (AGL). Given an onboard camera stream and an offline geo-indexed visual landmark field, MonARC solves both the cold-start "lost-in-space" problem (zero initial pose prior) and continuous 6-DoF trajectory tracking by estimating an SE(3) pose posterior without active RF signals, magnetic compass trust, or pre-flown perspective visual databases. The name derives from *monarch* (biological long-distance navigation over emergent visual cues) and *Automatic Retrieval Course*.

**v1 / MonARC-1 coverage is the state of Colorado** (state-boundary or bbox clip). Jefferson County / Front Range may be a first slice, not the product boundary. CONUS is v2, gated on Colorado working. Binding cost law: [`docs/cost.md`](./docs/cost.md).

```
+---------------------------------------------------------------------------------------+
|                     OFFLINE INGESTION (v1: COLORADO, us-west-2)                   |
|  NAIP visualization COGs (one vintage) + Colorado 3DEP DSM + Overture/OSM (state clip)|
|       |                     |                                                         |
|  [Frozen DINOv2]      [Fusion Stem]                                                   |
|       \                     /                                                         |
|        +-----> [FSQ Quantizer] -----> FSQ codes + Inverted Metric Index (Colorado)     |
|                                   (optional on-demand field; not CONUS Zarr)        |
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
|  [Where-Am-I Head] <------- Colorado Metric Index (S2 Shards; no CONUS map onboard)   |
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

1. **Map Representation and Codebook**: Dual-access geodata, **v1-scoped to Colorado**. Ingestion combines NAIP visualization RGB through a frozen DINOv2 backbone with Colorado 3DEP DSM and vector geometry rasters (OSM/Overture building/road masks) through a lightweight fusion stem. Finite Scalar Quantization (FSQ) emits discrete codes. The v1 export is an inverted metric index (code → 3D coordinates and co-visible bearings). Landmarks are emergent extrema, not every DINOv2 token and not hardcoded geology/landcover classes. A dense CONUS feature field is v2+, not v1.
2. **Perspective Encoder**: An onboard perception module executing a frozen vision backbone (or geometric tokens from a sequence transformer) with a lightweight projection head that aligns oblique perspectives to the orthographic metric feature space before FSQ discretization. Outputs sparse sets of `{code, pixel_uv, confidence}` tuples. Perspective pixels are consumed only here.
3. **Where-Am-I Estimation Head**: A set transformer (Perceiver-style architecture) taking sparse landmark correspondences and the prior pose distribution to regress log-weights and \( \mathfrak{se}(3) \) corrections over an SE(3) pose posterior. Retrieval (MegaLoc-class or code n-grams) seeds lost-in-space over the **Colorado index** (or a declared mission bbox inside Colorado). Differentiable PnP initializes particle clusters using true DSM metric heights, and metric constellation geometry breaks code aliasing. The aircraft does not carry a CONUS / global map.
4. **Hunter Active Perception Policy**: A compact transformer policy operating entirely on pose posterior entropy, mode dispersions, and rim landmark codes. Trained offline via Model Predictive Path Integral (MPPI) / Cross-Entropy Method (CEM) on expected information gain within an idealized camera frustum gym (CPU; laptop/workstation) and cloned into an onboard actor. Emits gaze and flight steering commands to actively reduce localization entropy. The policy never observes pixels (VLA ban).

---

## 3. Data Law

MonARC forbids grid-based aerial photographic sweeps of the continental United States at 50 ft AGL and bans flight simulator visual scrapers (such as MSFS or Unreal Engine) for policy training.

**v1 is Colorado-the-state, not one county and not CONUS.** Clip ingest to the Colorado state boundary or state bbox. Jefferson County / Front Range may be a first slice. Pull a single NAIP vintage from `s3://naip-visualization` (JPEG COG, ~0.6 m is the default; do not require 0.3 m). Prefer already-COG 3DEP 1/9 arc-second (~3 m) or 1 m only inside Colorado. Process in `us-west-2`; range-read; do not duplicate rasters. Export FSQ codes and an inverted metric index for Colorado — not a dense CONUS fp16 / Zarr field. Stage 1 trains on a sampled tile set, then infers on Colorado. Stage 2 uses public UAV benches only (University-1652, DenseUAV, SUES-200, OrthoLoC); no custom flight-log campaign. Stage 3 is a CPU frustum gym (no Unreal). CONUS ingest is v2, gated on Colorado working. Sentinel-2 / international coverage is out of v1. Detail: [`docs/cost.md`](./docs/cost.md).

The landmark field is constructed exclusively from open federal geodata (NAIP, USGS 3DEP) and open vector geometry (Overture Maps, OpenStreetMap). Real perspective pairs for cross-view alignment are sourced from rigorously geo-referenced public UAV benchmarks. Active vision policies are trained exclusively inside abstract frustum environments against noisy landmark fields.

---

## 4. Documentation Index

Detailed specifications, mathematical derivations, operating constraints, and engineering protocols are organized in the following sections:

- [`AGENTS.md`](./AGENTS.md): Strict engineering laws, development invariants, codebase navigation, and modification protocols for autonomous agents and contributors.
- [`docs/product.md`](./docs/product.md): Operational domain definition (80–150 m AGL), problem formulations, Turing-test localization criteria, and edge-case failure modes.
- [`docs/architecture.md`](./docs/architecture.md): Complete subsystem breakdowns, tensor-level input/output signatures, fusion stems, GLACE dilemma resolution, and metric constellation schemas.
- [`docs/cost.md`](./docs/cost.md): Binding v1 cost law: Colorado-state coverage, cheap 2026 vendors, Golden–Morrison rehearsal slice, planning envelope (not invoices).
- [`docs/data.md`](./docs/data.md): Data hierarchy (mass geodata vs. thin perspective pairs vs. abstract gym), dataset sources, licensing, and Aflora data factory ingestion pipelines.
- [`docs/training.md`](./docs/training.md): Three-stage sequential training schedule, loss functions, confidence calibration formulations, and freeze requirements.
- [`docs/evaluation.md`](./docs/evaluation.md): Protocol specifications, spatial/seasonal holdouts, metric reporting standards, and rejection of fabricated performance gates.
- [`docs/onboard.md`](./docs/onboard.md): Embedded edge compute specifications, qualitative onboard vs. offline split, execution profiling protocols, and working set models.
- [`docs/literature.md`](./docs/literature.md): Annotated bibliography of foundational visual localization, coordinate regression, and active vision literature with verified links.
- [`docs/non-goals.md`](./docs/non-goals.md): Explicit architectural exclusions, anti-patterns, and out-of-scope capabilities.

---

## 5. First working model

This repository now contains a CPU-executable subset of subsystems 1–3. Hunter/MPPI is not in this increment. Pose is matcher + PnP/LM, not a Perceiver-only regressor.

### 5.1 Install and tests

```
python -m pip install -e ".[dev]"
python -m pytest tests
```

Tests use a frozen patch-14 768-d DINOv2-B **contract stub**. They do not download DINOv2 weights, NAIP/3DEP rasters, or University-1652. CUDA is not required.

### 5.2 Dry-run CLI

Synthetic chips (no AWS): extract frozen-DINO-contract features, train a tiny FSQ projection, write a compact `codes.npy` + `xyz.npy` index, bag-of-codes retrieve, matcher + PnP/LM.

```
python -m monarc.cli dry-run --out artifacts/dry-run --steps 8 --seed 0
```

Observed numbers printed by that command are **that synthetic run only**. They are not Colorado flight metrics and not University-1652 Recall@1.

### 5.3 AOI ingest (Golden–Morrison rehearsal)

Intersects the 10×10 km box (center ~39.725°N, 105.220°W) with NAIP visualization STAC and 3DEP TNMAccess at launch time. Writes a manifest. Does not hardcode NAIP quarter-quad IDs. Does not download rasters.

```
python -m monarc.cli ingest-aoi --out artifacts/golden_morrison_manifest.json --offline tests/fixtures/inventory
```

Omit `--offline` only when live STAC/TNM HTTP is intended. v1 coverage remains Colorado-the-state; CONUS is v2. This box is a $150 rehearsal slice, not a rewrite of coverage ([`docs/cost.md`](./docs/cost.md) §12).

### 5.4 Public-UAV bench (University-1652)

University-1652 is the first loader: ImageFolder building IDs, optional local download, fixture for tests. OrthoLoC (~287 GB, npz+DOP/DSM, CC BY-NC-SA) is registered and deferred.

```
python -m monarc.cli bench-uav --list-benches
python -m monarc.cli bench-uav --root /path/to/University-1652 --list-only
```

### 5.5 Two report tracks

Keep these separate. Do not invent or copy numbers between them.

1. **Colorado retrieval** — FSQ codes and xyz over ingested NAIP visualization + 3DEP inside Colorado (Golden–Morrison rehearsal first; the product remains the state).
2. **Public-UAV adapter** — perspective encoder / retrieval on University-1652 (later DenseUAV, SUES-200, OrthoLoC).

A metric is reportable only when a script, split, and saved artifact exist for that track ([`docs/evaluation.md`](./docs/evaluation.md)).

---

## 6. Non-Goals Summary

MonARC is explicitly not:
- An autopilot flight controller or PX4 replacement.
- A close-quarters obstacle avoidance system (such as Skydio Autonomy).
- A primary Visual-Inertial Odometry (VIO) package.
- An unconstrained visual guessing heuristic (GeoGuessr) without map conditioning.
- A street-level visual place recognition pipeline designed for automotive ground views.
- A semantic taxonomy reliant on human labels, business categories, or administrative boundaries.
- An end-to-end monolithic Vision-Language-Action (VLA) network.
- A v1 CONUS raster factory, dense continental fp16 field, or day-one Sentinel-2 / international ingest. v1 is Colorado-the-state. See [`docs/cost.md`](./docs/cost.md).

---

## 7. License

MonARC is released under the terms of the [MIT License](./LICENSE).
