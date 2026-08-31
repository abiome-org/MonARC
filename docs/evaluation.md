# Evaluation Protocols and Validation Benchmarks

Date: 2026-08-30  
Status: Evaluation Protocol Specification  
Repository: [abiome-org/MonARC](https://github.com/abiome-org/MonARC)  

---

## 1. Empirical Discipline and Anti-Fabrication Invariants

MonARC enforces rigorous evaluation standards. Fabricating synthetic metric thresholds or presenting limited academic benchmark numbers as global flight performance is strictly prohibited.

1. **No Random Frame Splitting**: Never split train and test sets by randomly sampling video frames from the same flight path. Contiguous temporal frames share high spatial correlation, artificially inflating validation accuracy.
2. **Spatial Holdouts**: Evaluation must occur on geographically disjoint spatial bounding boxes. For v1 public benches, use the dataset's published splits. For any later Colorado flight trial, hold out a box inside Colorado (at least 25 km from Stage 1 training tiles when the state extent allows). Do not require a CONUS-wide holdout grid.
3. **Temporal & Vintage Shift**: Evaluation imagery must be tested against reference geodata from a different survey vintage (e.g., test 2024 UAV imagery against 2020 NAIP orthophotos) and across seasonal shifts (e.g., leaf-on training vs. leaf-off or snow-covered testing).
4. **Campus Retrieval vs. Flight in Colorado**: Metrics on small academic datasets (e.g., University-1652 Recall@1) measure closed-world campus building retrieval, not GPS-denied flight. High academic retrieval numbers must never be conflated with the flight mission envelope. v1 flight claims, if any, are restricted to **Colorado** plus public benches — not CONUS and not "Jefferson County is the product."

### 1.1 Two report tracks for the first working model

Until Colorado NAIP/3DEP indexes and public-UAV adapters have separate executed evaluation artifacts, reports must name the track explicitly:

| Track | What it measures | What it is not |
| :--- | :--- | :--- |
| **Colorado retrieval** | Map-side codes over ingested CO NAIP visualization + 3DEP xyz (Golden–Morrison 10×10 km rehearsal first; v1 product remains Colorado-the-state) | University-1652 Recall@1, OrthoLoC median translation |
| **Public-UAV adapter** | Perspective encoder alignment / retrieval on University-1652 (and later DenseUAV / OrthoLoC) | GPS-denied Colorado flight ATE |

Do not average, mix, or relabel these tracks. Do not publish numeric gates in documentation until a named script, split, and artifact exist for that track.

### 1.2 Colorado retrieval script (`monarc eval-retrieve`)

The Colorado retrieval track is executed by `monarc eval-retrieve --extract <dir> --fsq <dir>`. Inputs are local `features.npy` (extract), `codes.npy` and `xyz.npy` (FSQ train output). No network.

**Split**: axis-aligned spatial box. The high side of the longer east/north span of chip xyz is held out as queries; the complementary half-space is the gallery. Do not sample random neighboring chips.

**Metrics** (printed JSON; write `--out` to persist):

- Recall@1 / Recall@5 (top-level): fraction of queries whose spatially nearest gallery chip appears in the **bag-of-codes** ranking. This is the FSQ baseline.
- `modes["dino-pooled-cosine"]` / `modes["dino-grid-cosine"]`: the same split and Recall@K / rank-1 xyz error using frozen DINO descriptors from `features.npy` (mean-pooled cosine, and flattened-grid cosine). Set when extract features are present. `features_used` is true in that case.
- Median and P90 xyz error: Euclidean distance (3D if z is finite, else horizontal xy) between the query chip xyz and the rank-1 retrieved gallery chip. Oracle distances to the nearest gallery chip are reported alongside so the holdout gap is visible.
- `split.tiny` / `note`: set when `n_chips < 128` or `n_query < 32` (the Golden–Morrison 64-chip rehearsal is in this band). Tiny splits are not Colorado-state or flight results.

This is map-side chip retrieval. It is not matcher+PnP pose, not University-1652, and not a Hunter policy eval. Do not paste numbers into this file; the executable report is the JSON from a named run. Compare bag-of-codes and DINO modes from that JSON; do not copy a prior run's Recall@K into the docs.

### 1.3 Retrieved-candidate matcher + PnP script (`monarc eval-match-pnp`)

`monarc eval-match-pnp --extract <dir> --fsq <dir>` preserves the same spatial-box split but measures a different grain. It first retrieves top-K gallery chips (K=5 by default), then performs mutual-nearest cosine matching between frozen DINO patch grids only inside those candidates. The retrieval descriptor defaults to bag-of-codes; frozen-DINO pooled or flattened-grid retrieval can be selected explicitly. Inputs remain local and the command performs no network access.

Matched query patch uv and gallery xyz are passed through the existing correspondence and PnP/LM path. The present extract+FSQ contract supplies one coarse xyz per chip, so all patches from a candidate share its chip-center coordinate. This is not per-patch terrain geometry and is commonly degenerate for 6-DoF PnP. DSM z may also be NaN. The JSON therefore sets `xyz_kind="coarse-chip-center"`, `xyz_is_chip_center=true`, and `dsm_z_may_be_nan=true`. Non-finite-z ties are excluded from PnP rather than assigned an invented elevation.

Each query records top-K ids, local match inlier count, refined horizontal xy, horizontal error against the query chip center, and whether PnP succeeded. When PnP is underconstrained or fails, refined xy is the locally selected inlier candidate's chip center. Aggregates report median/P90 horizontal error for rank-1 retrieval and matcher refinement. Retrieve Recall@1/5 is recomputed for the candidate stage on the same run. `split.tiny` is true when `n_chips < 128` or `n_query < 32`.

This protocol is a map-cache diagnostic, not University-1652 and not GPS-denied flight ATE. Fixture comparisons are fixture results only; they neither establish nor refute performance on Golden–Morrison or statewide Colorado data. Report actual results only from a saved JSON artifact produced by the command.

---

## 2. Benchmark Evaluation Protocols

```
+===================================================================================================+
|                                PROTOCOL 1: COLD-START RELOCALIZATION                              |
| Prior: None (Uniform over spatial mission bounding box; v1: inside Colorado, not CONUS)          |
| Pre-declared Metric Suite:                                                                        |
|   - Horizontal Translation Error: Median, 75th percentile, 95th percentile (meters)               |
|   - Vertical Translation Error: Median, 95th percentile (meters)                                  |
|   - Heading / Yaw Error: Median, 95th percentile (degrees)                                        |
|   - Time-to-First-Fix (TTFF): Wall-clock time to unimodal posterior convergence (seconds)         |
|   - Pre-declared Reporting Bin: Percentage of trials exceeding 25.0 meters horizontal error       |
+===================================================================================================+
                                                  |
                                                  v
+===================================================================================================+
|                                PROTOCOL 2: CONTINUOUS 6-DoF TRACKING                              |
| Prior: Previous Posterior + IMU Dead-Reckoning Propagation                                        |
| Constraints: Query restricted to local S2 Level-12 Map Shard                                      |
| Pre-declared Metric Suite:                                                                        |
|   - Trajectory Drift Rate: Drift percentage (% of total distance traveled)                       |
|   - Absolute Trajectory Error (ATE): Root-mean-square error over flight path (meters)             |
|   - Relative Pose Error (RPE): Drift per 100 meters traveled                                      |
|   - Filter Consistency (NEES): Normalized Estimation Error Squared against ground truth RTK-GPS   |
+===================================================================================================+
                                                  |
                                                  v
+===================================================================================================+
|                                PROTOCOL 3: ACTIVE HUNTER EFFICIENCY                               |
| Task: Reduce pose uncertainty under high-entropy ambiguous initial conditions                     |
| Baseline: Random exploration yaw / fixed straight-line flight                                     |
| Pre-declared Metric Suite:                                                                        |
|   - Mean Entropy Drop Rate: Delta H per gaze/flight step                                          |
|   - Steps-to-Convergence: Number of active observations required to reach H < H_target           |
|   - Path Overhead Ratio: (Actual Flight Distance) / (Straight-Line Goal Distance)                 |
|   - Attractor Divergence: Path deviation caused by isolated high-saliency single landmarks       |
+===================================================================================================+
```

---

## 3. Diagnostic Metrics and Internal Correlation

MonARC requires reporting internal diagnostic correlations to assess system calibration and failure risk:

```
                      DIAGNOSTIC CORRELATION PROTOCOLS
                      
  (a) Inlier Count vs. Pose Error         (b) Constellation Uniqueness vs. Entropy
  
  Pose Error (m)                          Entropy H(p(T))
       ^                                       ^
       | *                                     | *
       |  *                                    |  *
       |   *                                   |   *
       |     *                                 |     *
       |       * * * * *                       |       * * * * *
   0m  +----------------------->               +----------------------->
       0           [Observed]                  0.0                  1.0
          Verified Inlier Count                     Uniqueness Ratio (lambda_c)
```

### 3.1 Verified Inlier Count vs. Pose Error
- **Metric**: Number of 2D-3D correspondence ties surviving metric constellation verification.
- **Invariant**: Localization error must exhibit monotonic decrease as inlier count increases. If high inlier counts produce high pose errors, it indicates geometric aliasing in the inverted index.

### 3.2 Constellation Uniqueness Ratio (\( \lambda_c \))
- **Definition**: The ratio of candidate spatial locations matching the observed metric constellation relative to the global code occurrence frequency:
  \[
  \lambda_c = \frac{1}{|\{ \mathbf{x} \in \mathcal{M} \mid \mathrm{ConstellationMatch}(\mathbf{x}, \mathcal{C}_{\mathrm{obs}}) \}|}
  \]
- **Invariant**: High uniqueness (\( \lambda_c \to 1.0 \)) must correlate with rapid Shannon entropy reduction in the Where-Am-I head.

---

## 4. Public Benchmark Evaluation & Ground-Truth Flight Validation

**v1 evaluation is the four public datasets.** University-1652, DenseUAV, SUES-200, and OrthoLoC are the training and reporting loop. Dedicated RTK flight trials are expansion / operational validation, not a v1 ingest or training requirement, and are never a CONUS campaign.

| Evaluation Suite | Domain / Ground Truth | Protocol Focus | Primary Metric | v1 |
| :--- | :--- | :--- | :--- | :--- |
| **OrthoLoC Benchmark** | 16,425 UAV images across 47 regions with DOPs/DSMs | 6-DoF pose estimation vs. orthographic geodata | Median Translation (m), Rotation (deg) | Required |
| **University-1652** | 72 University campuses, paired drone-satellite | Cross-view retrieval baseline | Recall@1, AP (Retrieval baseline only) | Required |
| **DenseUAV Benchmark** | Multi-altitude UAV-satellite imagery (80-100m AGL) | Cross-altitude scale robustness | Recall@1, Meter Error vs. GSD | Required |
| **SUES-200 Benchmark** | Multi-altitude drone imagery (150-300m AGL) | High-altitude cross-view matching | Recall@1, Recall@5 | Required |
| **Real-World Flight Trials** | Fixed-wing & multirotor with dual-frequency RTK-GPS | End-to-end 6-DoF GPS-denied trajectory in a declared Colorado mission | ATE (m), Drift (% distance), TTFF (s) | Expansion / validation; not a v1 training gate |

---

## 5. Failure Analysis Reporting Standard

Every evaluation report generated in this repository must record empirical findings against pre-declared metrics and reporting bins:

```
================================================================================
MONARC EVALUATION RUN REPORT
Benchmark ID:      [Observed Benchmark Identifier]
Test Extent:       [Observed Bounding Box / Colorado or Public-Bench Split]
Reference Geodata: [Observed Reference Source & Vintage] vs. [Test Imagery Vintage]
Operating Altitude:[Observed Altitude Range] AGL
================================================================================
1. COLD-START RELOCALIZATION (Lost-in-Space, N = [Observed Trial Count])
   - Median Horizontal Error:       [Observed Value] m
   - 95th Percentile Error:         [Observed Value] m
   - Median Altitude (Z) Error:     [Observed Value] m
   - Median Yaw / Heading Error:    [Observed Value] deg
   - Time-to-First-Fix (TTFF):      [Observed Value] s
   - Gross Error Bin (> 25m):       [Observed Value] %

2. CONTINUOUS 6-DoF TRACKING (Flight Length = [Observed Distance] km)
   - Absolute Trajectory Error:     [Observed Value] m
   - Total Trajectory Drift:        [Observed Value] % of distance traveled
   - S2 Shard Query Latency:        [Observed Value] ms

3. ACTIVE HUNTER POLICY
   - Entropy Drop per Step:         [Observed Value] bits/action
   - Steps to Unimodal Fix:         [Observed Value] steps
   - Attractor Detour Distance:     [Observed Value] m
================================================================================
```
