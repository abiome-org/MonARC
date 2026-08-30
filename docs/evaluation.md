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
