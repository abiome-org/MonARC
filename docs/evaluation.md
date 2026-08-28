# Evaluation Protocols and Validation Benchmarks

Date: 2026-08-28  
Status: Evaluation Protocol Specification  
Repository: [abiome-org/MonARC](https://github.com/abiome-org/MonARC)  

---

## 1. Empirical Discipline and Anti-Fabrication Invariants

MonARC enforces rigorous evaluation standards. Fabricating synthetic metric thresholds (e.g., claiming "achieves 99.4% accuracy across North America") or presenting limited academic benchmark numbers as global flight performance is strictly prohibited.

1. **No Random Frame Splitting**: Never split train and test sets by randomly sampling video frames from the same flight path. Contiguous temporal frames share high spatial correlation, artificially inflating validation accuracy.
2. **Spatial Holdouts**: Evaluation must occur on geographically disjoint spatial bounding boxes separated by at least 25 km from any training tile.
3. **Temporal & Vintage Shift**: Evaluation imagery must be tested against reference geodata from a different survey vintage (e.g., test 2024 UAV imagery against 2020 NAIP orthophotos) and across seasonal shifts (e.g., leaf-on training vs. leaf-off or snow-covered testing).
4. **Campus Retrieval vs. Continental Flight**: Metrics on small academic datasets (e.g., University-1652 Recall@1) measure closed-world campus building retrieval, not continental visual localization. High academic retrieval numbers must never be conflated with the GPS-denied flight mission envelope.

---

## 2. Benchmark Evaluation Protocols

```
+===================================================================================================+
|                                PROTOCOL 1: COLD-START RELOCALIZATION                              |
| Prior: None (Uniform over 100 km x 100 km region)                                                 |
| Metric Suite:                                                                                     |
|   - Horizontal Translation Error: Median, 75th percentile, 95th percentile (meters)               |
|   - Vertical Translation Error: Median, 95th percentile (meters)                                  |
|   - Heading / Yaw Error: Median, 95th percentile (degrees)                                        |
|   - Time-to-First-Fix (TTFF): Wall-clock time to unimodal posterior convergence (seconds)         |
|   - Global Failure Rate: Percentage of trials with horizontal error > 25.0 meters                 |
+===================================================================================================+
                                                  |
                                                  v
+===================================================================================================+
|                                PROTOCOL 2: CONTINUOUS 6-DoF TRACKING                              |
| Prior: Previous Posterior + IMU Dead-Reckoning Propagation                                        |
| Constraints: Query restricted to local S2 Level-12 Map Shard                                      |
| Metric Suite:                                                                                     |
|   - Trajectory Drift Rate: Drift percentage (% of total distance traveled)                       |
|   - Absolute Trajectory Error (ATE): Root-mean-square error over 10 km flight paths (meters)      |
|   - Relative Pose Error (RPE): Drift per 100 meters traveled                                      |
|   - Filter Consistency (NEES): Normalized Estimation Error Squared against ground truth RTK-GPS   |
+===================================================================================================+
                                                  |
                                                  v
+===================================================================================================+
|                                PROTOCOL 3: ACTIVE HUNTER EFFICIENCY                               |
| Task: Reduce pose uncertainty under high-entropy ambiguous initial conditions                     |
| Baseline: Random exploration yaw / fixed straight-line flight                                     |
| Metric Suite:                                                                                     |
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
                      DIAGNOSTIC CORRELATION CURVES
                      
  (a) Inlier Count vs. Pose Error         (b) Constellation Uniqueness vs. Entropy
  
  Pose Error (m)                          Entropy H(p(T))
       ^                                       ^
  50m  | *                                8.0  | *
       |  *                                    |  *
  20m  |   *                              4.0  |   *
       |     *                                 |     *
   5m  |       * * * * *                  1.0  |       * * * * *
   0m  +----------------------->               +----------------------->
       0   10   20   40   80                   0.0  0.2  0.5  0.8  1.0
          Verified Inlier Count                     Uniqueness Ratio (lambda_c)
```

### 3.1 Verified Inlier Count vs. Pose Error
- **Metric**: Number of 2D-3D correspondence ties surviving metric constellation verification.
- **Invariant**: Localization error must exhibit monotonic decrease as inlier count increases. If high inlier counts (\( > 30 \)) produce high pose errors (\( > 10 \text{ m} \)), it indicates geometric aliasing in the inverted index.

### 3.2 Constellation Uniqueness Ratio (\( \lambda_c \))
- **Definition**: The ratio of candidate spatial locations matching the observed metric constellation relative to the global code occurrence frequency:
  \[
  \lambda_c = \frac{1}{|\{ \mathbf{x} \in \mathcal{M} \mid \mathrm{ConstellationMatch}(\mathbf{x}, \mathcal{C}_{\mathrm{obs}}) \}|}
  \]
- **Invariant**: High uniqueness (\( \lambda_c \approx 1.0 \)) must strictly correlate with rapid Shannon entropy reduction in the Where-Am-I head.

---

## 4. Public Benchmark Evaluation & Ground-Truth Flight Validation

To ensure reproducibility, MonARC evaluates against four standardized public datasets alongside dedicated flight trials:

| Evaluation Suite | Domain / Ground Truth | Protocol Focus | Primary Metric |
| :--- | :--- | :--- | :--- |
| **OrthoLoC Benchmark** | 16,425 UAV images across 47 regions with DOPs/DSMs | 6-DoF pose estimation vs. orthographic geodata | Median Translation (m), Rotation (deg) |
| **University-1652** | 72 University campuses, paired drone-satellite | Cross-view retrieval baseline | Recall@1, AP (Retrieval baseline only) |
| **DenseUAV Benchmark** | Multi-altitude UAV-satellite imagery (80-100m AGL) | Cross-altitude scale robustness | Recall@1, Meter Error vs. GSD |
| **SUES-200 Benchmark** | Multi-altitude drone imagery (150-300m AGL) | High-altitude cross-view matching | Recall@1, Recall@5 |
| **Real-World Flight Trials** | Fixed-wing & multirotor with dual-frequency RTK-GPS | End-to-end 6-DoF GPS-denied trajectory | ATE (m), Drift (% distance), TTFF (s) |

---

## 5. Failure Analysis Reporting Standard

Every evaluation report generated in this repository must include the following breakdown:

```
================================================================================
MONARC EVALUATION RUN REPORT
Benchmark ID: EVAL-2026-08-28-REGION-OHIO-EAST
Test Extent: 100 km x 100 km bounding box (Spatial Holdout; 0% train overlap)
Reference Geodata: NAIP 2020 (Leaf-On) vs. Flight Test 2024 (Leaf-Off / Autumn)
Operating Altitude: 80m - 120m AGL
================================================================================
1. COLD-START RELOCALIZATION (Lost-in-Space, N = 1,000 trials)
   - Median Horizontal Error:       [Observed Value] m
   - 95th Percentile Error:         [Observed Value] m
   - Median Altitude (Z) Error:     [Observed Value] m
   - Median Yaw / Heading Error:    [Observed Value] deg
   - Time-to-First-Fix (TTFF):      [Observed Value] s
   - Gross Failure Rate (> 25m):    [Observed Value] %

2. CONTINUOUS 6-DoF TRACKING (Flight Length = 25.0 km)
   - Absolute Trajectory Error:     [Observed Value] m
   - Total Trajectory Drift:        [Observed Value] % of distance traveled
   - S2 Shard Query Latency:        [Observed Value] ms

3. ACTIVE HUNTER POLICY
   - Entropy Drop per Step:         [Observed Value] bits/action
   - Steps to Unimodal Fix:         [Observed Value] steps
   - Attractor Detour Distance:     [Observed Value] m
================================================================================
```
