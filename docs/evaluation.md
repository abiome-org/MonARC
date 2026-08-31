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

Matched query patch uv and gallery xyz are passed through the existing correspondence and PnP/LM path. `monarc fill-dsm-z --patches` derives each DINO cell center's longitude/latitude from the chip center, patch size, and window GSD, then batch-samples covering USGS TNM public-HTTPS 3DEP COGs with anonymous-SAS Planetary Computer `3dep-seamless` as fallback. It converts patch longitude, latitude, and sampled height together into local ENU and writes `patch_xyz.npy`; it does not persist a DSM raster or use requester-pays access. When that array exists, PnP uses finite per-patch ties from the candidate chip with the most mutual-nearest matches rather than unioning points across top-K candidates. JSON then reports `xyz_kind="per-patch-3dep"` and `xyz_is_chip_center=false`. Chip-center `xyz.npy` remains the retrieval and spatial-holdout coordinate.

If `patch_xyz.npy` is absent, compatibility behavior supplies one coarse xyz per chip, so all patches from a candidate share its chip-center coordinate. That geometry is commonly degenerate for 6-DoF PnP. JSON then keeps `xyz_kind="coarse-chip-center"` and `xyz_is_chip_center=true`. In both modes, `dsm_z_may_be_nan` reflects the chip-center array and non-finite PnP ties are excluded rather than assigned an invented elevation.

Each query records top-K ids and match counts, local match inlier count, refined horizontal xy, horizontal error against the query chip center, and whether PnP succeeded. When PnP/LM succeeds, refined xy is the first two ENU components of the camera-in-world translation obtained by inverting `pose_T_cw`; `xy_estimate_kind="pnp-horizontal"` identifies it. When PnP is underconstrained or fails, refined xy is the locally selected inlier candidate's chip center and the kind is `matched-chip-center-horizontal-fallback`. Separate matcher, PnP, and rank-1 fields and aggregates keep those errors comparable. Retrieve Recall@1/5 is recomputed for the candidate stage on the same run. `split.tiny` is true when `n_chips < 128` or `n_query < 32`.

This protocol is a map-cache diagnostic, not University-1652 and not GPS-denied flight ATE. Fixture comparisons are fixture results only; they neither establish nor refute performance on Golden–Morrison or statewide Colorado data. Report actual results only from a saved JSON artifact produced by the command.

### 1.4 Colorado place verification (`monarc eval-place-score`)

`monarc eval-place-score --extract <dir> --fsq <dir> --out <json>` measures the
same-place decision grain. Its default `stored-grid-crop` query kind slices
cached `features.npy` and `codes.npy`; this is a cache-alignment diagnostic, not
an independently encoded view.

The `reencoded-crop` query kind loads each gallery chip PNG,
applies the configured patch-margin crop and one-patch ordinal jitter, resizes
the pixel crop to the extract size with bilinear interpolation, and sends it
through frozen DINOv2-B/14 and the existing frozen Stage-1 FSQ checkpoint. For
a 224-pixel chip with patch size 14 and margin 2, the nominal 168-pixel crop is
resized to 224 pixels and produces a new 16-by-16 DINO grid. The gallery remains
the unchanged extract; it is never re-encoded by this evaluation. Downloads are
disabled by default, and the command records the query kind and local model
inputs in JSON.

The product-grain `reencoded-overlap` query kind instead selects the nearest
distinct gallery neighbor whose center is within one chip width, loads the
source neighbor PNG in full, and independently re-encodes it through the same
frozen DINO and FSQ path. Its true identity is the overlapping neighbor, and
the source chip is masked from ranking. It adds neither stored-grid crops nor
stored full-chip rows, so its headline AUROC and Recall@1 contain only genuine
neighbor-overlap positives against the spatially held-out far negatives.
This requires an ingest whose center spacing is smaller than the chip width.
`reencoded-crop` remains a same-PNG crop-and-resize test; it does not contain
pixels from an adjacent region.

With `--query-extract <dir>`, `reencoded-overlap` selects and re-encodes source
PNGs from that extract but scores them against every chip in the separate
`--extract`/`--fsq` gallery. This permits kilometer-scale far negatives without
re-extracting the gallery. Mixed-gallery truth is the nearest gallery center
within one chip width, otherwise the gallery footprint with the greatest
positive intersection area. If the query lies in a gallery-grid hole or
outside the gallery AOI and no footprint intersects, it has no positive:
`n_overlap_queries=0` is reported rather than assigning the nearest chip.

Cut mixed-gallery overlap queries from existing gallery chip footprints with
`ingest-aoi --align-to <gallery-manifest>`. Do not plan a new small AOI
independently: a regular coarse gallery can contain grid holes even when the
new AOI lies inside its bounding box. A result with `n_overlap_queries=0`
remains honest when query and gallery footprints do not intersect.

The gallery is the complement of the existing high-side spatial-box holdout.
For the crop query kinds, each selected gallery chip supplies one crop-jitter query whose true identity is
that gallery chip. Distinct gallery chips also supply spatial-overlap
queries when their horizontal center distance is no greater than
`chip_size_m = extract_size_px * gsd_m`; the JSON reports the observed count,
including zero. Self is excluded from this distinct-overlap count. The held-out
spatial box supplies geographically disjoint far queries and is used only for
negative pair scores.

A coarse AOI grid whose chip width is much smaller than its center spacing
honestly produces `n_overlap_queries=0`. Cropping or padding the same PNG must
not be used to manufacture adjacent-region overlap. Create a small, capped
range-read ingest with overlapping COG windows instead.

Bag-of-codes, mean-pooled frozen-DINO cosine, and sliding-window frozen-DINO
grid cosine each report same-place Recall@1 and AUROC against all far-query /
gallery pair scores. Recall@1 never includes far queries. Horizontal error is
reported only for same-place queries whose true gallery chip occurs in top-K.
Mutual-nearest DINO patch inlier totals and normalized rates use the cosine
threshold declared in the JSON. The top-level headline mirrors bag-of-codes;
all descriptor results remain available under `modes`.

This place-verification protocol is the product-bar grain. The retained
`eval-retrieve` spatial holdout asks whether a geographically separate chip can
retrieve its nearest gallery neighbor and is a different map diagnostic, not a
substitute for same-place overlap. Neither protocol is University-1652,
OrthoLoC, Colorado flight ATE, Hunter, or VLA evaluation. Documentation contains
no rehearsal result values; report only a saved artifact from a named run.

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
