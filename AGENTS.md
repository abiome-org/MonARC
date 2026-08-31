# AGENTS.md: Developer and Agent Operating Invariants

Date: 2026-08-30  
Repository: [abiome-org/MonARC](https://github.com/abiome-org/MonARC)  
Tone & Style: Strict engineering specification. No marketing text. No fabricated metrics.

---

## 1. Operating Laws and Invariants

Every autonomous agent and human contributor operating in this repository must adhere to the following non-negotiable architectural and engineering laws:

1. **Law of Emergent Visual Landmarks (No Human POIs)**:
   - Landmarks are emergent clusters in metric visual feature space (corners, road junctions, roof geometry, terrain textures).
   - Never inject human semantic points of interest (restaurants, administrative names, Wikipedia articles, county or state borders as landmarks). Semantic entities change arbitrarily and do not provide stable metric anchor points.
   - Do not hardcode geology or landcover class taxonomies as the landmark vocabulary.
   - A state or county bounding box may be used as an **ingest clip**. v1 product clip is the **state of Colorado**. Jefferson County / Front Range may be a first slice inside Colorado. Clips are not landmark identity.
2. **Law of Metric Constellations (No H3/Geohash Representation)**:
   - The map representation is not a discrete spatial bucket (H3 hexagon, S2 cell, or geohash bin).
   - The primary **v1 export** is an inverted index of landmark codes mapped to 3D metric coordinates \( (x, y, z) \) and relative bearing vectors to co-visible neighbors. A continuous feature field, if present, is interpolated/on-demand or a working-set grid — never a dense CONUS fp16 / Zarr store.
   - S2 indexing is permitted **only** as a spatial partitioning shard for runtime query acceleration during continuous tracking *after* the pose posterior has established concentrated mass. S2 must never be used to define landmark identity.
   - Landmarks are emergent extrema in fused feature space, not every DINOv2 token.
3. **Law of Ingestion Channel Separation (No Frozen DINO on 6 Channels)**:
   - Foundation visual encoders (e.g., DINOv2) are pretrained on 3-channel RGB. Passing concatenated 6-channel tensors (RGB + Elevation + OSM vectors) directly into a frozen RGB backbone is structurally invalid.
   - RGB imagery must flow through the frozen vision backbone. Elevation rasters (USGS 3DEP) and rasterized vector geometry (Overture/OSM) must pass through a lightweight trainable fusion stem. Fusion occurs *prior* to vector quantization.
4. **Law of Policy Isolation (Policy Never Sees Pixels)**:
   - The active vision policy ("Hunter") receives only distribution tokens (posterior entropy, mode positions, covariance vectors, rim landmark codes).
   - Under no circumstance may the policy take raw RGB images or feature maps as inputs.
   - Do not collapse perception, localization, and planning into a monolithic Vision-Language-Action (VLA) architecture.
5. **Law of Frustum Simulation (No Flight Simulator Bloat)**:
   - The Hunter policy is trained inside a mathematical frustum gym operating directly on the 2.5D landmark map with simulated occlusion, sensor noise, and code collision distributions.
   - Never build or mandate game engine environments (e.g., Unreal Engine, Unity, Microsoft Flight Simulator) for policy optimization.
6. **Law of Information-Theoretic Reward (No Water-Tower Attractors)**:
   - The reward function for active exploration is strictly the expected reduction in pose posterior entropy (\( \Delta H \)).
   - Visiting a unique or rare landmark constellation is a secondary bonus, never the primary objective. Policies trained on uniqueness rewards degenerate into non-viable attractor loops (e.g., continually seeking out isolated water towers or anomalous radio masts).
7. **Law of Empirical Integrity (No Fabricated Metrics)**:
   - Never report imaginary performance gates (e.g., "achieves 99.4% recall across CONUS").
   - Distinguish strictly between development benchmarks on limited public datasets (e.g., University-1652 campus retrieval) and verified 6-DoF GPS-denied flight performance.
   - Every metric stated in documentation or pull requests must trace to an executable test script, dataset split, and recorded evaluation artifact.
8. **Law of One-State Cost (Colorado v1; No CONUS Raster Factory)**:
   - **v1 / MonARC-1 coverage is the state of Colorado** (state-boundary or state bbox clip). Jefferson County / Front Range may be a first slice or example ingest bbox; it is **not** the v1 product boundary. Continental NAIP+3DEP is a data-availability statement, not a v1 ingest requirement. **v2 is CONUS**, gated on Colorado actually working. Sentinel-2 / international coverage is out of v1.
   - Pull one NAIP vintage. **Rehearsal / first slice (Golden–Morrison) defaults to Microsoft Planetary Computer STAC collection `naip` with anonymous SAS** (`https://planetarycomputer.microsoft.com/api/sas/v1/token/naip`). Do not require AWS billed credentials or an AWS shared-credentials file for that slice. Optional unsigned fallback: `s3://colorado-public-imagery` (`--no-sign-request` / HTTPS list). The AWS `s3://naip-visualization` JPEG COG path is an explicit flag only. Do not ingest `naip-source`, all historical years, or a 0.3 m mandate when ~0.6 m tiles exist. Prefer already-COG 3DEP 1/9 arc-second (~3 m) or 1 m **only** inside Colorado, from TNM public HTTPS or Planetary Computer `3dep-seamless` — not requester-pays `s3://prd-tnm` if it demands an account. Do not ingest CONUS 1 m lidar point clouds.
   - Process range-read COG **chips** only. Do not egress-copy full NAIP/3DEP GeoTIFFs. Codes+index may land on R2. Aflora may store source-byte pointers / small prefixes / hashes; MonARC must not duplicate the rasters. **R2 is codes+index only (no rasters).** A later statewide AWS `us-west-2` data plane next to open-data buckets remains allowed; it is **not** required for the Golden–Morrison rehearsal.
   - v1 export is FSQ codes + inverted metric index (LMDB/S2 shards) for Colorado. Do not store a dense CONUS fp16 / Zarr feature field. Do not hardcode geology or landcover classes.
   - Stage 1 trains fusion stem + FSQ on a sampled diverse tile set, then infers on Colorado. Frozen DINOv2. No foundation-model pretrain. Stage 2 uses public thin pairs only (University-1652, DenseUAV, SUES-200, OrthoLoC); no custom flight-log campaign. Stage 3 is a CPU frustum gym on a laptop/workstation. Onboard working set is Colorado mission shards on one Jetson-class payload; no CONUS/global map on the aircraft.
   - Cheap 2026 stack (listed prices as of late Aug 2026, not a measured bill): data plane AWS us-west-2 spot `g4dn.xlarge` / `g6.xlarge` with a VPC S3 gateway (no NAT); train plane Runpod/Vast/Salad RTX 4090, not A100/H100/`p4d`/`p5`; product store Cloudflare R2 (codes+index, target free tier). See [`docs/cost.md`](./docs/cost.md) §6 Compute Vendors.
   - Planning envelope: about $40–$150 first pass if hours stay in the T4/L4 + couple-of-4090-days band. Hours are the swing. County-scale "few hundred dollars" was a slice line, not the v1 product. Do not invent invoices. Binding detail: [`docs/cost.md`](./docs/cost.md).
   - The Golden–Morrison 10×10 km box (center ~39.725°N, 105.220°W) is a **$150 rehearsal / first slice** inside Colorado for CPU ingest dry-run **without AWS billed accounts**. It does not rewrite v1 coverage. See [`docs/cost.md`](./docs/cost.md) §12.

---

## 2. Codebase Map and Module Layout

Even when components are undergoing active development or migration, directory structures must reflect this layout:

```
MonARC/
+-- AGENTS.md                   # Agent operating rules and architectural invariants
+-- README.md                   # Project summary, architecture overview, pointers
+-- LICENSE                     # MIT License
+-- .gitignore                  # Git ignore patterns for ML, geodata, and artifacts
+-- docs/                       # Comprehensive specifications
|   +-- product.md              # Operational envelope, problem definition, failure modes
|   +-- architecture.md         # Subsystem contracts, tensor I/O, fusion stem, GLACE
|   +-- cost.md                 # v1 Colorado cost law, budget envelope, forbidden explosions
|   +-- data.md                 # Data tiers, geodata sources, licensing, Aflora pipeline
|   +-- training.md             # 3-stage training pipeline, loss formulas, freeze schedule
|   +-- evaluation.md           # Evaluation protocols, holdout splits, baseline standards
|   +-- onboard.md              # Flight payload compute, execution profiling, memory models
|   +-- cost.md                 # Colorado product boundary, rehearsal slice, cheap stack
|   +-- literature.md           # Annotated bibliography with verified URLs
|   +-- non-goals.md            # Out-of-scope capabilities and architectural anti-patterns
+-- monarc/                     # Core Python library
|   +-- __init__.py
|   +-- common/                 # Math primitives, SE(3) manifolds, coordinate transforms
|   |   +-- se3.py              # Lie group SE(3) and Lie algebra se(3) operations
|   |   +-- coordinates.py      # WGS84, UTM, local NED conversions, camera projection
|   |   +-- frustum.py          # Geometric frustum intersection on 2.5D terrain
|   +-- map/                    # Subsystem 1: Map ingestion, feature field, FSQ indexing
|   |   +-- dino_backbone.py    # Frozen DINOv2 / vision feature extractor
|   |   +-- fusion_stem.py      # Trainable raster fusion stem for DSM + vector masks
|   |   +-- quantizer.py        # Finite Scalar Quantization (FSQ) and residual VQ
|   |   +-- continuous_field.py # On-demand / working-set interpolated field (not CONUS store)
|   |   +-- metric_index.py     # Inverted index (code -> xyz + metric constellations)
|   |   +-- s2_shard.py         # S2-based spatial sharding for tracking-mode acceleration
|   +-- perspective/            # Subsystem 2: Perspective encoder & cross-view alignment
|   |   +-- encoder.py          # Perspective feature extractor with BEV projection head
|   |   +-- slot_head.py        # Sparse landmark code emission & slot attention
|   |   +-- calibrator.py       # Temperature scaling / isotonic confidence calibration
|   +-- localization/           # Subsystem 3: Where-am-I estimation head
|   |   +-- global_retrieve.py  # Lost-in-space bag-of-codes / frozen-DINO chip retrieve
|   |   +-- eval_retrieve.py    # Colorado-track bag-of-codes + frozen-DINO retrieve on spatial chip holdout
|   |   +-- eval_match_pnp.py   # Top-K local DINO-grid matches + PnP/LM with chip-center xy fallback
|   |   +-- dpnp.py             # Differentiable PnP particle initializer
|   |   +-- perceiver.py        # Set transformer for correspondence-to-pose cross-attention
|   |   +-- posterior.py        # SE(3) particle filter and Gaussian mixture representations
|   +-- hunter/                 # Subsystem 4: Active perception policy
|   |   +-- env/                # Abstract 2.5D landmark frustum gym
|   |   +-- mppi_expert.py      # MPPI / CEM trajectory optimizer on entropy reduction
|   |   +-- policy.py           # Compact transformer policy cloned from MPPI rollouts
|   +-- data/                   # Dataset loaders and ingestion utilities
|       +-- aflora_ingest.py    # NAIP, 3DEP, and Overture rasterization pipeline
|       +-- uav_benchmarks.py   # Loaders for University-1652, DenseUAV, SUES-200, OrthoLoC
+-- tests/                      # Unit, integration, and mathematical invariant tests
|   +-- test_se3.py             # SE(3) exponential/logarithmic map consistency
|   +-- test_fusion_stem.py     # Channel fusion dimensionality and gradient flow
|   +-- test_fsq.py             # Codebook quantization determinism and collision rates
|   +-- test_frustum.py         # Camera ray terrain intersection accuracy
|   +-- test_eval_retrieve.py   # Colorado-track spatial holdout Recall@K and xyz error (bag + DINO)
|   +-- test_eval_match_pnp.py  # CPU top-K patch matching, NaN-z fallback, and JSON/CLI contract
|   +-- test_perceiver.py       # Where-am-I permutation invariance over correspondences
|   +-- test_hunter_env.py      # Frustum gym state transitions and information gain
```

---

## 3. SOTA Verification Protocol (`j8ckfi/library`)

When considering baseline updates or evaluating modern replacements for sub-components, query the local research library (`j8ckfi/library`) or canonical references for current state-of-the-art standards:

- **Quantization**: Finite Scalar Quantization (FSQ; Mentzer et al., 2023) is the required quantization primitive. Do not revert to classical codebook-lookup Vector Quantization (VQ-VAE) with codebook collapse heuristics unless residual quantization ablation justifies it.
- **Visual Backbones**: DINOv2 / DINOv3 frozen features are standard. Fine-tuning must use parameter-efficient LoRA adapters on attention projections, keeping core spatial feature representations intact.
- **Global Retrieval**: Use MegaLoc or AnyLoc-class foundation descriptors for coarse tile retrieval in lost-in-space mode. Do not handcraft NetVLAD dictionaries from scratch.
- **Cross-View Geometry**: Scene coordinate regression without explicit 3D supervision must address the fundamental invariance vs. discrimination dilemma (GLACE; Wang et al., CVPR 2024).

---

## 4. How to Add a Dataset

To integrate a new dataset into the MonARC pipeline:

1. **Verify Licensing**: The data must possess an open commercial/research license (e.g., US Public Domain, CC-BY, Open Government Licence). Datasets scraped from consumer video platforms (e.g., YouTube) without explicit redistribution rights are forbidden.
2. **Georeference Verification**: Every image must possess verified 6-DoF ground truth poses or geodetic coordinates tied to real-world WGS84/UTM datums. Synthetic homography warps of nadir imagery cannot substitute for oblique flight sets.
3. **Document in `docs/data.md`**: Add the dataset specification, source URL, resolution, coverage extent, and license class to [`docs/data.md`](./docs/data.md).
4. **Respect the v1 cost law**: A new dataset does not authorize continental ingest, raster duplication, or a custom flight campaign. v1 Stage 2 remains the four public UAV benches. See [`docs/cost.md`](./docs/cost.md).
5. **Implement Data Loader**: Place parsing and caching logic under `monarc/data/` accompanied by unit tests validating coordinate frame conversions.

---

## 5. How to Modify Architecture

Architectural changes must preserve the four-part isolation contract:

1. **RFC in `docs/architecture.md`**: Before making code modifications that alter inter-subsystem data formats, update the corresponding tensor signature in [`docs/architecture.md`](./docs/architecture.md).
2. **Never Merge Subsystems**: Do not merge the Perspective Encoder and Where-Am-I head into an end-to-end regression network. Do not feed pixels to the Hunter policy.
3. **Verify Tensor Contracts**: Ensure updated modules pass all interface shape and type validation tests under `tests/`.
4. **Do Not Widen v1 Ingest**: Geographic coverage, raster products, storage artifacts, and training-set scale must remain inside [`docs/cost.md`](./docs/cost.md). v1 is Colorado-the-state, not one county and not CONUS. CONUS ingest, dense continental fields, and renderer/flight-log campaigns are v2+ expansion, gated on Colorado working.
