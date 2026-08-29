# Data Architecture and Geodata Pipeline

Date: 2026-08-29  
Status: Data Law & Source Registry  
Repository: [abiome-org/MonARC](https://github.com/abiome-org/MonARC)  

---

## 0. v1 Coverage (Binding)

v1 ingest is **one operational corridor**, not CONUS. Default: Jefferson County, Colorado (Colorado Front Range). Align the bbox to Aflora's existing Jefferson County CO proof geometry. County borders are a clip, not landmark identity.

Continental NAIP and 3DEP exist as **data availability**. That does not authorize a continental walk, a dense CONUS field, or Sentinel-2 / international coverage in v1. A full-CONUS inverted index is expansion: add corridors later. Cost, products, and storage: [`docs/cost.md`](./cost.md).

---

## 1. Data Tier Hierarchy: Mass vs. Thin vs. Gym

MonARC enforces a three-tier data hierarchy. Conflating these tiers or using high-overhead game engines for policy training is prohibited. **v1 compresses Tier A volume; it does not delete the tier.**

```
+===================================================================================================+
|                                  TIER A: MASS 2D/2.5D GEODATA                                     |
|  Role: Offline Codebook Construction & Corridor Index                                              |
|  v1 Volume: One operational corridor (county-scale). Not terabytes-to-petabytes of CONUS.       |
|  v1 Sources (range-read in us-west-2; do not duplicate rasters):                                 |
|    - NAIP visualization COGs, one vintage, native ~0.6 m GSD (s3://naip-visualization)           |
|    - USGS 3DEP already-COG DSM/DEM for the corridor bbox (prefer 1/9 arc-second ~3 m)            |
|    - Overture Maps / OSM vector geometry clipped to the same bbox                                |
|  Availability (not a v1 build): Full CONUS NAIP+3DEP. Expansion only.                           |
|  Out of v1: Sentinel-2, international basemaps, naip-source, all historical years, CONUS 1 m lidar |
+===================================================================================================+
                                                  |
                                                  v
+===================================================================================================+
|                               TIER B: THIN PERSPECTIVE-ORTHO PAIRS                                |
|  Role: Cross-View Perspective Encoder Alignment & Confidence Calibration                          |
|  v1 Volume: Public benchmarks only (gigabytes; verified aerial-ortho matches)                   |
|  v1 Sources (required):                                                                            |
|    - University-1652 (Campus UAV-Satellite multi-view benchmark)                                |
|    - DenseUAV (Dense multi-altitude UAV-satellite visual localization)                            |
|    - SUES-200 (Multi-altitude UAV geo-localization benchmark)                                     |
|    - OrthoLoC (Paired UAV-geodata benchmark with DOPs, DSMs, and calibrated 6-DoF poses)      |
|  Out of v1: Custom calibrated flight-log campaigns. Optional later for operational validation.    |
+===================================================================================================+
                                                  |
                                                  v
+===================================================================================================+
|                               TIER C: ABSTRACT FRUSTUM GYM                                        |
|  Role: Active Perception Policy (Hunter) Optimization via MPPI & Imitation Learning           |
|  Volume: Millions of synthetic trajectory episodes on a laptop/workstation CPU                     |
|  Representation: 2.5D Landmark coordinates + discrete FSQ codes + camera frustum intersections        |
|  Simulation Engine: Pure mathematical Python/C++ frustum ray-caster. Zero 3D meshes, zero textures|
+===================================================================================================+
```

---

## 2. Tier A: Mass Geodata Specifications

The landmark field relies exclusively on authoritative federal geodata and open vector geometry. Scraping unverified commercial imagery or unstructured video feeds (e.g., YouTube) is forbidden.

v1 **reads** these products for the corridor bbox in `us-west-2`. Aflora may store source-byte pointers, small prefixes, and hashes. MonARC must not store a second copy of the rasters.

### 2.1 National Agriculture Imagery Program (NAIP)
- **Provider**: USDA Farm Service Agency / USGS EROS.
- **v1 Access**: AWS Open Data `s3://naip-visualization/` (JPEG Cloud-Optimized GeoTIFF). Range-read from `us-west-2`.
- **Forbidden for v1**: `naip-source` uncompressed GeoTIFFs; all historical years; the 16 PB-class raw program archive; a 0.3 m GSD mandate when 0.6 m visualization tiles exist.
- **v1 Resolution**: Native visualization GSD (~0.6 m). Do not resample the corridor to 0.3 m as a requirement.
- **Spectral Bands**: RGB from the visualization COGs. NIR is not required for v1.
- **Vintage**: **One** vintage covering the corridor.
- **License**: US Public Domain (no copyright restrictions).
- **Availability (not v1 ingest)**: NAIP exists across CONUS. Full-CONUS ingest is expansion.

### 2.2 USGS 3D Elevation Program (3DEP)
- **Provider**: United States Geological Survey.
- **Access Portal**: [USGS 3D Elevation Program (3DEP)](https://www.usgs.gov/3d-elevation-program) / AWS Open Data `s3://prd-tnm/StagedProducts/Elevation/`.
- **v1 Products**: The cheapest already-COG product that still supplies metric \( z \) inside the corridor bbox. Prefer 1/9 arc-second (~3 m) DSM/DEM. 1 m rasters are allowed only for that bbox, and only if ~3 m is insufficient for constellation geometry.
- **Forbidden for v1**: CONUS 1 m lidar point clouds; continental 1 m DSM mosaics.
- **Role in MonARC**: Provides true metric vertical coordinates \( z \) for the inverted index. MonARC prohibits synthetic height extraction via mono-depth foundation models on orthophotos when authoritative 3DEP exists in the corridor.
- **License**: US Public Domain.

### 2.3 Overture Maps Foundation & OpenStreetMap Geometry
- **Provider**: Overture Maps Foundation / OpenStreetMap Contributors.
- **Access Portal**: [Overture Maps Foundation](https://overturemaps.org/) / AWS S3 `s3://overturemaps-us-west-2/release/`.
- **v1 Extent**: Clip to the corridor bbox. Do not rasterize CONUS.
- **Layers Ingested**:
  - `buildings`: Polygon footprint outlines (names and tenant tags discarded).
  - `transportation`: Road network centerlines and class flags (motorway, primary, residential, unpaved).
  - `water`: River and shoreline boundary polygons.
- **Rasterization Schema**: Vector layers are rasterized into a multi-channel binary tensor aligned to the NAIP visualization grid at the native visualization GSD.
- **License**: Open Data Commons Open Database License (ODbL) / CDLA Permissive 2.0.

### 2.4 Sentinel-2 Multispectral Constellation (Out of v1)
- **Provider**: European Space Agency (ESA) Copernicus Programme.
- **Access Portal**: [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/).
- **Resolution**: 10 m GSD (RGB + NIR).
- **Role**: Expansion only — coarse context outside CONUS NAIP coverage. **Not a v1 source.**
- **License**: Free, full, and open access under EU Copernicus legal notice.

---

## 3. Tier B: Thin Perspective Pairs Registry

Perspective pairs are used exclusively for cross-view alignment of the perspective encoder. Synthetic 2D homography warps of nadir orthophotos do not model 3D relief displacement or building facades and are insufficient on their own.

**v1 Stage 2 uses the four public sets only.** A custom flight-log campaign is not required for v1 and must not be treated as a gate.

| Dataset | Primary Focus | Platform / Altitude | Scale | Source / Reference | v1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **University-1652** | Campus UAV-satellite cross-view matching | Multi-rotor UAV (~100–150m AGL) | 1,652 buildings across 72 university campuses | [University1652-Baseline](https://github.com/layumi/University1652-Baseline) | Required |
| **DenseUAV** | Dense multi-scale visual geo-localization | UAV flights at 80m, 90m, 100m AGL | 14,000+ paired multi-altitude images | [DenseUAV Repository](https://github.com/Zgt-d/DenseUAV) | Required |
| **SUES-200** | Cross-view UAV localization under altitude shift | UAV flights at 150m, 200m, 250m, 300m AGL | 200 scenes, 120k frames | [SUES-200-Benchmark](https://github.com/Reagan-Zhu/SUES-200-Benchmark) | Required |
| **OrthoLoC** | UAV 6-DoF localization vs. DOP & DSM geodata | Calibrated multi-rotor UAV (50–120m AGL) | 16,425 images across 47 regions in Germany & USA | [OrthoLoC Project](https://deepscenario.github.io/OrthoLoC/) | Required |
| **Calibrated Flight Logs** | Operational flight validation | Fixed-wing / VTOL with RTK-GPS | Proprietary test corridors | Internal flight logs | Expansion / validation; not a v1 training requirement |

---

## 4. Tier C: The Frustum Gym Environment

Training an active policy inside heavyweight photorealistic simulators (e.g., Unreal Engine, Microsoft Flight Simulator) creates severe sim-to-real visual domain gaps and is **forbidden** for Hunter training in every release, including v1.

MonARC trains the Hunter policy in a lightweight mathematical **Frustum Gym** on a laptop or workstation CPU:
- **World State**: An array of 3D points \( \mathbf{x}_m = (x_m, y_m, z_m) \) each carrying an FSQ landmark code \( c_m \in \{0, \dots, K-1\} \).
- **Agent State**: 6-DoF vehicle pose \( T_t \in \mathrm{SE(3)} \) and camera gimbal attitude \( (\theta_{\mathrm{gimbal}}, \phi_{\mathrm{gimbal}}) \).
- **Observation Operator**: A geometric camera frustum intersection test:
  \[
  \mathrm{Visible}(m, T_t) = \mathbb{I}\left( \mathbf{x}_m \in \mathrm{Frustum}(T_t, \mathbf{K}) \land \neg \mathrm{Occluded}(\mathbf{x}_m, \mathrm{DSM}) \right)
  \]
- **Noise Model**:
  - Code corruption: Observed codes subjected to random codebook distribution noise.
  - Dropout: Random landmark omission simulating temporary line-of-sight occlusion.
  - Geometric perturbation: 2D pixel coordinates perturbed by Gaussian noise.
- **Throughput**: High-frequency step execution on CPU, enabling rapid MPPI trajectory rollouts. Millions of episodes; no GPU gym requirement.

---

## 5. Optional Renderer for Encoder Domain Adaptation (Not v1 Default)

Photorealistic rendering packages are **not** part of the v1 plan. They are permitted **only if** Stage 2 alignment on the four public benches actually fails, and then **only** for synthetic sim-to-real adaptation of the Perspective Encoder (Tier B), never for the Hunter policy (Tier C):
- **GISNav**: [hmakelin/gisnav](https://github.com/hmakelin/gisnav) – Airborne map-matching bridge for PX4/ROS simulation.
- **Splat-Nav & SOUS VIDE / FiGS**: [Splat-Nav](https://github.com/chengine/splatnav) and [SOUS VIDE / FiGS](https://stanfordmsl.github.io/SousVide/) – 3D Gaussian Splatting simulators for high-speed perspective view generation.
- **TerrAInav / TOPO-DataGen**: Procedural terrain-generation frameworks for synthetic DEM and orthophoto pair synthesis.

Do not stand up Unreal, 3DGS, or GISNav because they exist. Record the Stage 2 failure first.

---

## 6. Aflora Automated Ingestion Pipeline

Aflora **does not copy** NAIP/3DEP rasters into a second object store. It may warehouse source-byte pointers, small prefixes, and hashes. GDAL warp, frozen DINOv2, fusion, FSQ, and index export run in **`us-west-2`** on the **v1 corridor only**.

```
                            AFLORA INGESTION PIPELINE (v1)
                            
  Pointers / hashes to:
  USGS 3DEP COGs (corridor bbox)  ----+
                                        |
  NAIP visualization COGs --------------+--> [GDAL / Rasterio Warper in us-west-2]
  (one vintage, range-read)            |    * Reproject to local UTM/EPSG
                                        |    * Align to native visualization GSD (~0.6 m)
  Overture Maps (GeoParquet, bbox) -----+    * Tile chunks for the corridor only
                                                 |
                                                 v
                                        [Inference Pipeline]
                                         * Frozen DINOv2 on RGB
                                         * Fusion Stem on DSM + Vector
                                         * FSQ Codebook Quantization
                                         * Landmark extrema (not every token)
                                                 |
                                                 v
                                        [Export Artifacts]
                                         * FSQ codes (corridor)
                                         * Inverted Metric Index (LMDB/S2 shards)
                                         * Optional: small corridor interpolated field
                                           (NOT a dense continental Zarr / fp16 grid)
```

Stage 1 **training** of the fusion stem and FSQ uses a **sampled** diverse tile set (multiple biomes, tiny versus CONUS). Stage 1 **inference** that writes the index runs only on the v1 corridor. See [`docs/training.md`](./training.md).

Expansion adds further corridors with the same pipeline. It does not require a CONUS raster factory as a prerequisite.
