# Data Architecture and Geodata Pipeline

Date: 2026-08-28  
Status: Data Law & Source Registry  
Repository: [abiome-org/MonARC](https://github.com/abiome-org/MonARC)  

---

## 1. Data Tier Hierarchy: Mass vs. Thin vs. Gym

MonARC enforces a three-tier data hierarchy. Conflating these tiers or using high-overhead game engines for policy training is prohibited.

```
+===================================================================================================+
|                                  TIER A: MASS 2D/2.5D GEODATA                                     |
|  Role: Offline Codebook Construction & Continuous Feature Field Pretraining                       |
|  Volume: Continental Scale (Terabytes to Petabytes of open geodata)                               |
|  Sources:                                                                                         |
|    - NAIP Orthophotography (0.3m-0.6m GSD, 4-band RGB-NIR)                                        |
|    - USGS 3DEP LiDAR & Digital Surface Models (1m-3m vertical resolution)                         |
|    - Overture Maps Foundation & OpenStreetMap Vector Geometry (Footprints, Centerlines)           |
|    - Sentinel-2 Multispectral Global Basemap (10m GSD fallback)                                  |
+===================================================================================================+
                                                  |
                                                  v
+===================================================================================================+
|                               TIER B: THIN PERSPECTIVE-ORTHO PAIRS                                |
|  Role: Cross-View Perspective Encoder Alignment & Confidence Calibration                          |
|  Volume: Focused Benchmarks (Gigabytes; thousands of verified aerial-ortho matches)               |
|  Sources:                                                                                         |
|    - University-1652 (Campus UAV-Satellite multi-view benchmark)                                  |
|    - DenseUAV (Dense multi-altitude UAV-satellite visual localization)                            |
|    - SUES-200 (Multi-altitude UAV geo-localization benchmark)                                     |
|    - OrthoLoC (Paired UAV-geodata benchmark with DOPs, DSMs, and calibrated 6-DoF poses)          |
|    - Real Calibrated Flight Logs (RTK-GPS aerial imagery at 80-150m AGL)                          |
+===================================================================================================+
                                                  |
                                                  v
+===================================================================================================+
|                               TIER C: ABSTRACT FRUSTUM GYM                                        |
|  Role: Active Perception Policy (Hunter) Optimization via MPPI & Imitation Learning               |
|  Volume: Millions of synthetic trajectory episodes simulated at > 10,000 FPS                      |
|  Representation: 2.5D Landmark coordinates + discrete FSQ codes + camera frustum intersections    |
|  Simulation Engine: Pure mathematical Python/C++ frustum ray-caster. Zero 3D meshes, zero textures|
+===================================================================================================+
```

---

## 2. Tier A: Mass Geodata Specifications

The landmark field relies exclusively on authoritative federal geodata and open vector geometry. Scraping unverified commercial imagery or unstructured video feeds (e.g., YouTube) is forbidden.

### 2.1 National Agriculture Imagery Program (NAIP)
- **Provider**: USDA Farm Service Agency / USGS EROS.
- **Access Portal**: [USGS EROS NAIP Archive](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-aerial-photography-national-agriculture-imagery-program-naip) / AWS Open Data `s3://naip-visualization/`.
- **Resolution**: 0.3 m to 0.6 m Ground Sample Distance (GSD).
- **Spectral Bands**: 4-Band (Red, Green, Blue, Near-Infrared).
- **Update Cycle**: Acquired during agricultural growing seasons on a 2-to-3-year cyclical state rotation.
- **License**: US Public Domain (no copyright restrictions).

### 2.2 USGS 3D Elevation Program (3DEP)
- **Provider**: United States Geological Survey.
- **Access Portal**: [USGS 3D Elevation Program (3DEP)](https://www.usgs.gov/3d-elevation-program) / AWS Open Data `s3://prd-tnm/StagedProducts/Elevation/`.
- **Products**: 1-meter and 1/9 arc-second (~3-meter) seamless Digital Surface Models (DSM) and bare-earth Digital Elevation Models (DEM) derived from airborne LiDAR.
- **Role in MonARC**: Provides true metric vertical coordinates \( z \) for the inverted index. MonARC prohibits synthetic height extraction via mono-depth foundation models on orthophotos when authoritative 3DEP LiDAR exists.
- **License**: US Public Domain.

### 2.3 Overture Maps Foundation & OpenStreetMap Geometry
- **Provider**: Overture Maps Foundation / OpenStreetMap Contributors.
- **Access Portal**: [Overture Maps Foundation](https://overturemaps.org/) / AWS S3 `s3://overturemaps-us-west-2/release/`.
- **Layers Ingested**:
  - `buildings`: Polygon footprint outlines (names and tenant tags discarded).
  - `transportation`: Road network centerlines and class flags (motorway, primary, residential, unpaved).
  - `water`: River and shoreline boundary polygons.
- **Rasterization Schema**: Vector layers are rasterized into a multi-channel binary tensor aligned to the NAIP grid at identical GSD.
- **License**: Open Data Commons Open Database License (ODbL) / CDLA Permissive 2.0.

### 2.4 Sentinel-2 Multispectral Constellation
- **Provider**: European Space Agency (ESA) Copernicus Programme.
- **Access Portal**: [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/).
- **Resolution**: 10 m GSD (RGB + NIR).
- **Role**: Coarse global context and global tile retrieval fallback for international regions outside CONUS NAIP coverage.
- **License**: Free, full, and open access under EU Copernicus legal notice.

---

## 3. Tier B: Thin Perspective Pairs Registry

Perspective pairs are used exclusively for cross-view alignment of the perspective encoder. Synthetic 2D homography warps of nadir orthophotos do not model 3D relief displacement or building facades and are insufficient on their own.

| Dataset | Primary Focus | Platform / Altitude | Scale | Source / Reference |
| :--- | :--- | :--- | :--- | :--- |
| **University-1652** | Campus UAV-satellite cross-view matching | Multi-rotor UAV (~100–150m AGL) | 1,652 buildings across 72 university campuses | [University1652-Baseline](https://github.com/layumi/University1652-Baseline) |
| **DenseUAV** | Dense multi-scale visual geo-localization | UAV flights at 80m, 90m, 100m AGL | 14,000+ paired multi-altitude images | [DenseUAV Repository](https://github.com/Zgt-d/DenseUAV) |
| **SUES-200** | Cross-view UAV localization under altitude shift | UAV flights at 150m, 200m, 250m, 300m AGL | 200 scenes, 120k frames | [SUES-200-Benchmark](https://github.com/Reagan-Zhu/SUES-200-Benchmark) |
| **OrthoLoC** | UAV 6-DoF localization vs. DOP & DSM geodata | Calibrated multi-rotor UAV (50–120m AGL) | 16,425 images across 47 regions in Germany & USA | [OrthoLoC Project](https://deepscenario.github.io/OrthoLoC/) |
| **Calibrated Flight Logs** | Operational flight validation | Fixed-wing / VTOL with RTK-GPS | Proprietary test corridors | Internal flight logs |

---

## 4. Tier C: The Frustum Gym Environment

Training an active policy inside heavyweight photorealistic simulators (e.g., Unreal Engine, Microsoft Flight Simulator) creates severe sim-to-real visual domain gaps and limits training throughput to ~60 FPS.

MonARC trains the Hunter policy in a lightweight mathematical **Frustum Gym**:
- **World State**: An array of 3D points \( \mathbf{x}_m = (x_m, y_m, z_m) \) each carrying an FSQ landmark code \( c_m \in \{0, \dots, K-1\} \).
- **Agent State**: 6-DoF vehicle pose \( T_t \in \mathrm{SE(3)} \) and camera gimbal attitude \( (\theta_{\mathrm{gimbal}}, \phi_{\mathrm{gimbal}}) \).
- **Observation Operator**: A geometric camera frustum intersection test:
  \[
  \mathrm{Visible}(m, T_t) = \mathbb{I}\left( \mathbf{x}_m \in \mathrm{Frustum}(T_t, \mathbf{K}) \land \neg \mathrm{Occluded}(\mathbf{x}_m, \mathrm{DSM}) \right)
  \]
- **Noise Model**:
  - Code corruption: With probability \( p_{\mathrm{noise}} = 0.15 \), observed code is replaced with a random code from the global codebook distribution.
  - Dropout: With probability \( p_{\mathrm{drop}} = 0.10 \), visible landmark is omitted.
  - Geometric perturbation: 2D pixel coordinates perturbed by Gaussian noise \( \epsilon_{uv} \sim \mathcal{N}(0, \sigma_{\mathrm{pixel}}^2) \).
- **Throughput**: Runs at > 100,000 steps per second on a single CPU core, enabling rapid MPPI trajectory rollouts.

---

## 5. Optional Renderer for Encoder Domain Adaptation

Photorealistic rendering packages are permitted **only** for synthetic sim-to-real cross-view domain adaptation of the Perspective Encoder (Tier B), never for the Hunter policy (Tier C):
- **GISNav**: [hmakelin/gisnav](https://github.com/hmakelin/gisnav) – Airborne map-matching bridge for PX4/ROS simulation.
- **Splat-Nav & SOUS VIDE / FiGS**: [Splat-Nav](https://github.com/chengine/splatnav) and [SOUS VIDE / FiGS](https://stanfordmsl.github.io/SousVide/) – 3D Gaussian Splatting simulators for high-speed perspective view generation.
- **TerrAInav / TOPO-DataGen**: Procedural terrain-generation frameworks for synthetic DEM and orthophoto pair synthesis.

---

## 6. Aflora Automated Ingestion Pipeline

Ingestion of continental-scale geodata is managed by the Aflora automated pipeline:

```
                            AFLORA INGESTION PIPELINE
                            
  USGS 3DEP (Cloud-Optimized GeoTIFFs)  ----+
                                            |
  NAIP Ortho (S3 COG Archive) --------------+--> [GDAL / Rasterio Warper]
                                            |    * Reproject to local UTM/EPSG
  Overture Maps (GeoParquet on S3) ---------+    * Snap to unified 0.5m grid
                                                 * Tile into 2048 x 2048 chunks
                                                         |
                                                         v
                                                [Inference Pipeline]
                                                 * Frozen DINOv2 on RGB
                                                 * Fusion Stem on DSM + Vector
                                                 * FSQ Codebook Quantization
                                                         |
                                                         v
                                                [Export Artifacts]
                                                 * Continuous Feature Field (Zarr)
                                                 * Inverted Metric Index (LMDB/S2)
```

Ingestion scripts require no manual human tagging. Processing a standard 50 km \( \times \) 50 km mission corridor requires approximately 45 minutes on a workstation with 4 \( \times \) NVIDIA RTX 4090 GPUs.
