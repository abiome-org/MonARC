# Cost Law: Colorado-State v1 Envelope

Date: 2026-08-30  
Status: Binding v1 Cost Law  
Repository: [abiome-org/MonARC](https://github.com/abiome-org/MonARC)  

---

## 1. Binding Rule

**v1 / MonARC-1 coverage is the state of Colorado** (state-boundary or state bbox clip). It is not Jefferson County. It is not CONUS.

An agent reading this repository must build a **Colorado-state** FSQ + inverted metric index and a public-benchmark training loop. Jefferson County / Colorado Front Range may be used as an **example ingest bbox** or a **first slice** inside Colorado (Aflora already has a Jefferson County CO proof). That slice is not the v1 product boundary.

**v2** is a CONUS inverted index, **gated on Colorado actually working**. Do not start CONUS ingest as a v1 job. Sentinel-2 / international coverage is out of v1.

Treating `naip-source` archives, uncompressed historical stacks, or a dense continental fp16 / Zarr field as a v1 build requirement is a spec violation. The four-subsystem architecture, the VLA ban, and the three-tier data split remain.

Dollar figures in this document are **planning envelopes, not invoices**. They are not measured cloud bills, quoted GPU rates, or performance claims. Do not invent a Colorado invoice.

---

## 2. v1 Budget Envelope

County-scale **"a few hundred dollars"** was a planning line for a Jefferson County **slice**, not the v1 product. v1 is Colorado-the-state.

| Envelope | Scope | Status |
| :--- | :--- | :--- |
| **v1 (Colorado)** | Cloud compute + object storage + data transfer for Colorado-state ingest (one NAIP vintage, visualization COGs, `us-west-2` range-read), Stage 1 codebook training on a sampled tile set, Stage 2 public-bench alignment, and Colorado FSQ + S2 index export | **Far below CONUS.** Still the cheap path: visualization COGs, no raster egress copy, no dense continental field. |
| **Rejected (CONUS-as-v1)** | Continental NAIP+3DEP walk, dense CONUS fp16 / Zarr, egress copy to another cloud | Tens of thousands in-region, six figures with egress. **Not v1.** Do not use this as the v1 bill. |

What this envelope **does** cover:

- Range-read of JPEG visualization COGs and 3DEP products for **Colorado** in `us-west-2`.
- Frozen DINOv2 inference and fusion-stem + FSQ training on a **sampled** diverse tile set, then inference on **Colorado** (a Jefferson County slice may run first; the product is the state).
- Stage 2 on University-1652, DenseUAV, SUES-200, and OrthoLoC.
- Stage 3 on a CPU frustum gym (laptop / workstation).
- Export of FSQ codes + inverted metric index (LMDB / S2 shards) for Colorado.

What this envelope **does not** cover, and must not be spent on in v1:

- Continental NAIP+3DEP walks, all historical NAIP years, or the 16 PB-class raw program archive.
- Egress copies of NAIP/3DEP to another cloud (R2, GCS, or a second AWS region).
- A dense **CONUS** fp16 / Zarr feature field.
- Custom flight-log campaigns, photorealistic renderers, or game-engine gyms.
- Foundation-model pretraining.
- Hardcoded geology or landcover class taxonomies as landmarks.
- A CONUS / global map payload on the aircraft.

Record actual spend when ingest runs. Do not invent invoices.

---

## 3. Coverage Rule

| Release | Coverage | Gate |
| :--- | :--- | :--- |
| **v1 / MonARC-1** | State of Colorado (state boundary or state bbox clip) | This release |
| **First slice (optional)** | Jefferson County / Front Range inside Colorado | Smoke-test / first Aflora-aligned ingest; **not** the product boundary |
| **v2** | CONUS inverted index | **Only after Colorado actually works** |
| **Out of v1** | Sentinel-2, international basemaps | Expansion after v2 if ever justified |

State and county borders are **clips**, not landmark identity (Law of Emergent Visual Landmarks).

Lost-in-space retrieval in v1 is uniform over the designated mission area **inside Colorado**, not over CONUS.

---

## 4. Source Product Rule

### 4.1 NAIP

- Pull RGB from `s3://naip-visualization` (JPEG Cloud-Optimized GeoTIFF).
- **One vintage** covering Colorado. Do not ingest all historical years.
- Do not pull `naip-source` uncompressed GeoTIFFs.
- Do not require 0.3 m GSD when 0.6 m visualization tiles exist. Native visualization GSD (~0.6 m) is the v1 default.
- Do not target the 16 PB-class raw NAIP program archive.

### 4.2 3DEP

- Use the **cheapest product that still supplies metric \( z \)** inside the Colorado clip.
- Prefer already-COG 1/9 arc-second (~3 m) DSM/DEM.
- 1 m products are allowed **only** inside Colorado, and only if 1/9 arc-second is insufficient for metric constellation geometry in that box.
- Do not ingest CONUS 1 m lidar point clouds.

### 4.3 Overture / OSM

- Rasterize building footprints, road centerlines, and water polygons for the **Colorado clip only**.
- Discard names, tenant tags, administrative labels, geology classes, and landcover class maps. Vector masks are geometric (footprint / centerline / water), not a semantic taxonomy.

---

## 5. Compute and Locality Rule

- Run GDAL warp, frozen DINOv2, fusion stem, FSQ, and index export in **AWS `us-west-2`**, next to the open-data buckets (`s3://naip-visualization`, 3DEP staged products, `s3://overturemaps-us-west-2`).
- **Range-read** COGs. Do not copy rasters to another cloud or region for v1.
- Aflora may store **source-byte pointers, small prefixes, and hashes** (warehouse). MonARC must **not duplicate** NAIP/3DEP rasters.
- GDAL warp + DINOv2 + codebook stay next to the model, on **Colorado** only (a first slice may run before the rest of the state).

---

## 6. Storage Rule

v1 **export** is:

1. Discrete FSQ codes for Colorado.
2. An inverted metric index (LMDB and/or S2 shards) mapping codes to \( (x, y, z) \) and co-visible bearings.

A continuous aerial feature field, if any, is **interpolated / on-demand** or a working-set grid. It is **not** a dense CONUS Zarr and not an ~80 TB continental fp16 store. Do not persist a dense statewide fp16 token grid as the product; the product is the index.

Landmarks are **emergent extrema** in the fused field (corners, junctions, roof geometry, terrain texture peaks). Do not persist every DINOv2 token as a landmark. Do not hardcode geology or landcover classes as the landmark vocabulary.

---

## 7. Training Compute Rule

| Stage | v1 law | Forbidden in v1 |
| :--- | :--- | :--- |
| **Stage 1** | Train fusion stem + FSQ on a **sampled** diverse tile set (multiple biomes, still tiny versus CONUS). Then run inference on **Colorado**. Frozen DINOv2. A Jefferson County slice may infer first. | Walking CONUS. Training a new foundation backbone. Dense CONUS field pretrain. Hardcoded geology/landcover classes. |
| **Stage 2** | Public thin pairs only: University-1652, DenseUAV, SUES-200, OrthoLoC. | Custom flight-log campaigns. Photorealistic renderer / 3DGS / GISNav / Unreal **unless** Stage 2 alignment on those public sets actually fails. |
| **Stage 3** | CPU frustum gym. Millions of episodes on a laptop or workstation. | Microsoft Flight Simulator, Unreal Engine, Unity, or any game-engine visual gym. |

---

## 8. Onboard Working-Set Rule

- Offline index: **Colorado** S2 shards.
- The aircraft loads **mission shards from the Colorado index**, not a CONUS / global map. Paging neighboring Colorado shards is allowed. One Jetson-class payload.
- S2 remains a query shard after the pose posterior concentrates, never landmark identity.

---

## 9. Forbidden Because It Explodes Cost

| Forbidden path | Why it explodes cost / breaks the law |
| :--- | :--- |
| 50 ft AGL CONUS photographic sweeps | Physical fleet + storage; already a non-goal. |
| YouTube / unstructured web video | Licensing, uncalibrated cameras, useless geotags. |
| End-to-end VLA (policy sees pixels) | Collapses isolation; trains on images instead of distribution tokens. |
| Street-level VPR datasets in the aerial codebook | Wrong manifold; wasted alignment compute. |
| H3 / geohash as landmark identity | Discrete buckets, not metric constellations. |
| Hardcoded geology / landcover classes | Semantic taxonomy, not emergent metric landmarks; extra labeled rasters. |
| `naip-source` + all years + 0.3 m mandate | Orders of magnitude more bytes than visualization COGs. |
| CONUS 1 m lidar point clouds | Point-cloud ingest, not Colorado DSM COGs. |
| Egress copy of NAIP/3DEP to R2/GCS | Cross-cloud transfer dominates the bill. |
| Dense CONUS fp16 / Zarr field | Tens of terabytes stored; v1 does not need it. |
| Storing every DINOv2 token | Index size tracks the feature grid, not extrema. |
| Stage 1 CONUS walk | GPU hours scale with land area, not with codebook quality. CONUS is v2, gated. |
| Treating Jefferson County as the v1 product | Underscopes MonARC-1. County is a slice, not the release. |
| Game-engine / 3DGS gyms for Hunter | GPU-hour gym instead of CPU frustum. |
| Global / CONUS map on the aircraft | Payload storage for unused shards. |

---

## 10. Expansion (v2, Not v1)

v2 is **probably CONUS**: add states, re-run inference, merge shards, **after Colorado works**. Expansion does not rewrite v1 into a CONUS ingest job and does not require a dense continental fp16 field as a prerequisite.

When expanding:

1. Keep range-read in `us-west-2`; still no wholesale raster duplication.
2. Keep FSQ codes + inverted index as the exported object.
3. Keep Stage 1 as sampled training + per-region inference (Colorado, then additional states).
4. Keep existing bans (50 ft sweeps, YouTube, VLA, street VPR, H3-as-identity, Unreal/MSFS Hunter gyms, hardcoded geology/landcover classes).
