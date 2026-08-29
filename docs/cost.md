# Cost Law: Corridor-First v1 Envelope

Date: 2026-08-29  
Status: Binding v1 Cost Law  
Repository: [abiome-org/MonARC](https://github.com/abiome-org/MonARC)  

---

## 1. Binding Rule

v1 is a **county-scale operational corridor**, not a continental raster factory. An agent reading this repository must build a Jefferson County / Colorado Front Range index and a public-benchmark training loop. Full-CONUS NAIP+3DEP ingest, a dense continental feature field, and day-one Sentinel-2 / international coverage are **expansion work**, not v1.

Treating continental coverage, `naip-source` archives, uncompressed historical stacks, or an ~80 TB CONUS fp16 grid as a build requirement is a spec violation. The four-subsystem architecture and three-tier data split remain; only the **geographic and storage scope** of v1 is compressed.

Dollar figures in this document are **planning envelopes, not invoices**. They are caps that constrain architecture. They are not measured cloud bills, quoted GPU rates, or performance claims.

---

## 2. v1 Budget Envelope

| Envelope | Scope | Status |
| :--- | :--- | :--- |
| **Target cap** | Cloud compute + object storage + data transfer for v1 ingest, Stage 1 codebook training on a sampled tile set, Stage 2 public-bench alignment, and corridor index export | **Under $2,000** |
| **Preferred** | Same work when the corridor is county-scale (Jefferson County CO / Front Range) and rasters are range-read in-region | **A few hundred dollars** |

What this envelope **does** cover:

- Range-read of JPEG visualization COGs and corridor 3DEP products in `us-west-2`.
- Frozen DINOv2 inference and fusion-stem + FSQ training on a **sampled** diverse tile set, then inference **only** on the v1 corridor.
- Stage 2 on University-1652, DenseUAV, SUES-200, and OrthoLoC.
- Stage 3 on a CPU frustum gym (laptop / workstation).
- Export of FSQ codes + inverted metric index (LMDB / S2 shards) for the corridor.

What this envelope **does not** cover, and must not be spent on in v1:

- Continental NAIP+3DEP walks, all historical NAIP years, or the 16 PB-class raw program archive.
- Egress copies of NAIP/3DEP to another cloud (R2, GCS, or a second AWS region).
- A dense continental fp16 / Zarr feature field.
- Custom flight-log campaigns, photorealistic renderers, or game-engine gyms.
- Foundation-model pretraining.
- A global map payload on the aircraft.

In-region CONUS-scale ingest priced in the tens of thousands of dollars, or six figures after egress, is the rejected path. Do not revive it as "the real ingest."

---

## 3. Coverage Rule

v1 ingest is **one operational corridor**.

- **Default corridor**: Jefferson County, Colorado (Colorado Front Range). Align the ingest bbox to Aflora's existing Jefferson County CO proof geometry. County borders are a **clip**, not landmark identity (Law of Emergent Visual Landmarks).
- Continental NAIP and 3DEP exist as **data availability**. That statement does not authorize walking CONUS.
- A full-CONUS inverted index is a later expansion: add corridors, re-run inference, merge shards. It is not a v1 gate.
- Sentinel-2 and international basemaps are **out of v1**.

Lost-in-space retrieval in v1 is uniform over the designated corridor, not over the continental United States.

---

## 4. Source Product Rule

### 4.1 NAIP

- Pull RGB from `s3://naip-visualization` (JPEG Cloud-Optimized GeoTIFF).
- **One vintage** covering the corridor. Do not ingest all historical years.
- Do not pull `naip-source` uncompressed GeoTIFFs.
- Do not require 0.3 m GSD when 0.6 m visualization tiles exist. Native visualization GSD (~0.6 m) is the v1 default.
- Do not target the 16 PB-class raw NAIP program archive.

### 4.2 3DEP

- Use the **cheapest product that still supplies metric \( z \)** inside the corridor bbox.
- Prefer already-COG 1/9 arc-second (~3 m) DSM/DEM.
- 1 m products are allowed **only** for the corridor bbox, and only if 1/9 arc-second is insufficient for metric constellation geometry in that box.
- Do not ingest CONUS 1 m lidar point clouds.

### 4.3 Overture / OSM

- Rasterize building footprints, road centerlines, and water polygons for the **corridor bbox only**.
- Discard names, tenant tags, and administrative labels.

---

## 5. Compute and Locality Rule

- Run GDAL warp, frozen DINOv2, fusion stem, FSQ, and index export in **AWS `us-west-2`**, next to the open-data buckets (`s3://naip-visualization`, 3DEP staged products, `s3://overturemaps-us-west-2`).
- **Range-read** COGs. Do not copy rasters to another cloud or region for v1.
- Aflora may store **source-byte pointers, small prefixes, and hashes** (warehouse). MonARC must **not duplicate** NAIP/3DEP rasters.
- GDAL warp + DINOv2 + codebook stay next to the model, on the corridor only.

---

## 6. Storage Rule

v1 **export** is:

1. Discrete FSQ codes for the corridor.
2. An inverted metric index (LMDB and/or S2 shards) mapping codes to \( (x, y, z) \) and co-visible bearings.

A continuous aerial feature field, if any, is **interpolated / on-demand** or a **small corridor Zarr**. It is not an ~80 TB CONUS grid and not a dense continental fp16 store.

Landmarks are **emergent extrema** in the fused field (corners, junctions, roof geometry, terrain texture peaks). Do not persist every DINOv2 token as a landmark.

---

## 7. Training Compute Rule

| Stage | v1 law | Forbidden in v1 |
| :--- | :--- | :--- |
| **Stage 1** | Train fusion stem + FSQ on a **sampled** diverse tile set (multiple biomes, still tiny versus CONUS). Then run inference **only** on the v1 corridor. Frozen DINOv2. | Walking the country. Training a new foundation backbone. Dense CONUS field pretrain. |
| **Stage 2** | Public thin pairs only: University-1652, DenseUAV, SUES-200, OrthoLoC. | Custom flight-log campaigns. Photorealistic renderer / 3DGS / GISNav / Unreal **unless** Stage 2 alignment on those public sets actually fails. |
| **Stage 3** | CPU frustum gym. Millions of episodes on a laptop or workstation. | Microsoft Flight Simulator, Unreal Engine, Unity, or any game-engine visual gym. |

---

## 8. Onboard Working-Set Rule

- The aircraft loads **corridor shards only**.
- One Jetson-class payload.
- **No global map** on the aircraft.
- S2 remains a query shard after the pose posterior concentrates, never landmark identity.

---

## 9. Forbidden Because It Explodes Cost

These paths remain banned (they were already expensive). v1 additionally forbids treating them as "later this sprint":

| Forbidden path | Why it explodes cost |
| :--- | :--- |
| 50 ft AGL CONUS photographic sweeps | Physical fleet + storage; already a non-goal. |
| YouTube / unstructured web video | Licensing, uncalibrated cameras, useless geotags. |
| End-to-end VLA (policy sees pixels) | Collapses isolation; trains on images instead of distribution tokens. |
| Street-level VPR datasets in the aerial codebook | Wrong manifold; wasted alignment compute. |
| H3 / geohash as landmark identity | Discrete buckets, not metric constellations. |
| `naip-source` + all years + 0.3 m mandate | Orders of magnitude more bytes than visualization COGs. |
| CONUS 1 m lidar point clouds | Point-cloud ingest, not corridor DSM COGs. |
| Egress copy of NAIP/3DEP to R2/GCS | Cross-cloud transfer dominates the bill. |
| Dense continental fp16 / Zarr field | Tens of terabytes stored; v1 does not need it. |
| Storing every DINOv2 token | Index size tracks the feature grid, not extrema. |
| Stage 1 CONUS walk | GPU hours scale with land area, not with codebook quality. |
| Game-engine / 3DGS gyms for Hunter | GPU-hour gym instead of CPU frustum. |
| Global map on the aircraft | Payload storage and paging for unused shards. |

---

## 10. Expansion (Not v1)

Keep the long-term shape: additional corridors, then a merged continental inverted index, then (if ever justified) denser products or international Sentinel-2. Expansion **adds corridors**. It does not rewrite v1 into a CONUS ingest job.

When expanding:

1. Keep range-read in `us-west-2`; still no wholesale raster duplication.
2. Keep FSQ codes + inverted index as the exported object; do not introduce a dense continental fp16 field as a prerequisite.
3. Keep Stage 1 as sampled training + per-corridor inference.
4. Keep existing bans (50 ft sweeps, YouTube, VLA, street VPR, H3-as-identity, game-engine Hunter gyms).
