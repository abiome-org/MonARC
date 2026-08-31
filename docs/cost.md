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

Dollar figures in this document are **planning envelopes or listed catalog prices, not invoices**. Listed GPU and storage rates are **as of late August 2026**. They are not a measured cloud bill. Do not invent a Colorado invoice.

---

## 2. v1 Budget Envelope

County-scale **"a few hundred dollars"** was a planning line for a Jefferson County **slice**, not the v1 product. v1 is Colorado-the-state.

| Envelope | Scope | Status |
| :--- | :--- | :--- |
| **v1 first pass (this stack)** | Colorado ingest on us-west-2 T4/L4 spot + Stage 1/2 on a couple of 4090-days + R2 codes/index | **About $40–$150** if hours stay in that band. Hours are the swing, not the hourly rate. Planning envelope, not an invoice. |
| **Rejected (CONUS-as-v1)** | Continental NAIP+3DEP walk, dense CONUS fp16 / Zarr, egress of the NAIP vintage to a neocloud | Tens of thousands in-region, six figures with egress. **Not v1.** Do not use this as the v1 bill. |

What this envelope **does** cover:

- Range-read of JPEG visualization COGs and 3DEP products for **Colorado** in `us-west-2`.
- Frozen DINOv2 inference and fusion-stem + FSQ training on a **sampled** diverse tile set, then inference on **Colorado** (a Jefferson County slice may run first; the product is the state).
- Stage 2 on University-1652, DenseUAV, SUES-200, and OrthoLoC.
- Stage 3 on a CPU frustum gym (laptop / workstation).
- Export of FSQ codes + inverted metric index (LMDB / S2 shards) for Colorado.

What this envelope **does not** cover, and must not be spent on in v1:

- Continental NAIP+3DEP walks, all historical NAIP years, or the 16 PB-class raw program archive.
- Pulling the Colorado NAIP vintage off AWS to a neocloud, or any raster egress copy of NAIP/3DEP.
- A dense **CONUS** fp16 / Zarr feature field; storing rasters on R2.
- Custom flight-log campaigns, photorealistic renderers, or game-engine gyms.
- Foundation-model pretraining; A100 / H100 / p4d / p5 for v1.
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

- **Data plane** (GDAL warp, frozen DINOv2 inference, fusion+FSQ **inference** on Colorado): **AWS `us-west-2` only**, next to `s3://naip-visualization`, 3DEP staged products, and `s3://overturemaps-us-west-2`.
- **Range-read** COGs. Do not copy rasters to another cloud or region. Do not pull the Colorado NAIP vintage off AWS to a neocloud.
- Aflora may store **source-byte pointers, small prefixes, and hashes** (warehouse). MonARC must **not duplicate** NAIP/3DEP rasters.
- **Train plane** (Stage 1 sampled tiles + Stage 2 public benches) may run on a cheap 4090 vendor. It does not receive a copy of the Colorado NAIP vintage.
- **Product store** (FSQ codes + inverted index) may live on Cloudflare R2. Rasters must not.
- Vendor SKUs, listed rates, and the NAT/gateway lock: **§6 Compute Vendors**.

---

## 6. Compute Vendors (Cheap 2026 Stack)

Rates below are **listed catalog prices as of late August 2026**, not a measured bill and not an invoice. Hours are the swing, not the hourly rate.

### 6.1 Data plane + frozen-DINO inference — AWS `us-west-2` only

Stay next to `s3://naip-visualization`, 3DEP, and Overture.

| Item | v1 lock | Listed price (late Aug 2026) |
| :--- | :--- | :--- |
| Prefer | Spot `g4dn.xlarge` (T4) | ~$0.25 / hr |
| Alternate | Spot `g6.xlarge` (L4) | ~$0.46 / hr |
| S3 → EC2 same-region | Required path | **$0** |
| VPC S3 gateway endpoint | **Required** | — |
| NAT Gateway for this path | **Forbidden** | $0.045 / GB |

Do not use on-demand p4d / p5, A100, or H100 on the data plane. Frozen DINOv2 over Colorado visualization COGs does not need them.

### 6.2 Train plane — Stage 1 sampled tiles + Stage 2 public benches

| Item | v1 lock | Listed price (late Aug 2026) |
| :--- | :--- | :--- |
| Default | Runpod RTX 4090 Community | $0.34 / hr (page updated 2026-08-27) |
| Allowed | Vast.ai or Salad RTX 4090 if cheaper than that listed Runpod rate | listed at booking |
| Forbidden | A100, H100, `p4d`, `p5` | — |

Do **not** pull the Colorado NAIP vintage off AWS onto the train plane. That egress is listed **$0.09 / GB** and is the bill-exploding path. Stage 1 uses a sampled tile set (tiny versus the vintage). Stage 2 uses University-1652, DenseUAV, SUES-200, and OrthoLoC.

### 6.3 Product store — Cloudflare R2

| Item | v1 lock | Listed price (late Aug 2026) |
| :--- | :--- | :--- |
| Object store | Cloudflare R2 for FSQ codes + inverted metric index only | $0.015 / GB-month; **10 GB free**; **$0 egress** |
| Size target | Colorado codes + index | **Free tier** |
| Rasters | **Do not store** NAIP, 3DEP, or other rasters on R2 | — |

Aflora warehouse pointers/hashes may stay where Aflora already keeps them. The MonARC **product** artifact on R2 is the index, not a raster mirror.

### 6.4 First-pass planning envelope (this stack)

**About $40–$150** if GPU hours stay in the T4/L4 spot band plus a couple of 4090-days. That is a planning envelope, not an invoice. If hours blow past that band, the SKU was not the problem.

### 6.5 Forbidden on this stack

- Pulling the Colorado NAIP vintage off AWS to Runpod / Vast / Salad / any neocloud ($0.09 / GB listed egress).
- `naip-source`, all historical years, dense fp16 / Zarr field.
- NAT Gateway in front of S3 for the data plane ($0.045 / GB).
- A100 / H100 / `p4d` / `p5` for v1.
- Storing rasters on R2.

---

## 7. Storage Rule

v1 **export** is:

1. Discrete FSQ codes for Colorado.
2. An inverted metric index (LMDB and/or S2 shards) mapping codes to \( (x, y, z) \) and co-visible bearings.

A continuous aerial feature field, if any, is **interpolated / on-demand** or a working-set grid. It is **not** a dense CONUS Zarr and not an ~80 TB continental fp16 store. Do not persist a dense statewide fp16 token grid as the product; the product is the index.

The export may be stored on **Cloudflare R2** (codes + index only; target the 10 GB free tier). Do not store rasters on R2.

Landmarks are **emergent extrema** in the fused field (corners, junctions, roof geometry, terrain texture peaks). Do not persist every DINOv2 token as a landmark. Do not hardcode geology or landcover classes as the landmark vocabulary.

Landmarks are **emergent extrema** in the fused field (corners, junctions, roof geometry, terrain texture peaks). Do not persist every DINOv2 token as a landmark. Do not hardcode geology or landcover classes as the landmark vocabulary.

---

## 8. Training Compute Rule

| Stage | v1 law | Forbidden in v1 |
| :--- | :--- | :--- |
| **Stage 1** | Train fusion stem + FSQ on a **sampled** diverse tile set (multiple biomes, still tiny versus CONUS) on a 4090 vendor (§6). Then run **inference** on **Colorado** in us-west-2 (T4/L4). Frozen DINOv2. A Jefferson County slice may infer first. | Walking CONUS. Training a new foundation backbone. Dense CONUS field pretrain. Hardcoded geology/landcover classes. Pulling the Colorado NAIP vintage to the train plane. |
| **Stage 2** | Public thin pairs only: University-1652, DenseUAV, SUES-200, OrthoLoC. Same 4090 train plane (§6). | Custom flight-log campaigns. Photorealistic renderer / 3DGS / GISNav / Unreal **unless** Stage 2 alignment on those public sets actually fails. A100 / H100 / `p4d` / `p5`. |
| **Stage 3** | CPU frustum gym. Millions of episodes on a laptop or workstation. | Microsoft Flight Simulator, Unreal Engine, Unity, or any game-engine visual gym. |

---

## 9. Onboard Working-Set Rule

- Offline index: **Colorado** S2 shards.
- The aircraft loads **mission shards from the Colorado index**, not a CONUS / global map. Paging neighboring Colorado shards is allowed. One Jetson-class payload.
- S2 remains a query shard after the pose posterior concentrates, never landmark identity.

---

## 10. Forbidden Because It Explodes Cost

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
| Egress of the Colorado NAIP vintage to a neocloud | Listed $0.09 / GB (late Aug 2026). Inference stays in us-west-2. |
| NAT Gateway in front of S3 (data plane) | Listed $0.045 / GB. Use a VPC S3 gateway endpoint. |
| A100 / H100 / `p4d` / `p5` for v1 | Wrong SKU class; T4/L4 + 4090 is the stack. |
| Storing rasters on R2 | R2 is the codes+index store. Rasters stay on AWS open-data. |
| Egress copy of NAIP/3DEP rasters to R2/GCS | Cross-cloud raster transfer dominates the bill. Codes+index on R2 is allowed. |
| Dense CONUS fp16 / Zarr field | Tens of terabytes stored; v1 does not need it. |
| Storing every DINOv2 token | Index size tracks the feature grid, not extrema. |
| Stage 1 CONUS walk | GPU hours scale with land area, not with codebook quality. CONUS is v2, gated. |
| Treating Jefferson County as the v1 product | Underscopes MonARC-1. County is a slice, not the release. |
| Game-engine / 3DGS gyms for Hunter | GPU-hour gym instead of CPU frustum. |
| Global / CONUS map on the aircraft | Payload storage for unused shards. |

---

## 11. Expansion (v2, Not v1)

v2 is **probably CONUS**: add states, re-run inference, merge shards, **after Colorado works**. Expansion does not rewrite v1 into a CONUS ingest job and does not require a dense continental fp16 field as a prerequisite.

When expanding:

1. Keep range-read in `us-west-2` with an S3 gateway (no NAT); still no wholesale raster duplication and no NAIP vintage on a neocloud.
2. Keep FSQ codes + inverted index as the exported object (R2 is for that product, not rasters).
3. Keep Stage 1 as sampled training + per-region inference (Colorado, then additional states).
4. Keep existing bans (50 ft sweeps, YouTube, VLA, street VPR, H3-as-identity, Unreal/MSFS Hunter gyms, hardcoded geology/landcover classes).

---

## 12. Golden–Morrison rehearsal (this executable increment)

The 10×10 km Golden–Morrison box (center approximately 39.725°N, 105.220°W; 100 km²) is a **rehearsal / first slice** for CPU dry-run ingest: STAC ∩ TNMAccess manifest, tiny FSQ, code→xyz index. It sits inside Colorado (Front Range). It does **not** rewrite v1 coverage. Jefferson County remains a slice, not the product. CONUS remains v2.

This increment:

- Writes an AOI **manifest** of intersecting NAIP visualization COGs and 3DEP inventory records. Tile IDs come from catalogs at launch time, not a hardcoded list.
- Persists FSQ codes, metric xyz, and compact metadata. Does not download or store rasters. Does not persist `naip-source`. Does not walk the full state.
- Adds no Terraform. NAT Gateway, SageMaker, and EKS remain forbidden (§6.5). The cheap 2026 vendor lock in §6 is unchanged.

The $150 figure attached to this box is the **slice rehearsal envelope**, inside the v1 first-pass planning band in §2. It is not an invoice and not a performance claim.

Colorado retrieval reports and public-UAV adapter reports stay on separate tracks ([`evaluation.md`](./evaluation.md)). This file does not record Recall@1, median translation, or other numeric gates.
