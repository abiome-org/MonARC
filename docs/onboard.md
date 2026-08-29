# Onboard Embedded Runtime and SWaP-C Specifications

Date: 2026-08-29  
Status: Embedded Hardware Specification  
Repository: [abiome-org/MonARC](https://github.com/abiome-org/MonARC)  

---

## 1. Qualitative Onboard vs. Offline Compute Split

MonARC establishes a clear asymmetry between offline index generation and onboard execution:
- **Offline Infrastructure (v1)**: Range-reads NAIP visualization COGs (one vintage), corridor 3DEP DSM, and Overture vectors in `us-west-2`. Trains fusion stem + FSQ on a sampled tile set with frozen DINOv2. Infers FSQ codes on the **v1 corridor only** (default: Jefferson County / Colorado Front Range) and writes an inverted metric index (LMDB/S2 shards). Does not ingest CONUS, does not store a dense continental fp16 field, and does not duplicate rasters. Cost law: [`docs/cost.md`](./cost.md).
- **Onboard Flight Payload**: One Jetson-class System-on-Module. The drone **never carries a global map**. Working set is the **corridor shards only**. It runs a thin perspective encoder, a corridor S2 cache, a compact Perceiver Where-Am-I estimator, and a lightweight Hunter policy transformer.

```
+===================================================================================================+
|                                  ONBOARD HARDWARE ARCHITECTURE                                    |
|                                                                                                   |
|  [Global Shutter Camera]      [Tactical IMU]         [Altimeter / Barometer]                      |
|       | Real-Time Frames           | High-Rate Stream           | Surface / Altitude Stream       |
|       | MIPI CSI-2 / GMSL2         | SPI / UART                 | I2C                             |
|       v                            v                            v                                 |
|  +---------------------------------------------------------------------------------------------+  |
|  | EMBEDDED SYSTEM-ON-MODULE (SOM) — one Jetson-class payload                                  |  |
|  |                                                                                             |  |
|  | [Neural & Accelerated Perception Pipeline]       [Real-Time State Filter & I/O Thread]      |  |
|  |   1. Perspective Encoder (Frozen ViT-S/B + FSQ)    1. High-Rate Inertial Propagation       |  |
|  |   2. S2 Inverted Index Shard Lookup                2. Dynamic S2 Shard Working Set Cache   |  |
|  |   3. Perceiver Where-Am-I Estimation Head          3. SE(3) Posterior Maintenance          |  |
|  |   4. Hunter Mode-Attention Transformer Policy      4. Autopilot MAVLink / ROS Bridge       |  |
|  |                                                                                             |  |
|  | [Unified System Memory]                                                                     |  |
|  |   - Compact Neural Network Model Weights                                                    |  |
|  |   - Dynamic Execution Scratchpad Buffers                                                    |  |
|  |   - Local Corridor S2 Index Shards (Paged from Onboard Storage)                             |  |
|  +---------------------------------------------------------------------------------------------+  |
|                                            |                                                      |
|                                            v MAVLink / CAN / Serial                               |
|                               [Flight Controller (e.g., PX4 / ArduPilot)]                         |
+===================================================================================================+
```

---

## 2. Onboard Execution Pipeline and Latency Protocols

The onboard flight pipeline runs as an asynchronous dual-rate estimation system:
1. **High-Frequency Inertial Thread**: Integrates IMU kinematics at high rate to propagate the SE(3) pose distribution forward in time between camera observations.
2. **Visual State Update Loop**: Processes camera frames through the four-stage perception and localization pipeline:
   - **Step 1 (Perspective Encoding)**: Extracts visual patch tokens and discretizes features via FSQ into a sparse set of `{code, pixel_uv, confidence}` observations.
   - **Step 2 (Local Index Lookup)**: Queries the active S2 spatial shard cache for 3D landmark candidate coordinates and metric constellation bearings.
   - **Step 3 (Where-Am-I Estimation)**: Evaluates geometric hypotheses (dPnP initialization) and applies the Perceiver set transformer to update particle log-weights and \( \mathfrak{se}(3) \) corrections.
   - **Step 4 (Hunter Policy Evaluation)**: Evaluates the mode-attention transformer policy over posterior entropy and rim codes to emit active gaze and steering adjustments.

### 2.1 Benchmark Measurement Protocols
To evaluate candidate embedded accelerators and software engines (e.g., TensorRT, ONNX Runtime, CUDA kernels), test runs must record the following diagnostic benchmarks without relying on synthetic estimates:

```
================================================================================
ONBOARD RUNTIME PROFILING PROTOCOL
Target Hardware: [Observed Hardware Platform / SOM Model]
Power Profile:   [Observed Power Mode / Wattage Ceiling]
Inference Engine:[Observed Engine / Precision: FP16, INT8, etc.]
================================================================================
1. EXECUTION LATENCY BREAKDOWN (per frame)
   - Perspective Encoder Backbone:          [Observed Value] ms
   - Cross-View Adapter & FSQ Head:         [Observed Value] ms
   - S2 Inverted Index Query:               [Observed Value] ms
   - Differentiable PnP Initializer:        [Observed Value] ms
   - Perceiver Where-Am-I Head:             [Observed Value] ms
   - Hunter Policy Transformer:             [Observed Value] ms
   - Total Visual-to-Pose Latency:          [Observed Value] ms

2. MEMORY AND STORAGE PROFILING
   - Neural Model Memory Working Set:       [Observed Value] MB
   - Dynamic Tensor Scratchpad:             [Observed Value] MB
   - Active S2 Shard Working Set (RAM):     [Observed Value] MB
   - Shard Paging I/O Latency (NVMe -> RAM):[Observed Value] ms

3. ELECTRICAL AND THERMAL PROFILE
   - Steady-State Compute Power Draw:       [Observed Value] W
   - Peak Transient Power Draw:             [Observed Value] W
   - Steady-State Core Temperature:         [Observed Value] deg C
================================================================================
```

---

## 3. Storage and Spatial Shard Management

v1 has **no global map**. The corridor inverted index is partitioned offline into S2 cells (Level-12 spatial shards). The payload loads **corridor shards only**:
- **Working Set**: Active corridor S2 shards in RAM for inverted index lookups.
- **Paging inside the corridor**: As the vehicle approaches a loaded-shard boundary, background I/O may stream neighboring **corridor** shards from onboard NVMe into RAM. Do not page a continental catalog.
- **No global map onboard**: The aircraft does not carry, download, or query an index outside the assigned corridor. Metric constellations resolve code ambiguity inside that corridor.

One Jetson-class payload. Do not size storage for a CONUS working set.

---

## 4. Size, Weight, Power, and Interface Requirements

Embedded flight payloads must operate within the physical constraints of tactical UAV airframes:
- **Compute Mass**: Sized to meet platform payload capacity, including carrier board and passive/active thermal management.
- **Power Envelope**: Sized to match available auxiliary DC power rails from flight battery distribution systems without degrading mission endurance.
- **Thermal Dissipation**: Capable of continuous operation within sealed or airflow-assisted avionics bays across target operating temperature ranges.
- **Sensor and Autopilot Interfacing**: Direct hardware synchronization with global-shutter cameras (via MIPI CSI-2 or GMSL2) and flight controllers (via MAVLink serial, CAN, or Ethernet).
