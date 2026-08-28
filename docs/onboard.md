# Onboard Embedded Runtime and SWaP-C Specifications

Date: 2026-08-28  
Status: Embedded Hardware Specification  
Repository: [abiome-org/MonARC](https://github.com/abiome-org/MonARC)  

---

## 1. Onboard Payload Compute Architecture

MonARC is designed to run entirely on embedded edge compute payloads without external RF communication or cloud connectivity during flight.

```
+===================================================================================================+
|                                  ONBOARD HARDWARE ARCHITECTURE                                    |
|                                                                                                   |
|  [Global Shutter Camera]      [Tactical IMU]         [Altimeter / Barometer]                      |
|       | 10-30 FPS                  | 200 Hz                     | 50 Hz                           |
|       | MIPI CSI-2 / GMSL2         | SPI / UART                 | I2C                             |
|       v                            v                            v                                 |
|  +---------------------------------------------------------------------------------------------+  |
|  | EMBEDDED SYSTEM-ON-MODULE (SOM) (e.g., NVIDIA Jetson Orin NX 16GB / AGX Orin 32GB)          |  |
|  |                                                                                             |  |
|  | [GPU / TensorRT Execution Stream]                [CPU Real-Time Filter & IO Thread]         |  |
|  |   1. Perspective Encoder (~18 ms)                  1. IMU Dead-Reckoning (200 Hz)          |  |
|  |   2. S2 Inverted Index Lookup (~3 ms)              2. Local S2 Map Shard Cache (NVMe)      |  |
|  |   3. Perceiver Where-Am-I Head (~8 ms)             3. Particle Propagation & Resampling    |  |
|  |   4. Hunter Transformer Policy (~2 ms)             4. Autopilot MAVLink Stream (50 Hz)     |  |
|  |                                                                                             |  |
|  | [Unified LPDDR5 Memory Pool (16 GB - 32 GB)]                                                |  |
|  |   - Neural Network Weights: < 450 MB                                                        |  |
|  |   - Dynamic Execution Tensors: < 600 MB                                                     |  |
|  |   - Local S2 Metric Index Shards (50km x 50km): 1.2 GB                                      |  |
|  +---------------------------------------------------------------------------------------------+  |
|                                            |                                                      |
|                                            v MAVLink / CAN / Serial                               |
|                               [Flight Controller (e.g. PX4 / ArduPilot)]                          |
+===================================================================================================+
```

---

## 2. Real-Time Latency Budgets and Execution Pipeline

To maintain stable 6-DoF state estimation during aggressive maneuvers, total visual-to-pose latency must not exceed 40 ms per frame.

| Subsystem Component | Hardware Engine | Precision | Compute Budget | Wall-Clock Latency |
| :--- | :--- | :--- | :--- | :--- |
| **Perspective Vision Backbone** | GPU / TensorRT | FP16 / INT8 | 150 GFLOPs | 14.0 – 18.0 ms |
| **Cross-View Adapter & FSQ Head**| GPU / TensorRT | FP16 | 15 GFLOPs | 2.0 – 3.0 ms |
| **Local S2 Metric Index Query** | CPU / SIMD | Integer LUT | Memory Bound | 1.5 – 3.0 ms |
| **Differentiable PnP Initializer**| GPU / CUDA Kernel | FP32 | 5 GFLOPs | 1.0 – 2.0 ms |
| **Perceiver Set Transformer** | GPU / TensorRT | FP16 | 40 GFLOPs | 6.0 – 8.0 ms |
| **Hunter Policy Transformer** | GPU / TensorRT | FP16 | 2 GFLOPs | 1.0 – 2.0 ms |
| **SE(3) State Distribution Export**| CPU (ARM Cortex-A78AE)| FP64 | Negligible | < 0.5 ms |
| **Total Visual Pipeline Latency** | Full Loop | Mixed | ~212 GFLOPs | **25.5 – 36.5 ms** |

At an execution latency of ~31 ms, the visual estimation loop comfortably operates at a 10 Hz to 20 Hz update frequency, interleaved with 200 Hz IMU inertial propagation.

---

## 3. Memory Footprint and Storage Model

The offline landmark field is massive at continental scale (terabytes), but the onboard drone payload loads only the local operational corridor.

```
                      ONBOARD MEMORY & STORAGE ALLOCATION
                      
  Total Jetson Orin NX Unified Memory: 16,384 MB
  +--------------------------------------------------------------------+
  | OS Kernel, Drivers & CUDA Runtime           : 2,048 MB             |
  +--------------------------------------------------------------------+
  | MonARC Neural Network Weights (TensorRT)    :   420 MB             |
  |   - DINOv2-ViT-S Backbone (FP16)            :   170 MB             |
  |   - Perspective Head & FSQ (FP16)           :    45 MB             |
  |   - Perceiver Set Transformer (FP16)        :   180 MB             |
  |   - Hunter Policy Transformer (FP16)        :    25 MB             |
  +--------------------------------------------------------------------+
  | Dynamic CUDA Tensor Buffers & Scratchpad     :   600 MB             |
  +--------------------------------------------------------------------+
  | Active S2 Shard Inverted Metric Index        : 1,200 MB             |
  |   (Covers 2,500 km^2 corridor at 0.5m GSD)                         |
  +--------------------------------------------------------------------+
  | Free Headroom / System Reserve              : 12,116 MB            |
  +--------------------------------------------------------------------+
```

### 3.1 S2 Spatial Shard Format
- **Spatial Sharding**: The global index is partitioned into S2 Level-12 cells (approx. \( 3 \times 3 \text{ km} \) per cell).
- **Storage Primitive**: Flat binary LMDB or memory-mapped flatbuffers stored on onboard NVMe SSD (M.2 PCIe Gen4).
- **Dynamic Shard Paging**: As the vehicle approaches the boundary of active S2 cells, background threads pre-fetch adjacent cells from NVMe into RAM within < 10 ms without blocking the real-time inference loop.

---

## 4. Size, Weight, Power, and Cost (SWaP-C) Envelope

| Specification Parameter | Minimum Target | Nominal Target | Maximum Ceiling |
| :--- | :--- | :--- | :--- |
| **Compute Board Weight** | 70 g (Bare Module) | 250 g (With Carrier & Heatsink) | 400 g |
| **Power Consumption** | 15 W (Jetson 15W Mode) | 25 W (MAXN Mode) | 40 W (Jetson AGX Orin) |
| **Supply Voltage** | 9 V DC | 12 V – 19 V DC | 24 V DC |
| **Operating Temperature** | -20 °C | 25 °C | +65 °C |
| **Storage Capacity** | 128 GB NVMe SSD | 512 GB NVMe SSD | 1 TB NVMe SSD |
| **Sensor Ingestion Bus** | MIPI CSI-2 (2-lane) | GMSL2 (Coaxial) | USB3 Vision / GigE |
