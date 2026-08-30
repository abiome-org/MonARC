# Annotated Bibliography and Foundational Literature

Date: 2026-08-30  
Status: Reference Literature Index  
Repository: [abiome-org/MonARC](https://github.com/abiome-org/MonARC)  

---

## 1. Scene Coordinate Regression and Direct Visual Localization

1. **ACE: Accelerated Coordinate Encoding**
   - *Authors*: Eric Brachmann, Tommaso Cavallari, Victor Adrian Prisacariu (CVPR 2023)
   - *Repository*: [nianticlabs/ace](https://github.com/nianticlabs/ace)
   - *Summary*: Establishes rapid mapping and scene coordinate regression by training shallow scene-specific MLPs on top of pretrained feature backbones. ACE achieves fast convergence but remains constrained to small-scale individual scenes.

2. **GLACE: Global Local Accelerated Coordinate Encoding**
   - *Authors*: Fangjinhua Wang, Xudong Jiang, Silvano Galliani, Christoph Vogel, Marc Pollefeys (CVPR 2024)
   - *Repository*: [cvg/glace](https://github.com/cvg/glace) | [Paper PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_GLACE_Global_Local_Accelerated_Coordinate_Encoding_CVPR_2024_paper.pdf)
   - *Summary*: Identifies the fundamental scene-coordinate regression dilemma: the tension between viewpoint/illumination invariance and spatial discrimination across repetitive large-scale environments. GLACE introduces co-visibility grouping and feature diffusion to scale coordinate encoding across large scenes. MonARC builds upon GLACE's theoretical formulation by replacing direct coordinate regression with discrete FSQ visual codebooks coupled to metric constellation indexes.

3. **ACE-G: Improving Generalization of Scene Coordinate Regression Through Query Pre-Training**
   - *Authors*: David Bruns et al. (ICCV 2025)
   - *Paper / Project*: [nianticspatial.github.io/ace-g](https://nianticspatial.github.io/ace-g/) | [Paper PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Bruns_ACE-G_Improving_Generalization_of_Scene_Coordinate_Regression_Through_Query_Pre-Training_ICCV_2025_paper.pdf)
   - *Summary*: Disentangles scene-agnostic coordinate regressors from scene-specific map codes via query pre-training, validating that foundation vision features (DINO) drastically improve cross-view generalization.

4. **OrthoLoC: UAV 6-DoF Localization and Calibration Using Orthographic Geodata**
   - *Authors*: DeepScenario Research Team (arXiv:2509.18350, 2025)
   - *Project / Code*: [deepscenario.github.io/OrthoLoC](https://deepscenario.github.io/OrthoLoC/) | [arXiv:2509.18350](https://arxiv.org/abs/2509.18350)
   - *Summary*: Introduces a 16.4k-image paired UAV-geodata benchmark evaluated directly against digital orthophotos (DOP) and digital surface models (DSM). Demonstrates that 2.5D governmental geodata provides sufficient geometric constraint for 6-DoF UAV pose estimation without pre-existing 3D meshes.

5. **NGC-GeoLoc: Neural GeoCoordinate Regression for GPS-Denied UAV Geo-Localization**
   - *Authors*: Quan Chen et al. (IEEE Robotics and Automation Letters, 2026)
   - *Repository*: [djcrobo/NGC-Geoloc](https://github.com/djcrobo/NGC-Geoloc) | [DOI:10.1109/LRA.2026.3655216](https://doi.org/10.1109/lra.2026.3655216)
   - *Summary*: Regresses continuous map-space coordinates directly from UAV pixels to solve scale and rotation variations across satellite maps, refining candidate poses via dense homographies.

6. **PiLoT: Neural Pixel-to-3D Registration for UAV-based Ego and Target Geo-localization**
   - *Authors*: Autonomous Systems Lab (arXiv:2603.20778, 2026)
   - *Paper*: [arXiv:2603.20778](https://arxiv.org/abs/2603.20778)
   - *Summary*: Proposes a dual-thread neural engine that decouples on-the-fly map rendering from real-time feature registration, achieving 25+ FPS on NVIDIA Jetson Orin.

---

## 2. Visual Place Recognition (VPR) and Foundation Feature Encoders

7. **AnyLoc: Towards Universal Visual Place Recognition**
   - *Authors*: Nikhil Keetha et al. (IEEE Transactions on Robotics / arXiv:2308.00688, 2023)
   - *Project*: [anyloc.github.io](https://anyloc.github.io/) | [arXiv:2308.00688](https://arxiv.org/abs/2308.00688)
   - *Summary*: Proves that general-purpose self-supervised vision representations (DINOv2) combined with unsupervised aggregation (GeM/VLAD) enable universal place recognition across structured and unstructured domains without fine-tuning.

8. **MegaLoc: Large-Scale Foundation Descriptors for Visual Geo-Localization**
   - *Summary*: Compact foundation-model visual descriptor family for global tile retrieval. Serves as MonARC's initial coarse retrieval mechanism to seed lost-in-space candidate regions before metric constellation matching.

9. **DINOv2: Learning Robust Visual Features without Supervision**
   - *Authors*: Maxime Oquab et al. (Meta AI, arXiv:2304.07193, 2023)
   - *Repository*: [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
   - *Summary*: Provides the frozen 3-channel RGB vision backbone powering MonARC's spatial feature fields and perspective encoder tokens.

10. **Finite Scalar Quantization (FSQ): VQ-VAE Made Simple**
    - *Authors*: Fabian Mentzer, David Minnen, Eirikur Agustsson, Michael Tschannen (arXiv:2309.15505, 2023)
    - *Paper*: [arXiv:2309.15505](https://arxiv.org/abs/2309.15505)
    - *Summary*: Replaces learned codebooks and vector lookup with fixed scalar quantization levels. Eliminates codebook collapse, dead clusters, and complex commitment loss schedules.

11. **VGGT: Visual Geometry Grounded Transformers**
    - *Summary*: Sequence transformer leveraging multi-view geometric priors to output dense 3D visual geometry tokens. MonARC utilizes VGGT-style geometry tokens for perspective video streams.

---

## 3. Active Perception and Information-Theoretic Navigation

12. **GeoExplorer: Active Geo-Localization with Curiosity-Driven Exploration**
    - *Authors*: Li Mi, Manon Béchaz, Zeming Chen, Antoine Bosselut, Devis Tuia (ICCV 2025)
    - *Paper / Project*: [ICCV 2025 Paper PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Mi_GeoExplorer_Active_Geo-localization_with_Curiosity-Driven_Exploration_ICCV_2025_paper.pdf) | [DOI:10.1109/ICCV51701.2025.00578](https://doi.org/10.1109/iccv51701.2025.00578)
    - *Summary*: Formulates active visual geo-localization as an exploration problem where an agent navigates to reduce goal uncertainty. Introduces intrinsic curiosity rewards for active target discovery.

13. **ActLoc: Attention-Guided Active Camera Localization**
    - *Authors*: Multi-Agent Robotics Group (arXiv:2508.20981, 2025)
    - *Paper*: [arXiv:2508.20981](https://arxiv.org/abs/2508.20981)
    - *Summary*: Optimizes active camera orientation using attention saliency maps to steer perception toward high-confidence localization cues.

---

## 4. Aerial Simulators and Neural Rendering

14. **SOUS VIDE: Cooking Visual Drone Navigation Policies in a Gaussian Splatting Vacuum**
    - *Authors*: Stanford MSL (arXiv:2412.16346, 2024)
    - *Project / Paper*: [stanfordmsl.github.io/SousVide](https://stanfordmsl.github.io/SousVide/) | [arXiv:2412.16346](https://arxiv.org/abs/2412.16346)
    - *Summary*: Couples lightweight drone dynamics with photorealistic 3D Gaussian Splatting (FiGS simulator) generating 130 FPS visual rollouts for zero-shot sim-to-real flight policy distillation.

15. **Splat-Nav: Safe Real-Time Robot Navigation in Gaussian Splatting Maps**
    - *Authors*: Timothy Chen, Ola Shorinwa et al. (Stanford/UCSD/Temple, arXiv:2403.02751, 2024)
    - *Repository*: [chengine/splatnav](https://github.com/chengine/splatnav) | [Project](https://chengine.github.io/splatnav/)
    - *Summary*: Integrates real-time pose estimation (Splat-Loc) with safe trajectory corridor planning (Splat-Plan) directly over 3D Gaussian Splatting scene representations.

16. **GISNav: Aerial Map-Based Visual Navigation Bridge**
    - *Author*: Harri Makelin (2023–2026)
    - *Repository*: [hmakelin/gisnav](https://github.com/hmakelin/gisnav)
    - *Summary*: ROS 2 package bridging airborne camera video feeds to local GIS map servers for GPS-free simulation in PX4 and ArduPilot ecosystems.

---

## 5. Public Geodata Repositories and UAV Benchmarks

17. **University-1652 Benchmark**
    - *Authors*: Zhedong Zheng, Yunchao Wei, Yi Yang (ACM MM 2020)
    - *Repository*: [layumi/University1652-Baseline](https://github.com/layumi/University1652-Baseline)
    - *Summary*: Multi-view university campus dataset containing paired UAV, satellite, and street-level views of 1,652 buildings. Used exclusively as a cross-view alignment benchmark.

18. **DenseUAV Benchmark**
    - *Authors*: ACM Multimedia 2023 Benchmark Group
    - *Repository*: [Zgt-d/DenseUAV](https://github.com/Zgt-d/DenseUAV)
    - *Summary*: Dense multi-altitude UAV-satellite dataset capturing multi-scale aerial perspectives at 80m, 90m, and 100m AGL.

19. **SUES-200 Benchmark**
    - *Authors*: Zhu et al. (2022)
    - *Repository*: [Reagan-Zhu/SUES-200-Benchmark](https://github.com/Reagan-Zhu/SUES-200-Benchmark)
    - *Summary*: Cross-view UAV visual localization dataset covering 200 distinct geographic scenes captured at altitudes of 150m, 200m, 250m, and 300m AGL.

20. **USGS National Agriculture Imagery Program (NAIP)**
    - *Access*: [USGS EROS NAIP Portal](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-aerial-photography-national-agriculture-imagery-program-naip) / AWS `s3://naip-visualization/`
    - *Summary*: Authoritative orthophotography across CONUS. CONUS coverage is **data availability**. v1 ingest is **Colorado** from visualization JPEG COGs (one vintage, ~0.6 m); not `naip-source`, not all years, not one county as the product.

21. **USGS 3D Elevation Program (3DEP)**
    - *Access*: [USGS 3DEP Program](https://www.usgs.gov/3d-elevation-program)
    - *Summary*: Authoritative LiDAR-derived DSM/DEM products. v1 uses the cheapest already-COG product that supplies metric \( z \) in the **Colorado** clip (prefer 1/9 arc-second ~3 m). CONUS 1 m point clouds are not a v1 source.

22. **Overture Maps Foundation**
    - *Access*: [Overture Maps](https://overturemaps.org/)
    - *Summary*: Open global vector map datasets providing building footprints, road network centerlines, and administrative divisions formatted in Cloud-Optimized GeoParquet.
