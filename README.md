

# CUDA Path Tracer

# 1. Overview

High-performance C++/CUDA Path Tracer featuring Wavefront architecture, GPU-parallel LBVH, MIS, and bindless texture mapping.

![Sponza](img/Sponza.png)

| <img src="img\bunny_reflection.png" alt="bunny_reflection" style="zoom:40%;" /> | <img src="img\bunny_refraction.png" alt="bunny_refraction" style="zoom:40%;" /> | <img src="img\bunny_pbr.png" alt="bunny_pbr" style="zoom:40%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Specular Reflection**                                      | **Specular Refraction**                                      | **Microfacet PBR**                                           |

# 2. Features

- **Wavefront Architecture:**  Decouples the rendering pipeline into independent Logic, Shading, and RayCast stages, utilizing multi-level task queues for efficient management.
- **High-Performance LBVH:** Features fully parallel GPU-based construction using Morton Codes. 
- **Material System:** Supports materials including Lambertian diffuse, Microfacet PBR , and Dielectrics with refraction and reflection.
- **Multiple Importance Sampling :** Combines BSDF sampling and Next Event Estimation (NEE) to reduce variance.
- **Bindless Texture Mapping:** Implements a handle-based system to efficiently map textures  without binding limitations.
- **HDR Environment Sampling:** Implements importance sampling for HDR skyboxes using the Alias Method.
- **OptiX Integration:** Integrates NVIDIA OptiX to leverage RT Cores for accelerating BVH traversal and ray-triangle intersection.

# 3. Performance Analysis

## 3.1 Megakernel vs. Wavefront

This section analyzes the performance trade-offs between **Megakernel** and **Wavefront** architectures. The key finding is that while Wavefront successfully mitigates thread divergence, the overhead of global memory traffic becomes the bottleneck in moderate-complexity scenes.

### Test Environment

| Scene                                                        | Info                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="img/wavefront_test_scene.png" alt="wavefront_test_scene" style="zoom:50%;" /> | Scene：Fireplace Room<br/>Total Triangles : 143173<br/> MicrofacetPBR: 24421<br/>Diffuse: 107572 <br/>Specular reflection: 10<br/>Specular refraction: 11170<br/>Hardware: NVIDIA RTX 3060 Laptop |

---

### Phase I: Lightweight Shading 

In this scenario, the shading kernels are relatively lightweight. The Wavefront architecture trades memory bandwidth for execution coherence.

#### **Performance Summary:**

| Metric                     | Megakernel (Baseline) | Wavefront           | Impact        |
| :------------------------- | :-------------------- | :------------------ | :------------ |
| Throughput                 | 34.72 Mpaths/s        | 31.72 Mpaths/s      | -8.64%        |
| Avg. Theoretical Occupancy | 33.33%                | 74.38%              | +123.16%      |
| Avg. Active Threads / Warp | 6.81                  | 18.86               | +176.95%      |
| Global Memory Access       | 63.39 Gbytes/s        | **857.71 Gbytes/s** | **+1253.07%** |

**Conclusion:** For lightweight shaders, the **12x** increase in global memory traffic outweighs the benefits of improved occupancy. The GPU is effectively "waiting for data" rather than computing.

<details>
<summary><strong> Click to expand: Deep Dive into Bottlenecks</strong></summary>

#### 1. Compute vs. Memory Analysis

We focus on the **Ray Intersection Kernel** of Wavefront as it dominates the pipeline, consuming **63%** of the total frame time. The profiling results confirm the kernel is **Memory Bound**:  There is a significant gap between **SM Busy** (27.6%) and **Mem Busy** (51.80%). 

| Metric        | Value  | Metric         | Value       |
| :------------ | :----- | :------------- | ----------- |
| **SM Busy**   | 27.60% | **Mem Busy**   | 51.80%      |
| Max Bandwidth | 56.69% | Mem Throughput | 164.75 GB/s |

Root Cause of Memory Bound:

The ray intersection kernel writes a staggering **6.87 GB** from L2 to Device Memory (Reading only 1.93 GB). This 6.87 GB of traffic is solely for maintaining global ray state—an overhead completely absent in the Megakernel, which passes state efficiently via registers.

<img src="img/memory chart of wavefront.png" alt="image-20251226082936565" style="zoom: 80%;" />

#### 2. Register Pressure and Occupancy

A critical bottleneck in the **Megakernel** approach is excessive register usage, which forces data to "spill" into slow Local Memory. The **Wavefront** architecture mitigates this by splitting kernels, significantly reducing register pressure.

| Metric                    | Megakernel (Baseline) | Wavefront (Optimized) | Impact       |
| :------------------------ | :-------------------- | :-------------------- | :----------- |
| **Instruction Spills**    | 1240.17 Million       | 356.37 Million        | **-71.26%**  |
| **Local Memory Request**  | 1487.78 Million       | 234.82 Million        | **-84.21%**  |
| **Theoretical Occupancy** | 33.33%                | ~74.38%               | **+123.16%** |

Register Pressure:

The Megakernel spilled **1.24 billion instructions** to Local Memory.This generated massive request overhead (**947M Writes / 540M Reads**), saturating the L1/Texture cache bandwidth. By decomposing the complex logic, the Wavefront architecture reduced spilled instructions to just **356.37 million**.

Occupancy:

Theoretical Occupancy of Wavefront jumped from **33.33%** to **~74.38%** (reaching 100% in Logic/RayGen stages). This allows the GPU to hide memory latency more effectively by keeping more warps active.

#### 3. Threads Divergency

Instructions are executed in warps. The Megakernel achieves only **6.81 active threads per warp** (reduced to 6.69 by predication) due to severe branching.

In Wavefront, this metric improves significantly in compute-heavy stages:

- 🟢 **Shading**: `29.60` (Near optimal)
- 🟢 **Ray Generation**: `27.80`
- 🔴 **Logic**: `9.81` (Natural branch divergence)
- 🔴 **Intersection**: `8.23` (Stochastic nature of BVH traversal)

</details>

---

### Phase II: Heavy Shading Stress Test 

To analyze the architectural behavior under heavy computational loads (simulating complex material shaders), a synthetic loop computation ($\sin^2 \theta + \cos^2 \theta$) was injected into the PBR shading kernel. And the result shows Wavefront scales better with material complexity. 

<img src="img/stress_test_mega_vs_wavefront.png" alt="stress_test_mega_vs_wavefront" style="zoom:67%;" />

## 3.2 Hardware Accelerated Ray Tracing 

This section evaluates the performance impact of integrating **NVIDIA OptiX 9.1**, which leverages hardware-accelerated intersection and BVH traversal via **RT Cores**.

Experiment was conducted on **RTX 3060 laptop ** using the Wavefront architecture and  **Sponza scene** **(262,279 triangles).**

<img src="img/optix vs. software plot.png" alt="stress_test_mega_vs_wavefront" style="zoom:67%;" />

The integration yields a significant **8.19x speedup** in ray traversal throughput. In general, this performance leap is driven by two factors: First, the OptiX constructs highly optimized and efficient BVH structures. Second, the RT Cores leverage dedicated hardware circuits to execute ray-triangle intersection tests significantly faster than software Moller-Trumbore algorithm.



🚧 **Work In Progress**.... 🚧

