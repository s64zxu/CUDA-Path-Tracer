

# CUDA Path Tracer

# 1. Overview

High-performance C++/CUDA Path Tracer featuring Wavefront architecture, GPU-parallel LBVH, MIS, and bindless texture mapping.

![Sponza](img/Sponza.png)

# 2. Features

- **Wavefront Architecture:**  Decouples the rendering pipeline into independent Logic, Shading, and RayCast stages, utilizing multi-level task queues for efficient management.
- **High-Performance LBVH:** Features fully parallel GPU-based construction using Morton Codes. 
- **Material System:** Supports materials including Lambertian diffuse, Microfacet PBR , and Dielectrics with refraction and reflection.
- **Multiple Importance Sampling :** Combines BSDF sampling and Next Event Estimation (NEE) to reduce variance.
- **Bindless Texture Mapping:** Implements a handle-based system to efficiently map textures  without binding limitations.
- **HDR Environment Sampling:** Implements importance sampling for HDR skyboxes using the Alias Method.

# 3. Analysis

## Megakernel vs. Wavefront

This section analyzes the performance trade-offs between **Megakernel** and **Wavefront** architectures. The key finding is that while Wavefront successfully mitigates thread divergence, the overhead of global memory traffic becomes the bottleneck in moderate-complexity scenes.

### Test Environment

| Scene                                                        | Info                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="img/wavefront_test_scene.png" alt="wavefront_test_scene" style="zoom:50%;" /> | Scene：Fireplace Room<br/>Total Triangles : 143173<br/> MicrofacetPBR: 24421<br/>Diffuse: 107572 <br/>Specular reflection: 10<br/>Specular refraction: 11170<br/>Hardware: NVIDIA RTX 3060 Laptop |

---

### Phase I: Lightweight Shading 

In this scenario, the shading kernels are relatively lightweight. The Wavefront architecture trades memory bandwidth for execution coherence.

#### **Performance Summary:**

| Metric                     | Megakernel (Baseline) | Wavefront           | Difference  |
| :------------------------- | :-------------------- | :------------------ | :---------- |
| **Throughput**             | **34.72 Mpaths/s**    | 31.72 Mpaths/s      | 🔻 -8.64%    |
| Avg. Theoretical Occupancy | 33.33%                | **74.38%**          | 🟢 +123.16%  |
| Avg. Active Threads / Warp | 6.81                  | **18.86**           | 🟢 +176.95%  |
| Global Memory Access       | 63.39 Gbytes/s        | **857.71 Gbytes/s** | 🔴 +1253.07% |

**Conclusion:** For lightweight shaders, the 12x increase in global memory traffic outweighs the benefits of improved occupancy. The GPU is effectively "waiting for data" rather than computing.

<details>
<summary><strong>🔍 Click to expand: Deep Dive into Bottlenecks</strong></summary>
#### 1. Compute vs. Memory Analysis

For the Wavefront Ray Cast kernel, SM Busy is only 27.60% with an IPC of 1.10. In sharp contrast, DRAM Throughput reached 50.61%. The GPU is effectively "waiting for data" rather than **"**computing**."**

Compute Workload of Ray Cast Kernel:

<img src="img/Compute Workload of Ray Cast Kernel.png" alt="image-20251226083830643" style="zoom: 80%;" />

Memory Workload of Ray Cast Kernel:

<img src="img/Memory Workload of Ray Cast Kernel.png" alt="image-20251226083830643" style="zoom: 80%;" />

The intersection kernel writes a staggering 6.87 GB from L2 to Device Memory (Reading only 1.93 GB). This 6.87 GB of traffic is solely for maintaining global ray state—an overhead completely absent in the Megakernel, which passes state efficiently via registers.

<img src="img/memory chart of wavefront.png" alt="image-20251226082936565" style="zoom: 60%;" />

#### 2. Register Pressure and Occupancy

The **Megakernel** spilled **1.24 billion** instructions to Local Memory, generating massive request overhead (947M/540M). In contrast, the **Wavefront** architecture spilled only **356.37 million** instructions.

<img src="img/memory chart of megakernel.png" alt="image-20251226083830643" style="zoom: 80%;" />

Megakernel Occupancy: **30.29%** (Stifled by register pressure).

Wavefront Occupancy: **~74.38%** (Theoretical Occupancy: 100% in Logic/RayGen, ~50% in Intersection/Shading).

<img src="img/occupany of megakernel.png" alt="occupany of megakernel" style="zoom:60%;" />

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

To analyze the architectural behavior under heavy computational loads (simulating complex material shaders), a synthetic loop computation ($\sin^2 \theta + \cos^2 \theta$) was injected into the PBR shading kernel. And the result shows Wavefront scales better with shader complexity. 

<img src="img/stress_test_mega_vs_wavefront.png" alt="stress_test_mega_vs_wavefront" style="zoom:67%;" />

## Materials

| <img src="img\bunny_reflection.png" alt="bunny_reflection" style="zoom:40%;" /> | <img src="img\bunny_refraction.png" alt="bunny_refraction" style="zoom:40%;" /> | <img src="img\bunny_pbr.png" alt="bunny_pbr" style="zoom:40%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Specular Reflection**                                      | **Specular Refraction**                                      | **Microfacet PBR**                                           |
