#include "bvh.h"
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include "cuda_utilities.h"
#include <vector>
#include <stack>
#include <set>
#include <cstdio>
#include "intersections.h"

__global__ void ComputeAABB(LBVHData d_bvh_data, MeshData d_mesh_data)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= d_mesh_data.num_triangles) return;

    int4 indices_matid = d_mesh_data.indices_matid[tid];

    float4 v0 = d_mesh_data.pos[indices_matid.x];
    float4 v1 = d_mesh_data.pos[indices_matid.y];
    float4 v2 = d_mesh_data.pos[indices_matid.z];

    float4 min_v = Fmin4(Fmin4(v0, v1), v2);
    float4 max_v = Fmax4(Fmax4(v0, v1), v2);

    min_v.w = 1.0f; max_v.w = 1.0f;

    d_bvh_data.aabb_min[tid] = min_v;
    d_bvh_data.aabb_max[tid] = max_v;

    d_bvh_data.centroid[tid] = (max_v + min_v) * 0.5f;

    d_bvh_data.primitive_indices[tid] = tid;
}

void ComputeWorldAABB(LBVHData& d_bvh_data, const MeshData& d_mesh_data)
{
    int num_triangles = d_mesh_data.num_triangles;
    thrust::device_ptr<float4> min_ptr(d_bvh_data.aabb_min);
    thrust::device_ptr<float4> max_ptr(d_bvh_data.aabb_max);

    float4 init_min = make_float4(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX);
    float4 init_max = make_float4(-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);

    float4 cpu_world_min = thrust::reduce(min_ptr, min_ptr + num_triangles, init_min, Float4Min());
    float4 cpu_world_max = thrust::reduce(max_ptr, max_ptr + num_triangles, init_max, Float4Max());

    d_bvh_data.world_aabb_min = cpu_world_min;
    d_bvh_data.world_aabb_max = cpu_world_max;
}

__device__ unsigned int ExpandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ unsigned int Morton3D(float x, float y, float z)
{
    x = fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
    y = fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
    z = fminf(fmaxf(z * 1024.0f, 0.0f), 1023.0f);
    unsigned int xx = ExpandBits((unsigned int)x);
    unsigned int yy = ExpandBits((unsigned int)y);
    unsigned int zz = ExpandBits((unsigned int)z);
    return (xx << 2) + (yy << 1) + zz;
}


__global__ void ComputeMortonCode(LBVHData d_bvh_data, MeshData d_mesh_data)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= d_mesh_data.num_triangles) return;

    float3 centroid = MakeFloat3(d_bvh_data.centroid[tid]);

    float3 world_min = MakeFloat3(d_bvh_data.world_aabb_min);
    float3 world_max = MakeFloat3(d_bvh_data.world_aabb_max);

    float3 extent = world_max - world_min;

    // 鲁棒性检查
    if (extent.x < 1e-6f) extent.x = 1.0f;
    if (extent.y < 1e-6f) extent.y = 1.0f;
    if (extent.z < 1e-6f) extent.z = 1.0f;

    float3 scaled_centroid = (centroid - world_min) / extent;

    unsigned int morton_code_30 = Morton3D(scaled_centroid.x, scaled_centroid.y, scaled_centroid.z);

    d_bvh_data.morton_codes[tid] = ((unsigned long long)morton_code_30 << 32) | (unsigned long long)tid; // | 是位或运算
}

void SortMortonCode(LBVHData d_bvh_data, const MeshData& d_mesh_data)
{
    int num_triangles = d_mesh_data.num_triangles;
    thrust::device_ptr<unsigned long long> morton_code_ptr(d_bvh_data.morton_codes);
    thrust::device_ptr<int> primitive_indices_ptr(d_bvh_data.primitive_indices);
    thrust::sort_by_key(morton_code_ptr, morton_code_ptr + num_triangles, primitive_indices_ptr);
}

__global__ void SetupLeafNodes(LBVHData d_bvh_data, MeshData d_mesh_data)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int num_objects = d_mesh_data.num_triangles;
    if (i >= num_objects) return;

    // 莫顿码排序后的AABB的索引，因为排序的时候没有对AABB进行排序
    // 所以直接搬移AABB会使用未排序的索引
    int sorted_prim_idx = d_bvh_data.primitive_indices[i];

    // 叶子节点存放在 [N, 2N-1] 范围内
    int leaf_idx = num_objects + i;

    d_bvh_data.aabb_min[leaf_idx] = d_bvh_data.aabb_min[sorted_prim_idx];
    d_bvh_data.aabb_max[leaf_idx] = d_bvh_data.aabb_max[sorted_prim_idx];
    d_bvh_data.centroid[leaf_idx] = d_bvh_data.centroid[sorted_prim_idx];

    d_bvh_data.primitive_indices[leaf_idx] = sorted_prim_idx;
}



__device__ __forceinline__ int ComputeLCP(LBVHData d_bvh_data, MeshData d_mesh_data, int i, int j)
{
    if (j < 0 || j >= d_mesh_data.num_triangles) return -1;
    unsigned long long key_a = d_bvh_data.morton_codes[i];
    unsigned long long key_b = d_bvh_data.morton_codes[j];
    return __clzll(key_a ^ key_b); // ^是位异或运算符， __clzll指令给出二进制数前面连续的0的数量
}


__global__ void GenerateHierarchy(LBVHData d_bvh_data, MeshData d_mesh_data)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d_mesh_data.num_triangles - 1) return;

    // computer direction
    int dir = ComputeLCP(d_bvh_data, d_mesh_data, i, i + 1)
        - ComputeLCP(d_bvh_data, d_mesh_data, i, i - 1) >= 0 ? 1 : -1;

    // compute upper bound for the length of the range
    int l_max = 2;
    int delta_min = ComputeLCP(d_bvh_data, d_mesh_data, i, i - dir);
    while (ComputeLCP(d_bvh_data, d_mesh_data, i, i + l_max * dir) > delta_min)
    {
        l_max *= 2; // l_max 是第一个不满足条件的2的幂上的位置
    }

    int l = 0; // 当前节点包含叶节点的范围长度
    for (int t = l_max / 2; t >= 1; t /= 2)
    {
        if (ComputeLCP(d_bvh_data, d_mesh_data, i, i + (l + t) * dir) > delta_min)
        {
            l += t; // l 停在最后满足条件的位置上
        }
    }

    int j = i + l * dir; // 计算范围的另一端的索引
    // fint the split position
    int delta_node = ComputeLCP(d_bvh_data, d_mesh_data, i, j);
    int s = 0;
    for (int t = (l + 1) >> 1; t >= 1; t = (t == 1) ? 0 : ((t + 1) >> 1))
    {
        if (ComputeLCP(d_bvh_data, d_mesh_data, i, i + (s + t) * dir) > delta_node)
        {
            s += t;
        }
    }

    int gamma = i + s * dir + min(dir, 0);// min(dir, 0)保证gamma始终指向左子节点
    //int gamma = i + s * dir + (dir > 0 ? 0 : -1);
    int leaf_offset = d_mesh_data.num_triangles;

    // assign child and parent pointers
    int left_idx, right_idx;
    // 处理左子节点
    if (min(i, j) == gamma) { // 叶子节点
        left_idx = gamma + leaf_offset;
        // 位取反表示叶子节点(~x = -x-1，可以处理索引0为叶子节点的情况，直接取负则不行，-0 = +0)
        d_bvh_data.child_nodes[i].x = ~left_idx;
    }
    else {
        left_idx = gamma;
        d_bvh_data.child_nodes[i].x = left_idx;
    }
    // 处理右子节点
    if (max(i, j) == gamma + 1) {
        right_idx = gamma + 1 + leaf_offset;
        d_bvh_data.child_nodes[i].y = ~right_idx;
    }
    else {
        right_idx = gamma + 1;
        d_bvh_data.child_nodes[i].y = right_idx;
    }
    // 记录父节点指针 
    d_bvh_data.parent[left_idx] = i;
    d_bvh_data.parent[right_idx] = i;
}

// 供 Refit 使用的 UnionAABB
__device__ inline void UnionAABB(
    const float4& min_a, const float4& max_a,
    const float4& min_b, const float4& max_b,
    float4& res_min, float4& res_max)
{
    res_min = Fmin4(min_a, min_b);
    res_max = Fmax4(max_a, max_b);
}

__global__ void RefitAABB(LBVHData d_bvh_data, MeshData d_mesh_data, int* atomic_flags)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (d_mesh_data.num_triangles)) return; // 叶节点数量等于三角形数量
    int cur_node_idx = tid + d_mesh_data.num_triangles; // 叶节点索引
    while (true)
    {
        int parent_idx = d_bvh_data.parent[cur_node_idx];
        if (parent_idx == -1) break; // 到达根节点，结束
        __threadfence(); // 确保其他节点在更上一层进行合并时，当前节点的AABB已经合并完成
        int old_flag = atomicAdd(&atomic_flags[parent_idx], 1);
        if (old_flag == 0) return; // 第一个访问父节点，直接返回
        int left_child_idx = DecodeNode(d_bvh_data.child_nodes[parent_idx].x);
        int right_child_idx = DecodeNode(d_bvh_data.child_nodes[parent_idx].y);
        // 合并子节点的AABB
        float4 left_aabb_min = d_bvh_data.aabb_min[left_child_idx]; // abs用于处理叶子节点的负号
        float4 left_aabb_max = d_bvh_data.aabb_max[left_child_idx];
        float4 right_aabb_min = d_bvh_data.aabb_min[right_child_idx];
        float4 right_aabb_max = d_bvh_data.aabb_max[right_child_idx];
        float4 parent_aabb_min, parent_aabb_max;
        UnionAABB(left_aabb_min, left_aabb_max,
            right_aabb_min, right_aabb_max,
            parent_aabb_min, parent_aabb_max);
        d_bvh_data.aabb_min[parent_idx] = parent_aabb_min;
        d_bvh_data.aabb_max[parent_idx] = parent_aabb_max;
        cur_node_idx = parent_idx; // 向上继续处理父节点
    }
}

__global__ void BuildEscapeIdx(LBVHData d_bvh_data, MeshData d_mesh_data)
{
    int cur_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cur_idx >= (d_mesh_data.num_triangles - 1)) return; // 只处理内部节点
    int2 children = d_bvh_data.child_nodes[cur_idx];
    // Decode child indices because child_nodes may store encoded (negated) leaf indices
    int left_child = DecodeNode(children.x);
    int right_child = DecodeNode(children.y);

    // write escape for left child -> right sibling (decoded indices)
    d_bvh_data.escape_indices[left_child] = right_child;

    int escape = -1; // 默认跳出整个树
    // 根节点(0)的右孩子如果没有命中，直接结束(-1)
    if (cur_idx == 0) {
        escape = -1;
    }
    else {
        // 开始回溯：从当前节点 idx 向上找
        int p = d_bvh_data.parent[cur_idx];
        int node = cur_idx;
        while (p != -1) {
            // 读取父节点的两个孩子，看 node 是左还是右
            int2 p_children = d_bvh_data.child_nodes[p];
            int p_left = DecodeNode(p_children.x);
            int p_right = DecodeNode(p_children.y);

            // if current node is left child of p, then escape is p's right child
            if (p_left == node) {
                escape = p_right;
                break;
            }
            // otherwise climb up
            node = p;
            p = d_bvh_data.parent[node];
        }
    }

    // write escape for right child
    d_bvh_data.escape_indices[right_child] = escape;
}


int DecodeIndex(int idx) {
    return (idx < 0) ? ~idx : idx;
}

void TestHierarchyLogic(const LBVHData& d_bvh, int num_triangles) {
    if (num_triangles < 2) {
        printf("[TestHierarchy] Too few triangles to build a tree.\n");
        return;
    }

    int num_internal = num_triangles - 1;
    int total_nodes = 2 * num_triangles; // 实际上用到的是 [0..N-2] 和 [N..2N-1]

    // 1. 将数据下载到 CPU
    std::vector<int2> h_children(num_internal);
    std::vector<int> h_parent(total_nodes);

    // 拷贝 child_nodes (只有内部节点有)
    cudaMemcpy(h_children.data(), d_bvh.child_nodes, sizeof(int2) * num_internal, cudaMemcpyDeviceToHost);
    // 拷贝 parent (所有节点都有)
    cudaMemcpy(h_parent.data(), d_bvh.parent, sizeof(int) * total_nodes, cudaMemcpyDeviceToHost);

    printf("\n====== [TEST] Hierarchy Topology Check ======\n");

    int error_count = 0;

    // --- 检查 1: 根节点父指针 ---
    if (h_parent[0] != -1) {
        printf("[Error] Root node (0) parent is %d, expected -1.\n", h_parent[0]);
        error_count++;
    }

    // --- 检查 2: 父子双向一致性 ---
    // 遍历所有内部节点，检查它们指向的孩子是否认它们做父亲
    for (int i = 0; i < num_internal; ++i) {
        int left_encoded = h_children[i].x;
        int right_encoded = h_children[i].y;

        int left_idx = DecodeIndex(left_encoded);
        int right_idx = DecodeIndex(right_encoded);

        // 检查左孩子
        if (left_idx >= total_nodes) {
            printf("[Error] Node %d left child index out of bounds: %d\n", i, left_idx);
            error_count++;
        }
        else if (h_parent[left_idx] != i) {
            printf("[Error] Consistency Fail: Node %d thinks %d is Left Child, but %d thinks Parent is %d\n",
                i, left_idx, left_idx, h_parent[left_idx]);
            error_count++;
        }

        // 检查右孩子
        if (right_idx >= total_nodes) {
            printf("[Error] Node %d right child index out of bounds: %d\n", i, right_idx);
            error_count++;
        }
        else if (h_parent[right_idx] != i) {
            printf("[Error] Consistency Fail: Node %d thinks %d is Right Child, but %d thinks Parent is %d\n",
                i, right_idx, right_idx, h_parent[right_idx]);
            error_count++;
        }
    }

    // --- 检查 3: 树的遍历与叶子节点统计 (DFS) ---
    // 同时也用于检查是否存在环 (Cycle)
    std::vector<bool> visited(total_nodes, false);
    std::stack<int> s;
    s.push(0); // 从根节点 0 开始

    int visited_internal_count = 0;
    int visited_leaf_count = 0;
    bool cycle_detected = false;

    while (!s.empty()) {
        int curr = s.top();
        s.pop();

        if (visited[curr]) {
            printf("[Error] Cycle detected! Node %d visited twice.\n", curr);
            cycle_detected = true;
            break;
        }
        visited[curr] = true;

        // 判断是否为叶子节点
        // 根据你的内存布局: 叶子索引范围 [N, 2N-1]
        bool is_leaf = (curr >= num_triangles);

        if (is_leaf) {
            visited_leaf_count++;
        }
        else {
            visited_internal_count++;
            // 将子节点压栈
            // 注意：因为 curr 是内部节点索引，可以直接查 h_children
            if (curr >= num_internal) {
                // 理论上不应该发生，因为 gap 区域 (N-1) 不应该被访问到
                printf("[Error] Traversed into gap/invalid internal index: %d\n", curr);
                continue;
            }

            int left = DecodeIndex(h_children[curr].x);
            int right = DecodeIndex(h_children[curr].y);

            s.push(right);
            s.push(left);
        }
    }

    // --- 汇总报告 ---
    printf("Topology Summary:\n");
    printf("  > Internal Nodes Visited: %d (Expected: %d)\n", visited_internal_count, num_internal);
    printf("  > Leaf Nodes Visited    : %d (Expected: %d)\n", visited_leaf_count, num_triangles);

    if (visited_leaf_count != num_triangles) {
        printf("[Error] Unreachable leaves detected! Expected %d, found %d.\n", num_triangles, visited_leaf_count);
        error_count++;
    }

    if (!cycle_detected && error_count == 0) {
        printf("[Pass] Tree topology is valid.\n");
    }
    else {
        printf("[Fail] Found %d errors in tree topology.\n", error_count);
    }
    printf("=============================================\n\n");
}
// 内部递归函数
int GetDepthRecursive(int nodeIdx, int num_tris, const std::vector<int2>& h_children) {
    // 解码索引
    int realIdx = (nodeIdx < 0) ? ~nodeIdx : nodeIdx;

    // 判断是否为叶子节点 (根据你的布局：索引 >= N 为叶子)
    if (realIdx >= num_tris) {
        return 1;
    }

    // 只有内部节点 (0 ~ num_tris-2) 才有孩子
    if (realIdx >= num_tris - 1) return 0;

    int left = h_children[realIdx].x;
    int right = h_children[realIdx].y;

    int leftDepth = GetDepthRecursive(left, num_tris, h_children);
    int rightDepth = GetDepthRecursive(right, num_tris, h_children);

    return 1 + std::max(leftDepth, rightDepth);
}
// 主入口函数
void ComputeAndPrintMaxDepth(const LBVHData& d_bvh, int num_triangles) {
    if (num_triangles <= 0) return;
    if (num_triangles == 1) {
        printf("BVH Max Depth: 1\n");
        return;
    }

    int num_internal = num_triangles - 1;
    std::vector<int2> h_children(num_internal);

    // 从 GPU 下载孩子节点信息
    cudaMemcpy(h_children.data(), d_bvh.child_nodes, sizeof(int2) * num_internal, cudaMemcpyDeviceToHost);

    // 从根节点 0 开始计算
    int maxDepth = GetDepthRecursive(0, num_triangles, h_children);

    printf("LBVH Max Tree Depth:  %d\n", maxDepth);
}

void BuildLBVH(LBVHData& d_bvh_data, const MeshData& d_mesh_data)
{
    int num_triangles = d_mesh_data.num_triangles;
    int blockSize = 256;
    int numBlocks = (num_triangles + blockSize - 1) / blockSize;

    // prepare data
    int* d_atomic_flags;
    cudaMalloc((void**)&d_atomic_flags, sizeof(int) * num_triangles);
    cudaMemset(d_atomic_flags, 0, sizeof(int) * num_triangles);
    cudaMemset(d_bvh_data.parent, -1, sizeof(int) * (2 * num_triangles));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Step 1: 计算每个三角形的 AABB 和质心 (Kernel)
    ComputeAABB << <numBlocks, blockSize >> > (d_bvh_data, d_mesh_data);

    // Step 2: 计算世界包围盒 (Thrust Host Call)
    ComputeWorldAABB(d_bvh_data, d_mesh_data);

    // Step 3: 计算 Morton 码 (Kernel)
    ComputeMortonCode << <numBlocks, blockSize >> > (d_bvh_data, d_mesh_data);

    // Step 4: 排序 (Thrust Host Call)
    SortMortonCode(d_bvh_data, d_mesh_data);

    // Step 5: 生成 BVH 层次结构 (Kernel)
    GenerateHierarchy << <numBlocks, blockSize >> > (d_bvh_data, d_mesh_data);

    // Step 6: 设置叶子节点 (Kernel)
    SetupLeafNodes << <numBlocks, blockSize >> > (d_bvh_data, d_mesh_data);

    // Step 7: 构建内部AABB包围盒 (Kernel)
    RefitAABB << <numBlocks, blockSize >> > (d_bvh_data, d_mesh_data, d_atomic_flags);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop); 

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
	printf("LBVH Build Time: %.3f ms\n", milliseconds);

    // Step 8：构建Escape Index
    // 初始化整个 escape_indices 缓冲区为 -1 (覆盖所有节点)
    cudaMemset(d_bvh_data.escape_indices, -1, sizeof(int) * (2 * num_triangles));
    BuildEscapeIdx << <numBlocks, blockSize >> > (d_bvh_data, d_mesh_data);

    // 叶子节点测试函数
    //TestPrintLeafNodes(d_bvh_data, num_triangles); // 叶子节点没错误
    // cudaDeviceSynchronize();
    // DebugPrintBVH(d_bvh_data, num_triangles);
    cudaDeviceSynchronize();
    ComputeAndPrintMaxDepth(d_bvh_data, d_mesh_data.num_triangles);
	// 测试拓扑结构正确性
    cudaDeviceSynchronize();
    TestHierarchyLogic(d_bvh_data, num_triangles);

    cudaFree(d_atomic_flags);
}

__device__ glm::vec3 TemperatureColor(float t) {
    t = glm::clamp(t, 0.0f, 1.0f);
    glm::vec3 c = glm::vec3(0.0);
    if (t < 0.25f) { // Blue -> Cyan
        c = glm::mix(glm::vec3(0, 0, 1), glm::vec3(0, 1, 1), t * 4.0f);
    }
    else if (t < 0.5f) { // Cyan -> Green
        c = glm::mix(glm::vec3(0, 1, 1), glm::vec3(0, 1, 0), (t - 0.25f) * 4.0f);
    }
    else if (t < 0.75f) { // Green -> Yellow
        c = glm::mix(glm::vec3(0, 1, 0), glm::vec3(1, 1, 0), (t - 0.5f) * 4.0f);
    }
    else { // Yellow -> Red
        c = glm::mix(glm::vec3(1, 1, 0), glm::vec3(1, 0, 0), (t - 0.75f) * 4.0f);
    }
    return c;
}

__global__ void VisualizeLBVHKernel(
    glm::vec3* output_buffer,
    int width, int height,
    Camera cam, // 注意：Camera 直接传值，方便 Kernel 内访问
    LBVHData d_bvh_data,
    int num_triangles)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * width);

    if (x >= width || y >= height) return;

    // 1. 生成光线 (简化版 PinHole，不带 Jitter)
    // 假设 cam.view, cam.right, cam.up 是单位向量
    glm::vec3 dir = glm::normalize(cam.view
        + cam.right * cam.pixelLength.x * ((float)x - (float)width * 0.5f)
        - cam.up * cam.pixelLength.y * ((float)y - (float)height * 0.5f)
    );

    Ray ray;
    ray.origin = cam.position;
    ray.direction = dir;
    glm::vec3 inv_dir = 1.0f / dir; // 预计算倒数，加速 AABB 测试

    // 2. 遍历 BVH 并计数
    int steps = 0;
    int stack[64];
    int stack_ptr = 0;
    int node_idx = 0; // Root

    // 检查 Root AABB
    float t_root = BoudingboxIntersetionTest(
        MakeVec3(d_bvh_data.aabb_min[0]),
        MakeVec3(d_bvh_data.aabb_max[0]), ray, inv_dir);

    if (t_root == -1.0f) node_idx = -1;

    // 最大显示步数（超过这个数变红，可调节）
    const float MAX_STEPS = 300.0f;

    while (node_idx != -1 || stack_ptr > 0)
    {
        if (node_idx == -1) node_idx = stack[--stack_ptr];

        steps++; // 增加开销计数

        if (node_idx >= num_triangles) // 叶子节点
        {
            node_idx = -1; // 不深入三角形，只看 AABB 结构
        }
        else // 内部节点
        {
            int2 children = d_bvh_data.child_nodes[node_idx];
            int left = DecodeNode(children.x);
            int right = DecodeNode(children.y);

            float t_l = BoudingboxIntersetionTest(
                MakeVec3(d_bvh_data.aabb_min[left]),
                MakeVec3(d_bvh_data.aabb_max[left]), ray, inv_dir);
            float t_r = BoudingboxIntersetionTest(
                MakeVec3(d_bvh_data.aabb_min[right]),
                MakeVec3(d_bvh_data.aabb_max[right]), ray, inv_dir);

            bool hit_l = (t_l != -1.0f);
            bool hit_r = (t_r != -1.0f);

            if (hit_l && hit_r) {
                // 按距离排序入栈：先访问近的
                int first = (t_l < t_r) ? left : right;
                int second = (t_l < t_r) ? right : left;
                stack[stack_ptr++] = second;
                node_idx = first;
            }
            else if (hit_l) node_idx = left;
            else if (hit_r) node_idx = right;
            else node_idx = -1;
        }

        if (stack_ptr >= 64) break; // 栈溢出保护
    }

    // 3. 输出颜色到 Final Image Buffer
    output_buffer[index] = TemperatureColor((float)steps / MAX_STEPS);
}

void VisualizeLBVH(
    glm::vec3* output_buffer,
    int width, int height,
    const Camera& cam,
    const LBVHData& d_bvh_data,
    int num_triangles)
{
    const dim3 blockSize(16, 16);
    const dim3 blocksPerGrid(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y);

    VisualizeLBVHKernel << <blocksPerGrid, blockSize >> > (
        output_buffer,
        width, height,
        cam,
        d_bvh_data,
        num_triangles
        );
}