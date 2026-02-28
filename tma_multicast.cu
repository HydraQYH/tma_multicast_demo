#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/device_kernel.h>
#include <cute/tensor.hpp>

#include <cub/cub.cuh>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

template <
    typename T,
    typename GTensor,
    typename SmemLayout,
    typename TMA,
    typename TiledCopy>
__global__ void multicast_kernel(GTensor g_tensor, CUTLASS_GRID_CONSTANT TMA const tma, int row, int col, TiledCopy tiled_copy) {
  using namespace cute;
  extern __shared__ __align__(128) uint8_t shared_memory[];
  T* data = reinterpret_cast<T*>(shared_memory);
  uint64_t* mbar = reinterpret_cast<uint64_t*>(shared_memory + sizeof(T) * cosize(SmemLayout{}));

  dim3 cta_idx_in_clutser = block_id_in_cluster();
  Tensor s_tensor = make_tensor(make_smem_ptr(data), SmemLayout{});
  Tensor g_tensor_coord = tma.get_tma_tensor(make_shape(row, col));
  Tensor g_tensor_tile = local_tile(g_tensor_coord, shape(s_tensor), 0);

  auto [tma_g, tma_s] = tma_partition(
    tma,
    make_coord(cta_idx_in_clutser.x, cta_idx_in_clutser.y, cta_idx_in_clutser.z),
    Layout<Shape<_2, _2, _1>>{},
    group_modes<0, 2>(s_tensor),
    group_modes<0, 2>(g_tensor_tile)
  );
  uint16_t tma_mask = create_tma_multicast_mask(
    Layout<Shape<_2, _2, _1>>{},
    make_coord(_, _, _)
    // make_coord(cta_idx_in_clutser.x, cta_idx_in_clutser.y, cta_idx_in_clutser.z)
  );
#ifndef NDEBUG
  if (thread(0, 3)) {
    print(g_tensor_tile);
    printf("\n");
    print(group_modes<0, 2>(s_tensor));
    printf("\n");
    print(group_modes<0, 2>(g_tensor_tile));
    printf("\n");
    printf("CTA Index: (%u, %u, %u)\n", cta_idx_in_clutser.x, cta_idx_in_clutser.y, cta_idx_in_clutser.z);
    print(tma_g);
    printf("\n");
    print(tma_s);
    printf("\n");
    printf("%x\n", tma_mask);
  }
#endif
  // Initialize Barriers
  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane_predicate = cute::elect_one_sync();
  using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;
  if ((warp_idx == 0) && lane_predicate) {
    ProducerBarType::init(mbar, 1);
  }
  cluster_sync();
  constexpr int tma_transaction_bytes = size(s_tensor) * sizeof(T) / 4;
  if ((warp_idx == 0) && lane_predicate) {
    // Set expected Tx Bytes after each reset / init
    ProducerBarType::arrive_and_expect_tx(mbar, tma_transaction_bytes);
    copy(tma.with(*mbar, tma_mask), tma_g, tma_s);
  }
  ProducerBarType::wait(mbar, 0);

  if (thread(0, 3)) {
    print_tensor(s_tensor);
  }

#if 0
  using Tiler_MN = typename TiledCopy::Tiler_MN;
  auto tiler_mn = Tiler_MN{};
  auto s_tensor_tiled = flat_divide(s_tensor, tiler_mn);

  ThrCopy thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
  auto s_tensor_tiled_thr_part = thr_copy.partition_S(s_tensor_tiled);
  auto r_tensor_tiled_thr_part = make_fragment_like(s_tensor_tiled_thr_part);
  copy(tiled_copy, s_tensor_tiled_thr_part, r_tensor_tiled_thr_part);
#ifndef NDEBUG
  if (thread0()) {
    print(s_tensor_tiled);
    printf("\n");
    print(s_tensor_tiled_thr_part);
    printf("\n");
  }
#endif
  T acc(0);
  for (int i = 0; i < size(r_tensor_tiled_thr_part); i++) {
    acc += r_tensor_tiled_thr_part(i);
  }
  
  namespace cg = cooperative_groups;
  cg::thread_block cta = cg::this_thread_block();
  auto cta_tiled_partition = cg::tiled_partition<128>(cta);
  acc = cg::reduce(cta_tiled_partition, acc, cg::plus<T>());
  if (thread0()) {
    printf("CTA Sum: %d\n", acc);
  }
#endif
}

template<typename T>
void tma_multicast_demo(T* inp, int row, int col) {
  using namespace cute;
  constexpr int BLOCK_M = 64;
  constexpr int BLOCK_N = 64;

  auto g_tensor = make_tensor(
    make_gmem_ptr(inp),
    make_layout(make_shape(row, col), cute::LayoutRight{})  // Row-Major
  );

  auto s_block_layout = make_layout(make_shape(Int<BLOCK_M>{}, Int<BLOCK_N>{}), cute::LayoutRight{});
  auto tma_atom = make_tma_atom(SM90_TMA_LOAD_MULTICAST{}, g_tensor, s_block_layout, shape(s_block_layout), Int<4>{});
  print(tma_atom);

  // S2R Copy
  auto thr_layout = make_layout(make_shape(Int<8>{}, Int<16>{}), LayoutRight{});
  auto val_layout = make_layout(make_shape(Int<1>{}, Int<4>{}));
  using CopyOp = cute::UniversalCopy<cutlass::AlignedArray<T, cute::size(val_layout)>>;
  using CopyAtom = cute::Copy_Atom<CopyOp, T>;
  auto tiled_copy = cute::make_tiled_copy(CopyAtom{}, thr_layout, val_layout);
  print(tiled_copy);

  // Launch parameter setup
  int smem_size = sizeof(T) * cosize(s_block_layout) + sizeof(uint64_t);  // mbarrier
  printf("Shared Memory Size: %d\n", smem_size);
  dim3 dimBlock(size(tiled_copy), 1, 1);
  dim3 dimCluster(2, 2, 1);
  dim3 dimGrid(row / BLOCK_M * 2, col / BLOCK_N * 2, 1);
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};

  void const* kernel_ptr = reinterpret_cast<void const*>(&multicast_kernel<T, decltype(g_tensor), decltype(s_block_layout), decltype(tma_atom), decltype(tiled_copy)>);

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
    kernel_ptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    smem_size));

  char const* status_str = cutlassGetStatusString(cutlass::launch_kernel_on_cluster(params, kernel_ptr, g_tensor, tma_atom, row, col, tiled_copy));
  std::cout << status_str << std::endl;
  CUTE_CHECK_ERROR(cudaDeviceSynchronize());
}

int main(void) {
  using T = int;
  thrust::host_vector<T> h_A(64 * 64);
  T* ptr = h_A.data();
  for (size_t i = 0; i < 64; i++) {
    for (size_t j = 0; j < 64; j++) {
      ptr[i * 64 + j] = static_cast<T>(1); 
    }
  }
  thrust::device_vector<T> d_A = h_A;
  tma_multicast_demo(d_A.data().get(), 64, 64);
  return 0;
}
