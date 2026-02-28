# TMA Multicast Demo
Compile CMD:
```
nvcc -forward-unknown-to-host-compiler \
  -O3 -DNDEBUG -std=c++17 \
  -lcuda \
  -I${CUTLASS_PATH}$/include \
  "--generate-code=arch=compute_90a,code=[sm_90a]" \
  -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 \
  -DCUTLASS_DEBUG_TRACE_LEVEL=0 \
  --expt-relaxed-constexpr \
  -ftemplate-backtrace-limit=0 \
  -o tma_multicast \
  tma_multicast.cu
```