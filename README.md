# Note-on-study-Machine-learning
## Why do we need batches? What does pinned memory mean?
See: https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
Purpose: optimize the data transfer.
Reason:  Mismatch in the performance between CPU (8 GB/s data transfer to memory) and GPU (144 GB/s). 10 years ago  

This leads to consequences: GPU is already processed but CPU is still transforming data. Or they equally process.

Solution: batching many small transfers (avoiding repeatedly read data); optimizing the time (measure CPU and GPU time); GPU access data directly from memory.

Answer: 1. On the GPU kernel, it could be faster than the data transfers. Small batches could avoid such a situation.
2. Due to GPU not read the memory directly, the intermediate transfers are from CPU pageable to pinned in memory. We can skip this process by pinning the data in memory.
