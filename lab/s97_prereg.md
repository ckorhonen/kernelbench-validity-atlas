# s97 prereg — E1: reference LayerNorm realization stability (frozen pre-run, 2026-11-13 23:1xZ)

Question: is the reference realization for KernelBench L1/40 LayerNorm stable across input draws at fixed
shape? The exploitation plan's order-budget direction builds a merge-tree distance coordinate against "the
reference's tree"; if the reference dispatches more than one configuration at fixed shape, that coordinate is
undefined.

Design: three shapes (official (16,64,256,256) with 4,194,304 elements per row; reduced (16,32,128,128) 524,288;
reduced (16,16,64,64) 65,536), 8 draws each (seeds 9700+d), fp32, TF32 off both matmul and cudnn backends.
Per draw: torch.profiler CUDA activity -> realized kernel name set and counts; output sha256, mean, std;
20-rep median wall time (GPU at 96% from a resident service, so timing is recorded but carries NO speed claim).

Coded verdict: REFERENCE_STABLE iff exactly one distinct kernel-name set per shape across 8 draws and 8 distinct
output checksums (i.e. inputs differ, realization does not). REFERENCE_UNSTABLE otherwise.
Kill clause: REFERENCE_UNSTABLE at any shape => the tree-distance coordinate is undefined at that shape =>
pivot the lane to (B) domain atlas for that shape, and record which shape broke it.
Secondary record (descriptive only): kernel names are expected to carry the vectorized-layer-norm block/thread
hints; if a draw changes vectorization width, that is a realization change even under one kernel name.
