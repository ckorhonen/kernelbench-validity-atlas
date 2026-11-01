# GPU kernel optimization

Develop faster, more efficient GPU kernels, using KernelBench as a shared experimental foundation. Research may explore individual operations, operation fusion, memory management, hardware adaptation and automated kernel generation. The broad goal is to improve computational performance while preserving correctness, and discover optimization methods that generalize across workloads and hardware.

Use the tasks and evaluation instructions in the [official KernelBench repository](https://github.com/ScalingIntelligence/KernelBench). A kernel must preserve the operation's specified behavior for the stated input shapes, data types and numerical tolerances. Declare any restrictions on supported inputs and test correctness before measuring speed.

Compare against a reproducible reference implementation on the same hardware and inputs. Record the repository revision, GPU, software environment, precision and timing procedure. Separate compilation and warm-up costs from steady-state timing, synchronize asynchronous GPU work, and use enough repetitions to assess variability. Report which tasks improve, which regress and which fail correctness checks; a performance summary should not conceal unsuccessful cases.

This is an open-ended research problem. Useful findings and reproducible improvements constitute progress; there is no single final solution. A result may address a particular operation or a general method, provided its scope is clear and claims of transfer are supported by tests on additional workloads or hardware.

Sources: [KernelBench repository](https://github.com/ScalingIntelligence/KernelBench) · [KernelBench: Can LLMs Write Efficient GPU Kernels?](https://arxiv.org/abs/2502.10517).
