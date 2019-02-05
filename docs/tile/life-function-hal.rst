.. tile/life-function-hal.rst:

==============================
Through the Hardware Interface
==============================

PlaidML manages access to devices capable of executing Tile code through an internal hardware abstraction layer, or "HAL", which is implemented by architecture-specific hardware modules. After compiling a Tile function into a ``lang::KernelList``, the PlaidML runtime calls into the hardware module associated with the target device, which generates machine code, manages memory, and schedules execution.

HAL Interface
-------------

PlaidML currently offers two implementations of the HAL interface. The OpenCL module connects to the system's OpenCL driver and provides access to all available OpenCL devices, such as GPUs; the OpenCL driver may also offer execution on the CPU. The LLVM module needs no driver; it provides a single device representing the system CPU.

The PlaidML core loads both hardware modules at startup so they can query the system for available hardware. You can call :doc:`plaidml <../plaidml>`\ ``.devices`` to get the list of devices. Once you've chosen a device, you can use the :doc:`../api/plaidml.Device` object to create :doc:`../api/plaidml.Tensor` instances; running a function bound to these tensors will associate its execution with the corresponding device.

Each hardware module provides a compiler backend, which is an implementation of ``hal::Compiler``. The ``hal::Compiler::Build`` function accepts a ``KernelList`` produced by the Tile compiler frontend as its input, and as output it produces an instance of some device-specific implementation of ``hal::Library``. A ``Library`` is an abstract representation of executable code, in whatever form it is that the target device expects. Each implementation of ``Compiler`` is therefore free to implement code generation in whatever way suits the target architecture.

Since GPU performance is generally much better than CPU performance, you'll most often want to use devices provided by the OpenCL module; let's look at its backend architecture first.


OpenCL Compiler
---------------

While the semtree built for an OpenCL device will be optimized for that device's memory and register characteristics, the semtree itself has no knowledge of OpenCL-specific semantics, so the OpenCL module begins by checking for driver-specific features such as the availability of fp16 and fp64 types. Next, an optimization pass walks the semtree, replacing expression patterns with optimized OpenCL builtins, such as ``mad()``.

After optimization, a second traversal of the semtree produces OpenCL source code. A ``sem::Type`` is computed for each expression node. For expressions with multiple subexpressions, the subtypes are promoted to a common output type, and the subexpressions are converted accordingly. The promotion rule is specific to Tile semantics, so that semtree behavior is consistent between OpenCL and non-OpenCL devices; the OpenCL source code produced in this stage will be subject to a second round of type analysis when it is compiled by the OpenCL driver.

The generated source code is then passed on to the OpenCL driver for asynchronous parsing and code generation. The result will be a ``cl_program``, and this ``cl_program`` is then wrapped up with the original ``lang::KernelInfo`` objects to become an ``opencl::Library``, which is the compiler backend's output.

Example Kernels
_______________

Here is OpenCL code produced by the example::

    #pragma OPENCL EXTENSION cl_khr_fp64 : enable

    __kernel void kernel_c1_sdk_0(__global float* X_T2, __global const float* X_I_0)
    {
      int tid = get_local_id(0);
      int i1_i2_gid = (get_group_id(0) * 32);
      int i1_i2_tid = (tid % 32);
      int i1_i2_cond = ((i1_i2_gid != 256) || (i1_i2_tid < 4));
      if (i1_i2_cond)
      {
        int gout_idx = (i1_i2_gid + i1_i2_tid);
        float LX_I_0 = X_I_0[gout_idx];
        float LX_T2 = native_log(LX_I_0);
        X_T2[gout_idx] = LX_T2;
      }
    }

    __kernel void kernel_c1_sdk_1(__global float* X_T3, __global const float* in1, __global const float* in2)
    {
      int tid = get_local_id(0);
      float agg[1] = {0, };
      __local float in1_shared[26];
      __local float in2_shared[26];
      int x_gid = get_group_id(0);
      for (int y_gid = 0; y_gid < 26; y_gid += 26)
      {
        {
          int gbase = (y_gid + (x_gid * 26));
          int y_x_tid = (tid % 32);
          int y_x_cond = (y_x_tid < 26);
          if (y_x_cond)
          {
            int gidx = (gbase + y_x_tid);
            in1_shared[y_x_tid] = in1[clamp((int)gidx, (int)0, (int)259)];
          }
        }
        {
          int gbase = (y_gid + (x_gid * 26));
          int y_x_tid = (tid % 32);
          int y_x_cond = (y_x_tid < 26);
          if (y_x_cond)
          {
            int gidx = (gbase + y_x_tid);
            in2_shared[y_x_tid] = in2[clamp((int)gidx, (int)0, (int)259)];
          }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        int y_tid = (tid % 32);
        int y_cond = (y_tid < 26);
        if (y_cond)
        {
          float val1 = in1_shared[y_tid];
          float val2 = in2_shared[y_tid];
          float agg_rhs = mad(val2, val1, agg[0]);
          agg[0] = agg_rhs;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
      }
      __local float merge_shared[32];
      {
        merge_shared[tid] = agg[0];
        barrier(CLK_LOCAL_MEM_FENCE);
        if ((tid < 16))
        {
          merge_shared[tid] = (merge_shared[tid] + merge_shared[(tid + 16)]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if ((tid < 8))
        {
          merge_shared[tid] = (merge_shared[tid] + merge_shared[(tid + 8)]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if ((tid < 4))
        {
          merge_shared[tid] = (merge_shared[tid] + merge_shared[(tid + 4)]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if ((tid < 2))
        {
          merge_shared[tid] = (merge_shared[tid] + merge_shared[(tid + 2)]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if ((tid < 1))
        {
          merge_shared[tid] = (merge_shared[tid] + merge_shared[(tid + 1)]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if ((tid < 1))
        {
          agg[0] = merge_shared[tid];
        }
      }
      if ((tid < 1))
      {
        float LX_T0 = agg[0];
        float LX_T3 = (-LX_T0);
        X_T3[x_gid] = LX_T3;
      }
    }

Note that the produced kernels are hardware-specific, so you will likely see different kernels produced if you examine this example on your own machine.

LLVM Compiler
-------------

The LLVM-based backend begins by traversing the semtree, generating LLVM IR. As with the OpenCL backend, each expression node's type is computed and its subtypes are promoted using a Tile-specific algorithm, so that behavior is consistent between backends.

Since this backend generates a lower-level IR, instead of source code, it is necessary to be more specific about numeric types and the use of builtin functions. Each arithmetic expression is compiled differently depending on its use of signed, unsigned, or floating-point arithmetic.

Instead of optimizing the semtree first, as the OpenCL backend does, the LLVM compiler translates the semtree into LLVM IR as-is, then runs LLVM optimization passes on the output. These standard optimizations include loop simplification, instruction combination and simplification, and dead code elimination, among others. 

Finally, the compiled kernel is wrapped in an ``llvm::ExecutionEngine``, and this is the contents of the ``Library`` object which is the compiler backend's output.


Execution
---------

After the ``KernelList`` representing the original function has been compiled into a ``hal::Library``, the hardware module must also provide some means of executing it. An implementation of the ``hal::Executor`` interface represents the runtime execution environment.

The ``Executor::Prepare`` function readies one kernel from a ``hal::Library``, in whatever way is meaningful for the target device. The kernel is specified by index, and the function returns a ``hal::Kernel`` instance.

The ``Executor::Copy`` function handles dependency-sensitive dataflow, moving buffer contents back and forth between device and shared memory regions. This is the core mechanism which implements the ``mmap_current`` and ``mmap_discard`` functions provided in the :doc:`PlaidML API <../plaidml>`.

The copy operation begins by waiting for a list of events to complete. These events may represent the completion of other copy operations, presumably because those copies are moving data into the source buffer, or they may represent the completion of kernel execution; in any case the copy will not proceed until the completion of all dependency events signals that the source data is complete. The result of the copy is another event, which signals readiness of the output data.

An instance of ``hal::Kernel`` represents invokable code. Its primary method is ``hal::Kernel::Run``, which accepts an array of parameters and an array of dependency events. As with ``Executor::Copy``, the kernel will first wait for all dependency events to complete, signalling the readiness of all input parameters, and the result of this asynchronous call is another event, which will be resolved when kernel execution completes.

Having first been decomposed into flat contractions, each kernel invocation represents one single application of the original function within a notional three-dimensional stride pattern across the parameters. Each hardware module implements this iteration differently.

In the OpenCL module, the mechanism used is ``clEnqueueNDRangeKernel``. This function establishes a work group for a given kernel, so that multiple instances of the work group can be executed in parallel on different compute units. Each kernel invocation receives a unique ID, which can then be used to look up the corresponding parameter tensor elements.

In the LLVM module, kernel instances are executed in a thread pool. Using one thread for each CPU core, each thread invokes the kernel function, computing a unique ID in the same fashion as OpenCL's range kernel system, dividing the work group stride pattern evenly among available threads. The originating thread then blocks until all workers have completed.

After execution completes, each ``Run`` method signals the completion of the corresponding event. Since the act of binding the function to its output tensor instances will attach the execution completion event to their dependency lists, the ``Executor::Copy`` call resulting from an ``mmap_current`` or ``mmap_discard`` on those :doc:`../api/plaidml.Tensor` instances will therefore block until kernel execution completes, allowing synchronization back up the stack to the original client.


