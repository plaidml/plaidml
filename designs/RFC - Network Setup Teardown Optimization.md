# RFC - Network Setup/Teardown Optimization

## Intro
Here's the current (2020-10-08) comp dialect representation of main() from the EDSL C++ Dot test, just before lowering to LLVMIR:

```mlir
func @main(%arg0: !comp.device, %arg1: memref<8x16xf32>, %arg2: memref<16x32xf32>, %arg3: memref<8x32xf32>) {
  %c2 = constant 2 : index
  %c32 = constant 32 : index
  %c1 = constant 1 : index
  %c8 = constant 8 : index
  %c4 = constant 4 : index
  %0 = comp.create_execenv %arg0 : (!comp.device) -> !comp.execenv<ocl:0,(11)>
  %1 = comp.alloc %0 %arg3 : (!comp.execenv<ocl:0,(11)>, memref<8x32xf32>) -> memref<8x32xf32, 11>
  %2 = "comp.schedule_func"(%0) ( {
    "gpu.launch_func"(%c4, %c1, %c1, %c1, %c2, %c32, %1) {kernel = @main_kernel::@main_kernel} : (index, index, index, index, index, index, memref<8x32xf32, 11>) -> ()
    "comp.schedule_end"() : () -> ()
  }) : (!comp.execenv<ocl:0,(11)>) -> !comp.event<ocl>
  %3 = comp.schedule_read %arg3 from %1 on %0 wait for %2 : (memref<8x32xf32>, memref<8x32xf32, 11>, !comp.execenv<ocl:0,(11)>, !comp.event<ocl>) -> !comp.event<ocl>
  comp.wait %3 : !comp.event<ocl>
  %4 = comp.alloc %0 %arg2 : (!comp.execenv<ocl:0,(11)>, memref<16x32xf32>) -> memref<16x32xf32, 11>
  %5 = comp.alloc %0 %arg1 : (!comp.execenv<ocl:0,(11)>, memref<8x16xf32>) -> memref<8x16xf32, 11>
  %6 = comp.schedule_write %arg3 to %1 on %0 : (memref<8x32xf32>, memref<8x32xf32, 11>, !comp.execenv<ocl:0,(11)>) -> !comp.event<ocl>
  comp.wait %6 : !comp.event<ocl>
  %7 = "comp.schedule_func"(%0) ( {
    "gpu.launch_func"(%c1, %c1, %c1, %c8, %c4, %c1, %4, %5, %1) {kernel = @main_kernel_0::@main_kernel} : (index, index, index, index, index, index, memref<16x32xf32, 11>, memref<8x16xf32, 11>, memref<8x32xf32, 11>) -> ()
    "comp.schedule_end"() : () -> ()
  }) : (!comp.execenv<ocl:0,(11)>) -> !comp.event<ocl>
  %8 = comp.schedule_read %arg2 from %4 on %0 wait for %7 : (memref<16x32xf32>, memref<16x32xf32, 11>, !comp.execenv<ocl:0,(11)>, !comp.event<ocl>) -> !comp.event<ocl>
  %9 = comp.schedule_read %arg1 from %5 on %0 wait for %7 : (memref<8x16xf32>, memref<8x16xf32, 11>, !comp.execenv<ocl:0,(11)>, !comp.event<ocl>) -> !comp.event<ocl>
  %10 = comp.schedule_read %arg3 from %1 on %0 wait for %7 : (memref<8x32xf32>, memref<8x32xf32, 11>, !comp.execenv<ocl:0,(11)>, !comp.event<ocl>) -> !comp.event<ocl>
  comp.wait %8, %9, %10 : !comp.event<ocl>, !comp.event<ocl>, !comp.event<ocl>
  comp.dealloc %0 %4 : (!comp.execenv<ocl:0,(11)>, memref<16x32xf32, 11>) -> ()
  comp.dealloc %0 %5 : (!comp.execenv<ocl:0,(11)>, memref<8x16xf32, 11>) -> ()
  comp.dealloc %0 %1 : (!comp.execenv<ocl:0,(11)>, memref<8x32xf32, 11>) -> ()
  comp.destroy_execenv %0 : !comp.execenv<ocl:0,(11)>
  return
}
```

A network typically isn't run just once; usually, it's compiled once, loaded once, and then run multiple times before being deleted (once).  So we can get a substantial performance boost by factoring out the bits that can be done once, instead of performing them every time the network is run&mdash;e.g. resource allocation and constant pre-computation.

## High-Level Design

We want to perform the setup/teardown optimization at the comp dialect:
* At the comp dialect, we have semantic information about the resources used by the network, making it straightforward
  to move setup and teardown as appropriate; the next lowering after comp is to transform the code to LLVMIR, at which
  point the semantic information's been lost.
* If we perform these optimizations at the comp dialect, we get their benefit for all devices that use comp.

### Reified Looping

One simple way to accomplish setup/teardown optimization is to phrase the problem in a way that lets us apply
standard compilation patterns.  Looking at the gestalt, what we have when we're running a network repeatedly might be
thought of as:

    for (inputs: requests()) {
      yield main(inputs);
    }

We can reify this by defining a `comp.loop` operation.  Here's what it looks like, when applied to our original code:

```mlir
func @main(%arg0: !comp.device) {
  comp.loop(%arg1: memref<8x16xf32>, %arg2: memref<16x32xf32>, %arg3: memref<8x32xf32>) {
    %c2 = constant 2 : index
    %c32 = constant 32 : index
    %c1 = constant 1 : index
    %c8 = constant 8 : index
    %c4 = constant 4 : index
    %0 = comp.create_execenv %arg0 : (!comp.device) -> !comp.execenv<ocl:0,(11)>
    %1 = comp.alloc %0 %arg3 : (!comp.execenv<ocl:0,(11)>, memref<8x32xf32>) -> memref<8x32xf32, 11>
    %2 = "comp.schedule_func"(%0) ( {
      "gpu.launch_func"(%c4, %c1, %c1, %c1, %c2, %c32, %1) {kernel = @main_kernel::@main_kernel} : (index, index, index, index, index, index, memref<8x32xf32, 11>) -> ()
      "comp.schedule_end"() : () -> ()
    }) : (!comp.execenv<ocl:0,(11)>) -> !comp.event<ocl>
    %3 = comp.schedule_read %arg3 from %1 on %0 wait for %2 : (memref<8x32xf32>, memref<8x32xf32, 11>, !comp.execenv<ocl:0,(11)>, !comp.event<ocl>) -> !comp.event<ocl>
    comp.wait %3 : !comp.event<ocl>
    %4 = comp.alloc %0 %arg2 : (!comp.execenv<ocl:0,(11)>, memref<16x32xf32>) -> memref<16x32xf32, 11>
    %5 = comp.alloc %0 %arg1 : (!comp.execenv<ocl:0,(11)>, memref<8x16xf32>) -> memref<8x16xf32, 11>
    %6 = comp.schedule_write %arg3 to %1 on %0 : (memref<8x32xf32>, memref<8x32xf32, 11>, !comp.execenv<ocl:0,(11)>) -> !comp.event<ocl>
    comp.wait %6 : !comp.event<ocl>
    %7 = "comp.schedule_func"(%0) ( {
      "gpu.launch_func"(%c1, %c1, %c1, %c8, %c4, %c1, %4, %5, %1) {kernel = @main_kernel_0::@main_kernel} : (index, index, index, index, index, index, memref<16x32xf32, 11>, memref<8x16xf32, 11>, memref<8x32xf32, 11>) -> ()
      "comp.schedule_end"() : () -> ()
    }) : (!comp.execenv<ocl:0,(11)>) -> !comp.event<ocl>
    %8 = comp.schedule_read %arg2 from %4 on %0 wait for %7 : (memref<16x32xf32>, memref<16x32xf32, 11>, !comp.execenv<ocl:0,(11)>, !comp.event<ocl>) -> !comp.event<ocl>
    %9 = comp.schedule_read %arg1 from %5 on %0 wait for %7 : (memref<8x16xf32>, memref<8x16xf32, 11>, !comp.execenv<ocl:0,(11)>, !comp.event<ocl>) -> !comp.event<ocl>
    %10 = comp.schedule_read %arg3 from %1 on %0 wait for %7 : (memref<8x32xf32>, memref<8x32xf32, 11>, !comp.execenv<ocl:0,(11)>, !comp.event<ocl>) -> !comp.event<ocl>
    comp.wait %8, %9, %10 : !comp.event<ocl>, !comp.event<ocl>, !comp.event<ocl>
    comp.dealloc %0 %4 : (!comp.execenv<ocl:0,(11)>, memref<16x32xf32, 11>) -> ()
    comp.dealloc %0 %5 : (!comp.execenv<ocl:0,(11)>, memref<8x16xf32, 11>) -> ()
    comp.dealloc %0 %1 : (!comp.execenv<ocl:0,(11)>, memref<8x32xf32, 11>) -> ()
    comp.destroy_execenv %0 : !comp.execenv<ocl:0,(11)>
  }
  return
}
```

With this in place, setup and teardown are simply a matter of hoisting operations out of the loop.

### Persistent Execution Environment

The most obvious initial transformation is to hoist the execution environment setup and teardown.  With our example code, we get:

```mlir
func @main(%arg0: !comp.device) {
  %c2 = constant 2 : index
  %c32 = constant 32 : index
  %c1 = constant 1 : index
  %c8 = constant 8 : index
  %c4 = constant 4 : index
  %0 = comp.create_execenv %arg0 : (!comp.device) -> !comp.execenv<ocl:0,(11)>
  comp.loop(%arg1: memref<8x16xf32>, %arg2: memref<16x32xf32>, %arg3: memref<8x32xf32>) {
    %1 = comp.alloc %0 %arg3 : (!comp.execenv<ocl:0,(11)>, memref<8x32xf32>) -> memref<8x32xf32, 11>
    %2 = "comp.schedule_func"(%0) ( {
      "gpu.launch_func"(%c4, %c1, %c1, %c1, %c2, %c32, %1) {kernel = @main_kernel::@main_kernel} : (index, index, index, index, index, index, memref<8x32xf32, 11>) -> ()
      "comp.schedule_end"() : () -> ()
    }) : (!comp.execenv<ocl:0,(11)>) -> !comp.event<ocl>
    %3 = comp.schedule_read %arg3 from %1 on %0 wait for %2 : (memref<8x32xf32>, memref<8x32xf32, 11>, !comp.execenv<ocl:0,(11)>, !comp.event<ocl>) -> !comp.event<ocl>
    comp.wait %3 : !comp.event<ocl>
    %4 = comp.alloc %0 %arg2 : (!comp.execenv<ocl:0,(11)>, memref<16x32xf32>) -> memref<16x32xf32, 11>
    %5 = comp.alloc %0 %arg1 : (!comp.execenv<ocl:0,(11)>, memref<8x16xf32>) -> memref<8x16xf32, 11>
    %6 = comp.schedule_write %arg3 to %1 on %0 : (memref<8x32xf32>, memref<8x32xf32, 11>, !comp.execenv<ocl:0,(11)>) -> !comp.event<ocl>
    comp.wait %6 : !comp.event<ocl>
    %7 = "comp.schedule_func"(%0) ( {
      "gpu.launch_func"(%c1, %c1, %c1, %c8, %c4, %c1, %4, %5, %1) {kernel = @main_kernel_0::@main_kernel} : (index, index, index, index, index, index, memref<16x32xf32, 11>, memref<8x16xf32, 11>, memref<8x32xf32, 11>) -> ()
      "comp.schedule_end"() : () -> ()
    }) : (!comp.execenv<ocl:0,(11)>) -> !comp.event<ocl>
    %8 = comp.schedule_read %arg2 from %4 on %0 wait for %7 : (memref<16x32xf32>, memref<16x32xf32, 11>, !comp.execenv<ocl:0,(11)>, !comp.event<ocl>) -> !comp.event<ocl>
    %9 = comp.schedule_read %arg1 from %5 on %0 wait for %7 : (memref<8x16xf32>, memref<8x16xf32, 11>, !comp.execenv<ocl:0,(11)>, !comp.event<ocl>) -> !comp.event<ocl>
    %10 = comp.schedule_read %arg3 from %1 on %0 wait for %7 : (memref<8x32xf32>, memref<8x32xf32, 11>, !comp.execenv<ocl:0,(11)>, !comp.event<ocl>) -> !comp.event<ocl>
    comp.wait %8, %9, %10 : !comp.event<ocl>, !comp.event<ocl>, !comp.event<ocl>
    comp.dealloc %0 %4 : (!comp.execenv<ocl:0,(11)>, memref<16x32xf32, 11>) -> ()
    comp.dealloc %0 %5 : (!comp.execenv<ocl:0,(11)>, memref<8x16xf32, 11>) -> ()
    comp.dealloc %0 %1 : (!comp.execenv<ocl:0,(11)>, memref<8x32xf32, 11>) -> ()
  }
  comp.destroy_execenv %0 : !comp.execenv<ocl:0,(11)>
  return
}
```

### Hoisting Allocs

The current `comp.alloc` instruction has two parts: it creates an allocation, and arranges for the allocation to hold
the contents of some other buffer.  We can move the allocations out of the loop by splitting them into seperate
`comp.alloc` and `comp.schedule_write` operations&mdash;making the data movement explicit&mdash;and then hoisting
the `comp.alloc` operations.

After this split and hoist, our example code look something like this (preserving value names where possible, for clarity):

```mlir
func @main(%arg0: !comp.device) {
  %c2 = constant 2 : index
  %c32 = constant 32 : index
  %c1 = constant 1 : index
  %c8 = constant 8 : index
  %c4 = constant 4 : index
  %0 = comp.create_execenv %arg0 : (!comp.device) -> !comp.execenv<ocl:0,(11)>
  %1 = comp.alloc %0 : (!comp.execenv<ocl:0,(11)>) -> memref<8x32xf32, 11>
  %4 = comp.alloc %0 : (!comp.execenv<ocl:0,(11)>) -> memref<16x32xf32, 11>
  %5 = comp.alloc %0 : (!comp.execenv<ocl:0,(11)>) -> memref<8x16xf32, 11>
  comp.loop(%arg1: memref<8x16xf32>, %arg2: memref<16x32xf32>, %arg3: memref<8x32xf32>) {
    %w1 = comp.schedule_write %arg3 to %1 on %0 : (!comp.execenv<ocl:0,(11)>, memref<8x32xf32>, memref<8x32xf32, 11>) -> !comp.event<ocl>
    comp.wait %w1 : !comp.event<ocl>
    %2 = "comp.schedule_func"(%0) ( {
      "gpu.launch_func"(%c4, %c1, %c1, %c1, %c2, %c32, %1) {kernel = @main_kernel::@main_kernel} : (index, index, index, index, index, index, memref<8x32xf32, 11>) -> ()
      "comp.schedule_end"() : () -> ()
    }) : (!comp.execenv<ocl:0,(11)>) -> !comp.event<ocl>
    %3 = comp.schedule_read %arg3 from %1 on %0 wait for %2 : (memref<8x32xf32>, memref<8x32xf32, 11>, !comp.execenv<ocl:0,(11)>, !comp.event<ocl>) -> !comp.event<ocl>
    comp.wait %3 : !comp.event<ocl>
    %w4 = comp.schedule_write %arg2 to %4 on %0 : (!comp.execenv<ocl:0,(11)>, memref<16x32xf32>, memref<16x32xf32, 11>) -> !comp.event<ocl>
    comp.wait %w4 : !comp.event<ocl>
    %w5 = comp.schedule_write %arg1 to %5 on %0 : (!comp.execenv<ocl:0,(11)>, memref<8x16xf32>, memref<8x16xf32, 11>) -> !comp.event<ocl>
    comp.wait %w5 : !comp.event<ocl>
    %6 = comp.schedule_write %arg3 to %1 on %0 : (memref<8x32xf32>, memref<8x32xf32, 11>, !comp.execenv<ocl:0,(11)>) -> !comp.event<ocl>
    comp.wait %6 : !comp.event<ocl>
    %7 = "comp.schedule_func"(%0) ( {
      "gpu.launch_func"(%c1, %c1, %c1, %c8, %c4, %c1, %4, %5, %1) {kernel = @main_kernel_0::@main_kernel} : (index, index, index, index, index, index, memref<16x32xf32, 11>, memref<8x16xf32, 11>, memref<8x32xf32, 11>) -> ()
      "comp.schedule_end"() : () -> ()
    }) : (!comp.execenv<ocl:0,(11)>) -> !comp.event<ocl>
    %8 = comp.schedule_read %arg2 from %4 on %0 wait for %7 : (memref<16x32xf32>, memref<16x32xf32, 11>, !comp.execenv<ocl:0,(11)>, !comp.event<ocl>) -> !comp.event<ocl>
    %9 = comp.schedule_read %arg1 from %5 on %0 wait for %7 : (memref<8x16xf32>, memref<8x16xf32, 11>, !comp.execenv<ocl:0,(11)>, !comp.event<ocl>) -> !comp.event<ocl>
    %10 = comp.schedule_read %arg3 from %1 on %0 wait for %7 : (memref<8x32xf32>, memref<8x32xf32, 11>, !comp.execenv<ocl:0,(11)>, !comp.event<ocl>) -> !comp.event<ocl>
    comp.wait %8, %9, %10 : !comp.event<ocl>, !comp.event<ocl>, !comp.event<ocl>
  }
  comp.dealloc %0 %4 : (!comp.execenv<ocl:0,(11)>, memref<16x32xf32, 11>) -> ()
  comp.dealloc %0 %5 : (!comp.execenv<ocl:0,(11)>, memref<8x16xf32, 11>) -> ()
  comp.dealloc %0 %1 : (!comp.execenv<ocl:0,(11)>, memref<8x32xf32, 11>) -> ()
  comp.destroy_execenv %0 : !comp.execenv<ocl:0,(11)>
  return
}
```

### Hoisting Kernel Creation

Another operation we'd like to hoist is kernel creation.  We can't do this with `gpu.launch_func`, since it requires a kernel as input.  We can instead define a set of new operations, and lower `gpu.launch_func` into a sequence of calls:

  * `comp.create_kernel` &mdash; create a kernel
  * `comp.schedule_kernel` &mdash; schedule a kernel for execution
  * `comp.destroy_kernel` &mdash; delete a kernel (with implicit wait on all outstanding scheduled executions)

To connect these, we reify `!comp.kernel` as a type parameterized by the kernel's operands (to make validation straightforward).

After hoisting, our example program looks like this (again preserving value names where possible):

```mlir
func @main(%arg0: !comp.device) {
  %c2 = constant 2 : index
  %c32 = constant 32 : index
  %c1 = constant 1 : index
  %c8 = constant 8 : index
  %c4 = constant 4 : index
  %0 = comp.create_execenv %arg0 : (!comp.device) -> !comp.execenv<ocl:0,(11)>
  %1 = comp.alloc %0 : (!comp.execenv<ocl:0,(11)>) -> memref<8x32xf32, 11>
  %4 = comp.alloc %0 : (!comp.execenv<ocl:0,(11)>) -> memref<16x32xf32, 11>
  %5 = comp.alloc %0 : (!comp.execenv<ocl:0,(11)>) -> memref<8x16xf32, 11>
  %k1 = comp.create_kernel %0 {kernel = @main_kernel::@main_kernel} : (!comp.execenv<ocl:0,(11)>) -> !comp.kernel<ocl, 8x32xf32>
  %k2 = comp.create_kernel %0 {kernel = @main_kernel_0::@main_kernel} : (!comp.execenv<ocl:0,(11)>) -> !comp.kernel<ocl, 16x32xf32, 8x16xf32, 8x32xf32>
  comp.loop(%arg1: memref<8x16xf32>, %arg2: memref<16x32xf32>, %arg3: memref<8x32xf32>) {
    %w1 = comp.schedule_write %arg3 to %1 on %0 : (!comp.execenv<ocl:0,(11)>, memref<8x32xf32>, memref<8x32xf32, 11>) -> !comp.event<ocl>
    comp.wait %w1 : !comp.event<ocl>
    %2 = comp.schedule_kernel %0, %k1, grid(%c4, %c1, %c1, %c1, %c2, %c32), args(%1) : (!comp.kernel<ocl, 8x32xf32>, index, index, index, index, index, index, memref<8x32xf32, 11>) -> !comp.event<ocl>
    %3 = comp.schedule_read %arg3 from %1 on %0 wait for %2 : (memref<8x32xf32>, memref<8x32xf32, 11>, !comp.execenv<ocl:0,(11)>, !comp.event<ocl>) -> !comp.event<ocl>
    comp.wait %3 : !comp.event<ocl>
    %w4 = comp.schedule_write %arg2 to %4 on %0 : (!comp.execenv<ocl:0,(11)>, memref<16x32xf32>, memref<16x32xf32, 11>) -> !comp.event<ocl>
    comp.wait %w4 : !comp.event<ocl>
    %w5 = comp.schedule_write %arg1 to %5 on %0 : (!comp.execenv<ocl:0,(11)>, memref<8x16xf32>, memref<8x16xf32, 11>) -> !comp.event<ocl>
    comp.wait %w5 : !comp.event<ocl>
    %6 = comp.schedule_write %arg3 to %1 on %0 : (memref<8x32xf32>, memref<8x32xf32, 11>, !comp.execenv<ocl:0,(11)>) -> !comp.event<ocl>
    comp.wait %6 : !comp.event<ocl>
    %7 = comp.schedule_kernel %0, %k2, grid(%c1, %c1, %c1, %c8, %c4, %c1), args(%4, %5, %1) : (!comp.kernel<ocl, 16x32xf32, 8x16xf32, 8x32xf32>, index, index, index, index, index, index, memref<16x32xf32, 11>, memref<8x16xf32, 11>, memref<8x32xf32, 11>) -> !comp.event<ocl>
    %8 = comp.schedule_read %arg2 from %4 on %0 wait for %7 : (memref<16x32xf32>, memref<16x32xf32, 11>, !comp.execenv<ocl:0,(11)>, !comp.event<ocl>) -> !comp.event<ocl>
    %9 = comp.schedule_read %arg1 from %5 on %0 wait for %7 : (memref<8x16xf32>, memref<8x16xf32, 11>, !comp.execenv<ocl:0,(11)>, !comp.event<ocl>) -> !comp.event<ocl>
    %10 = comp.schedule_read %arg3 from %1 on %0 wait for %7 : (memref<8x32xf32>, memref<8x32xf32, 11>, !comp.execenv<ocl:0,(11)>, !comp.event<ocl>) -> !comp.event<ocl>
    comp.wait %8, %9, %10 : !comp.event<ocl>, !comp.event<ocl>, !comp.event<ocl>
  }
  comp.destroy_kernel %k1 : !comp.kernel<ocl, 8x32xf32>
  comp.destroy_kernel %k2 : !comp.kernel<ocl, 16x32xf32, 8x16xf32, 8x32xf32>
  comp.dealloc %0 %4 : (!comp.execenv<ocl:0,(11)>, memref<16x32xf32, 11>) -> ()
  comp.dealloc %0 %5 : (!comp.execenv<ocl:0,(11)>, memref<8x16xf32, 11>) -> ()
  comp.dealloc %0 %1 : (!comp.execenv<ocl:0,(11)>, memref<8x32xf32, 11>) -> ()
  comp.destroy_execenv %0 : !comp.execenv<ocl:0,(11)>
  return
}
```

### Wait Cleanup

Although it's not strictly necessary for setup/teardown optimization, the code's a bit more optimal (and easier to grok)
if we coalesce `comp.wait` operations into subsequent operations that depend on the values touched by the `comp.wait`;
we can also optimize a bit by eliminating redundant memory transfers.

(Note that we can analyze the kernel code to determine which values are inputs and outputs, so that kernels reading the
same data aren't forced to serialize, and we can elide readbacks of constant inputs.)

With our example code, this produces:

```mlir
func @main(%arg0: !comp.device) {
  %c2 = constant 2 : index
  %c32 = constant 32 : index
  %c1 = constant 1 : index
  %c8 = constant 8 : index
  %c4 = constant 4 : index
  %0 = comp.create_execenv %arg0 : (!comp.device) -> !comp.execenv<ocl:0,(11)>
  %1 = comp.alloc %0 : (!comp.execenv<ocl:0,(11)>) -> memref<8x32xf32, 11>
  %4 = comp.alloc %0 : (!comp.execenv<ocl:0,(11)>) -> memref<16x32xf32, 11>
  %5 = comp.alloc %0 : (!comp.execenv<ocl:0,(11)>) -> memref<8x16xf32, 11>
  %k1 = comp.create_kernel %0 {kernel = @main_kernel::@main_kernel} : (!comp.execenv<ocl:0,(11)>) -> !comp.kernel<ocl, 8x32xf32>
  %k2 = comp.create_kernel %0 {kernel = @main_kernel_0::@main_kernel} : (!comp.execenv<ocl:0,(11)>) -> !comp.kernel<ocl, 16x32xf32, 8x16xf32, 8x32xf32>
  comp.loop(%arg1: memref<8x16xf32>, %arg2: memref<16x32xf32>, %arg3: memref<8x32xf32>) {
    %w1 = comp.schedule_write %arg3 to %1 on %0 : (!comp.execenv<ocl:0,(11)>, memref<8x32xf32>, memref<8x32xf32, 11>) -> !comp.event<ocl>
    %2 = comp.schedule_kernel %0, %k1, grid(%c4, %c1, %c1, %c1, %c2, %c32), args(%1), events(%w1): (!comp.kernel<ocl, 8x32xf32>, index, index, index, index, index, index, memref<8x32xf32, 11>, !comp.event<ocl>) -> !comp.event<ocl>
    %w4 = comp.schedule_write %arg2 to %4 on %0 : (!comp.execenv<ocl:0,(11)>, memref<16x32xf32>, memref<16x32xf32, 11>) -> !comp.event<ocl>
    %w5 = comp.schedule_write %arg1 to %5 on %0 : (!comp.execenv<ocl:0,(11)>, memref<8x16xf32>, memref<8x16xf32, 11>) -> !comp.event<ocl>
    %7 = comp.schedule_kernel %0, %k2, grid(%c1, %c1, %c1, %c8, %c4, %c1), args(%4, %5, %1) events(%2, %w4, %w5): (!comp.kernel<ocl, 16x32xf32, 8x16xf32, 8x32xf32>, index, index, index, index, index, index, memref<16x32xf32, 11>, memref<8x16xf32, 11>, memref<8x32xf32, 11>, !comp.event<ocl>, !comp.event<ocl>, !comp.event<ocl>) -> !comp.event<ocl>
    %10 = comp.schedule_read %arg3 from %1 on %0 wait for %7 : (memref<8x32xf32>, memref<8x32xf32, 11>, !comp.execenv<ocl:0,(11)>, !comp.event<ocl>) -> !comp.event<ocl>
    comp.wait %10 : !comp.event<ocl>
  }
  comp.destroy_kernel %k1 : !comp.kernel<ocl, 8x32xf32>
  comp.destroy_kernel %k2 : !comp.kernel<ocl, 16x32xf32, 8x16xf32, 8x32xf32>
  comp.dealloc %0 %4 : (!comp.execenv<ocl:0,(11)>, memref<16x32xf32, 11>) -> ()
  comp.dealloc %0 %5 : (!comp.execenv<ocl:0,(11)>, memref<8x16xf32, 11>) -> ()
  comp.dealloc %0 %1 : (!comp.execenv<ocl:0,(11)>, memref<8x32xf32, 11>) -> ()
  comp.destroy_execenv %0 : !comp.execenv<ocl:0,(11)>
  return
}
```

### Final Lowering

With a minor tweak, we could run the above code as a coroutine&mdash;we could pass in an extra parameter to represent
a bidirectional request/response pipe, and pass it to `comp.loop`.  Our sense is that it's more obvious to split the
current `main()` function into multiple functions invoked individually to set up the network, run it, and tear it down.

For our example code, this looks like:

```mlir
func @plaidml_init(%arg0: !comp.device) -> (!comp.execenv<ocl:0,(11)>, memref<8x32xf32, 11>, memref<16x32xf32, 11>, memref<8x16xf32, 11>, !comp.kernel<ocl, 8x32xf32>, !comp.kernel<ocl, 16x32xf32, 8x16xf32, 8x32xf32>) {
  %0 = comp.create_execenv %arg0 : (!comp.device) -> !comp.execenv<ocl:0,(11)>
  %1 = comp.alloc %0 : (!comp.execenv<ocl:0,(11)>) -> memref<8x32xf32, 11>
  %4 = comp.alloc %0 : (!comp.execenv<ocl:0,(11)>) -> memref<16x32xf32, 11>
  %5 = comp.alloc %0 : (!comp.execenv<ocl:0,(11)>) -> memref<8x16xf32, 11>
  %k1 = comp.create_kernel %0 {kernel = @main_kernel::@main_kernel} : (!comp.execenv<ocl:0,(11)>) -> !comp.kernel<ocl, 8x32xf32>
  %k2 = comp.create_kernel %0 {kernel = @main_kernel_0::@main_kernel} : (!comp.execenv<ocl:0,(11)>) -> !comp.kernel<ocl, 16x32xf32, 8x16xf32, 8x32xf32>
  return %0, %1, %4, %5, %k1, %k2
}

func @plaidml_exec(%0: !comp.execenv<ocl:0,(11)>, %1: memref<8x32xf32, 11>, %4: memref<16x32xf32, 11>, %5: memref<8x16xf32, 11>, %k1: !comp.kernel<ocl, 8x32xf32>, %k2: !comp.kernel<ocl, 16x32xf32, 8x16xf32, 8x32xf32>, %arg1: memref<8x16xf32>, %arg2: memref<16x32xf32>, %arg3: memref<8x32xf32>) {
  %c2 = constant 2 : index
  %c32 = constant 32 : index
  %c1 = constant 1 : index
  %c8 = constant 8 : index
  %c4 = constant 4 : index
  %w1 = comp.schedule_write %arg3 to %1 on %0 : (!comp.execenv<ocl:0,(11)>, memref<8x32xf32>, memref<8x32xf32, 11>) -> !comp.event<ocl>
  %2 = comp.schedule_kernel %0, %k1, grid(%c4, %c1, %c1, %c1, %c2, %c32), args(%1), events(%w1): (!comp.kernel<ocl, 8x32xf32>, index, index, index, index, index, index, memref<8x32xf32, 11>, !comp.event<ocl>) -> !comp.event<ocl>
  %w4 = comp.schedule_write %arg2 to %4 on %0 : (!comp.execenv<ocl:0,(11)>, memref<16x32xf32>, memref<16x32xf32, 11>) -> !comp.event<ocl>
  %w5 = comp.schedule_write %arg1 to %5 on %0 : (!comp.execenv<ocl:0,(11)>, memref<8x16xf32>, memref<8x16xf32, 11>) -> !comp.event<ocl>
  %7 = comp.schedule_kernel %0, %k2, grid(%c1, %c1, %c1, %c8, %c4, %c1), args(%4, %5, %1), events(%2, %w4, %w5): (!comp.kernel<ocl, 16x32xf32, 8x16xf32, 8x32xf32>, index, index, index, index, index, index, memref<16x32xf32, 11>, memref<8x16xf32, 11>, memref<8x32xf32, 11>, !comp.event<ocl>, !comp.event<ocl>, !comp.event<ocl>) -> !comp.event<ocl>
  %10 = comp.schedule_read %arg3 from %1 on %0 wait for %7 : (memref<8x32xf32>, memref<8x32xf32, 11>, !comp.execenv<ocl:0,(11)>, !comp.event<ocl>) -> !comp.event<ocl>
  comp.wait %10 : !comp.event<ocl>
  return
}

func @plaidml_fini(%0: !comp.execenv<ocl:0,(11)>, %1: memref<8x32xf32, 11>, %4: memref<16x32xf32, 11>, %5: memref<8x16xf32, 11>, %k1: !comp.kernel<ocl, 8x32xf32>, %k2: !comp.kernel<ocl, 16x32xf32, 8x16xf32, 8x32xf32>) {
  memref<8x16xf32, 11>, !comp.kernel<ocl, 8x32xf32>, !comp.kernel<ocl, 16x32xf32, 8x16xf32, 8x32xf32>
  comp.destroy_kernel %k1 : !comp.kernel<ocl, 8x32xf32>
  comp.destroy_kernel %k2 : !comp.kernel<ocl, 16x32xf32, 8x16xf32, 8x32xf32>
  comp.dealloc %0 %4 : (!comp.execenv<ocl:0,(11)>, memref<16x32xf32, 11>) -> ()
  comp.dealloc %0 %5 : (!comp.execenv<ocl:0,(11)>, memref<8x16xf32, 11>) -> ()
  comp.dealloc %0 %1 : (!comp.execenv<ocl:0,(11)>, memref<8x32xf32, 11>) -> ()
  comp.destroy_execenv %0 : !comp.execenv<ocl:0,(11)>
  return
}
```

[TODO: Discuss this more.  The coroutine style is actually kind of elegant, and may provide additional opportunities for
optimization.]

### Pass Structure

Because the function split is required in order to match the ABI expected by the caller, it's not optional, making it a
lowering pass.  Because the function split depends on the addition of the `comp.loop` instruction, adding it is also a
lowering pass.

Adding the `comp.loop` instruction is simple enough that we propose to do it in the initial lowering to the comp
dialect, so that comp programs never appear without it.

In order to more easily connect the results of `plaidml_init` with the arguments of `plaidml_exec` and `plaidml_fini`,
we've decided to do the split as part of the lowering to LLVMIR.

The transformation of `gpu.launch_func` is also required, so we'll perform it as part of the initial lowering to the
comp dialect.

The actual hoisting is just an optimization.  We believe we can accomplish this with the standard
`mlir::LoopInvariantCodeMotion` pass.  There's some research to be done around whether we can ensure that we're hoisting
alloc/dealloc pairs as pairs (&Implies; only hoisting one if we're also hoisting the other); if that turns out to be tricky, we
may switch to implicit deallocation.

Wait coalescing is also just an optimization, to be performed by its own pass.

## Notes

  * All of this code was hand-written; please excuse any errors that may have crept in.

  * We're currently assuming that the generated code will assume that newly-allocated buffers have no particular
    contents.  If this assumption proves to be incorrect (&Implies; if the generated code is expecting zeroed
    allocations), we may need to insert zeroing calls.  This is probably best accomplished with a new `comp.zero`
    operation, which can be optimized as it's lowered (e.g. to `clEnqueueFillBuffer()`).

Separately, we may implement a few more bits of functionality:

  * In the past, we've had users suggest that they might want to integrate PlaidML networks as part of bigger processing
    pipelines&mdash;passing in `cl_event`s that the network should wait on, and returning a `cl_event` to indicate that
    the network is complete (or perhaps a `cl_event` per output buffer).  It's fairly trivial to implement this by
    adding event arguments and results to `plaidml_exec`, if it turns out to be needed for current use cases.

  * We considered adding constant folding to the current proposal, but decided to deal with that separately.

      * Adding constant folding is fairly straightforward: we can move constant buffers from being execution-phase
        parameters to being setup-phase parameters, and then move operations&mdash;including kernel launches&mdash;that
        don't depend on IO buffers to the setup phase.

      * Note that it's important to be careful about fusing constant-input kernels with non-constant-input kernels; it
        may be useful to forego this optimization of the network runs faster by evaluating the constant-input kernels
        during setup.

  * We considered adding arena allocation to the comp dialect, but decided to defer it for now.

  * We considered implementing memory reuse, but that's probably best deferred until after arena allocation.

  * We considered separating kernel reuse from memory reuse, so that the caller could run multiple instances of a single
    network in parallel (sharing kernel code), but decided to defer it for now.

  * It's probably also worth moving host-side allocations into the execution environment&mdash;most of these should be
    elided, but we may need to preserve some spill/fill operations to fit within device memory constraints.
