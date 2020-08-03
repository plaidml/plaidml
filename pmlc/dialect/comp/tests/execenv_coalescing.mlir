// RUN: pmlc-opt -pmlc-execenv-coalescing %s | FileCheck %s

module attributes {gpu.container_module, spv.target_env = #spv.target_env<#spv.vce<v1.0, [Shader, Int64, Int16, Int8, Float64, Float16], [SPV_KHR_storage_buffer_storage_class]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  func @dot(%arg0: memref<16x32xf32>, %arg1: memref<8x16xf32>, %arg2: memref<8x32xf32>) {
    %c8 = constant 8 : index
    %c32 = constant 32 : index
    %c1 = constant 1 : index
    %0 = comp.create_execenv : !comp.execenv<vk:0,(11)>
    %1 = comp.alloc %0 %arg2 : (!comp.execenv<vk:0,(11)>, memref<8x32xf32>) -> memref<8x32xf32, 11>
    %2 = "comp.schedule_func"(%0) ( {
      "gpu.launch_func"(%c8, %c1, %c1, %c32, %c1, %c1, %1) {kernel = @dot_kernel::@dot_kernel} : (index, index, index, index, index, index, memref<8x32xf32, 11>) -> ()
      "comp.schedule_end"() : () -> ()
    }) : (!comp.execenv<vk:0,(11)>) -> !comp.event<vk>
    %3 = comp.schedule_read %arg2 from %1 on %0 wait for %2 : (memref<8x32xf32>, memref<8x32xf32, 11>, !comp.execenv<vk:0,(11)>, !comp.event<vk>) -> !comp.event<vk>
    comp.wait %3 : !comp.event<vk>
    comp.dealloc %1 : (memref<8x32xf32, 11>) -> ()
    comp.destroy_execenv %0 : !comp.execenv<vk:0,(11)>
    %4 = comp.create_execenv : !comp.execenv<vk:0,(11)>
    %5 = comp.alloc %4 %arg1 : (!comp.execenv<vk:0,(11)>, memref<8x16xf32>) -> memref<8x16xf32, 11>
    %6 = comp.alloc %4 %arg0 : (!comp.execenv<vk:0,(11)>, memref<16x32xf32>) -> memref<16x32xf32, 11>
    %7 = comp.alloc %4 %arg2 : (!comp.execenv<vk:0,(11)>, memref<8x32xf32>) -> memref<8x32xf32, 11>
    %8 = "comp.schedule_func"(%4) ( {
      "gpu.launch_func"(%c8, %c1, %c1, %c32, %c1, %c1, %5, %6, %7) {kernel = @dot_kernel_0::@dot_kernel} : (index, index, index, index, index, index, memref<8x16xf32, 11>, memref<16x32xf32, 11>, memref<8x32xf32, 11>) -> ()
      "comp.schedule_end"() : () -> ()
    }) : (!comp.execenv<vk:0,(11)>) -> !comp.event<vk>
    %9 = comp.schedule_read %arg1 from %5 on %4 wait for %8 : (memref<8x16xf32>, memref<8x16xf32, 11>, !comp.execenv<vk:0,(11)>, !comp.event<vk>) -> !comp.event<vk>
    %10 = comp.schedule_read %arg0 from %6 on %4 wait for %8 : (memref<16x32xf32>, memref<16x32xf32, 11>, !comp.execenv<vk:0,(11)>, !comp.event<vk>) -> !comp.event<vk>
    %11 = comp.schedule_read %arg2 from %7 on %4 wait for %8 : (memref<8x32xf32>, memref<8x32xf32, 11>, !comp.execenv<vk:0,(11)>, !comp.event<vk>) -> !comp.event<vk>
    %12 = comp.group_events %9, %10, %11 : (!comp.event<vk>, !comp.event<vk>, !comp.event<vk>) -> !comp.event<vk>
    comp.wait %12 : !comp.event<vk>
    comp.dealloc %5 : (memref<8x16xf32, 11>) -> ()
    comp.dealloc %6 : (memref<16x32xf32, 11>) -> ()
    comp.dealloc %7 : (memref<8x32xf32, 11>) -> ()
    comp.destroy_execenv %4 : !comp.execenv<vk:0,(11)>
    return
    // CHECK: comp.create_execenv
    // CHECK: comp.schedule_func
    // CHECK-NOT: comp.create_execenv
    // CHECK: comp.schedule_func
    // CHECK: comp.destroy_execenv
  }
  gpu.module @dot_kernel {
    gpu.func @dot_kernel(%arg0: memref<8x32xf32, 11>) kernel attributes {spv.entry_point_abi = {local_size = dense<[32, 1, 1]> : vector<3xi32>}} {
      %cst = constant 0.000000e+00 : f32
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = "gpu.thread_id"() {dimension = "x"} : () -> index
      store %cst, %arg0[%0, %1] : memref<8x32xf32, 11>
      gpu.return
    }
  }
  gpu.module @dot_kernel_0 {
    gpu.func @dot_kernel(%arg0: memref<8x16xf32, 11>, %arg1: memref<16x32xf32, 11>, %arg2: memref<8x32xf32, 11>) kernel attributes {spv.entry_point_abi = {local_size = dense<[32, 1, 1]> : vector<3xi32>}} {
      %c0 = constant 0 : index
      %c16 = constant 16 : index
      %c1 = constant 1 : index
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = "gpu.thread_id"() {dimension = "x"} : () -> index
      scf.for %arg3 = %c0 to %c16 step %c1 {
        %2 = load %arg0[%0, %arg3] : memref<8x16xf32, 11>
        %3 = load %arg1[%arg3, %1] : memref<16x32xf32, 11>
        %4 = mulf %2, %3 : f32
        %5 = load %arg2[%0, %1] : memref<8x32xf32, 11>
        %6 = addf %5, %4 : f32
        store %6, %arg2[%0, %1] : memref<8x32xf32, 11>
      }
      gpu.return
    }
  }
}
