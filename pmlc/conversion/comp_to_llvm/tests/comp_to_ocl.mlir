// RUN: pmlc-opt -pmlc-convert-comp-to-ocl --split-input-file %s | FileCheck %s

module {
  // CHECK-LABEL: func @create_destroy
  //  CHECK-SAME:     %[[DEV:.*]]: !llvm.ptr<i8>
  //       CHECK:   %[[ENV:.*]] = llvm.call @oclCreate(%[[DEV]])
  //       CHECK:   llvm.call @oclDestroy(%[[ENV]])
  func @create_destroy(%dev: !comp.device) {
    %env = comp.create_execenv %dev : (!comp.device) -> !comp.execenv<ocl:0,(11)>
    comp.destroy_execenv %env : !comp.execenv<ocl:0,(11)>
    return
  }
}

// -----

module {
  // CHECK-LABEL: func @no_gpu_func
  //  CHECK-SAME:     %[[DEV:.*]]: !llvm.ptr<i8>
  //       CHECK:   %[[ENV:.*]] = llvm.call @oclCreate(%[[DEV]])
  //       CHECK:   %[[CST0:.*]] = llvm.mlir.constant(0
  //       CHECK:   %[[EV:.*]] = llvm.call @oclBarrier(%[[ENV]], %[[CST0]])
  //       CHECK:   llvm.call @oclSubmit(%[[ENV]])
  //       CHECK:   %[[CST1:.*]] = llvm.mlir.constant(1
  //       CHECK:   llvm.call @oclWait(%[[CST1]], %[[EV]])
  //       CHECK:   llvm.call @oclDestroy(%[[ENV]])
  func @no_gpu_func(%dev: !comp.device) {
    %env = comp.create_execenv %dev : (!comp.device) -> !comp.execenv<ocl:0,(11)>
    %ev = comp.schedule_barrier %env : (!comp.execenv<ocl:0,(11)>) -> !comp.event<ocl>
    comp.submit %env : !comp.execenv<ocl:0,(11)>
    comp.wait %ev : !comp.event<ocl>
    comp.destroy_execenv %env : !comp.execenv<ocl:0,(11)>
    return
  }
}

// -----

module {
  // CHECK-LABEL: func @alloc_dealloc
  //  CHECK-SAME:     %[[DEV:.*]]: !llvm.ptr<i8>
  //       CHECK:   %[[ENV:.*]] = llvm.call @oclCreate(%[[DEV]])
  //       CHECK:   %[[CNT:.*]] = llvm.mlir.constant(1024
  //       CHECK:   %[[NUL:.*]] = llvm.mlir.null
  //       CHECK:   %[[MEM:.*]] = llvm.call @oclAlloc(%[[ENV]], %[[CNT]], %[[NUL]])
  //       CHECK:   llvm.call @oclDealloc(%[[ENV]], %[[MEM]])
  //       CHECK:   llvm.call @oclDestroy(%[[ENV]])
  func @alloc_dealloc(%dev: !comp.device) {
    %env = comp.create_execenv %dev : (!comp.device) -> !comp.execenv<ocl:0,(11)>
    %mem = comp.alloc %env : (!comp.execenv<ocl:0,(11)>) -> memref<8x32xf32, 11>
    comp.dealloc %env %mem : (!comp.execenv<ocl:0,(11)>, memref<8x32xf32, 11>) -> ()
    comp.destroy_execenv %env : !comp.execenv<ocl:0,(11)>
    return
  }
}

// -----

module attributes {gpu.container_module} {
  // CHECK-LABEL: func @one_gpu_func
  //  CHECK-SAME:     %[[DEV:[a-zA-Z0-9]*]]: !llvm.ptr<i8>
  //  CHECK-SAME:     %{{[a-zA-Z0-9]*}}: memref<8x32xf32>
  //       CHECK:   %[[ENV:.*]] = llvm.call @oclCreate(%[[DEV]])
  //       CHECK:   %[[CNT:.*]] = llvm.mlir.constant(1024
  //       CHECK:   %[[MEM:.*]] = llvm.call @oclAlloc(%[[ENV]], %[[CNT]], %{{.*}})
  //       CHECK:   %[[CST0:.*]] = llvm.mlir.constant(0
  //       CHECK:   %[[WEV:.*]] = llvm.call @oclWrite(%{{.*}}, %[[MEM]], %[[ENV]], %[[CST0]])
  //       CHECK:   %[[KRNL:.*]] = llvm.call @oclCreateKernel(%[[ENV]]
  //       CHECK:   llvm.call @oclSetKernelArg(%[[KRNL]], {{.*}}, %[[MEM]])
  //       CHECK:   llvm.call @oclAddKernelDep(%[[KRNL]], %[[WEV]])
  //       CHECK:   %[[FEV:.*]] = call @oclScheduleFunc(%[[ENV]], %[[KRNL]]
  //       CHECK:   %[[CST1:.*]] = llvm.mlir.constant(1
  //       CHECK:   %[[REV:.*]] = llvm.call @oclRead(%{{.*}}, %[[MEM]], %[[ENV]], %[[CST1]], %[[FEV]])
  //       CHECK:   %[[CST1:.*]] = llvm.mlir.constant(1
  //       CHECK:   llvm.call @oclWait(%[[CST1]], %[[REV]])
  //       CHECK:   llvm.call @oclDealloc(%[[ENV]], %[[MEM]])
  //       CHECK:   llvm.call @oclDestroy(%[[ENV]])
  func @one_gpu_func(%dev: !comp.device, %arg0: memref<8x32xf32>) {
    %c8 = constant 8 : index
    %c32 = constant 32 : index
    %c1 = constant 1 : index
    %env = comp.create_execenv %dev : (!comp.device) -> !comp.execenv<ocl:0,(11)>
    %mem0 = comp.alloc %env %arg0 : (!comp.execenv<ocl:0,(11)>, memref<8x32xf32>) -> memref<8x32xf32, 11>
    %wev = comp.schedule_write %arg0 to %mem0 on %env : (memref<8x32xf32>, memref<8x32xf32, 11>, !comp.execenv<ocl:0,(11)>) -> !comp.event<ocl>
    %fev = "comp.schedule_func"(%env, %wev) ( {
      "gpu.launch_func"(%c8, %c1, %c1, %c32, %c1, %c1, %mem0) {kernel = @gpu_module::@zero} : (index, index, index, index, index, index, memref<8x32xf32, 11>) -> ()
      "comp.schedule_end"() : () -> ()
    }) : (!comp.execenv<ocl:0,(11)>, !comp.event<ocl>) -> !comp.event<ocl>
    %rev = comp.schedule_read %arg0 from %mem0 on %env wait for %fev : (memref<8x32xf32>, memref<8x32xf32, 11>, !comp.execenv<ocl:0,(11)>, !comp.event<ocl>) -> !comp.event<ocl>
    comp.wait %rev : !comp.event<ocl>
    comp.dealloc %env %mem0 : (!comp.execenv<ocl:0,(11)>, memref<8x32xf32, 11>) -> ()
    comp.destroy_execenv %env : !comp.execenv<ocl:0,(11)>
    return
  }
  // CHECK-NOT: spv.module
  // CHECK-NOT: gpu.module
  spv.module Physical64 OpenCL requires #spv.vce<v1.0, [Addresses, Kernel], []> {
    spv.globalVariable @__builtin_var_LocalInvocationId__ built_in("LocalInvocationId") : !spv.ptr<vector<3xi32>, Input>
    spv.globalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spv.ptr<vector<3xi32>, Input>
    spv.func @zero(%arg0: !spv.ptr<!spv.struct<!spv.array<256 x f32, stride=4> [0]>, CrossWorkgroup>) "None" attributes {spv.entry_point_abi = {local_size = dense<[32, 1, 1]> : vector<3xi32>}, workgroup_attributions = 0 : i64} {
      %0 = spv.constant 0.000000e+00 : f32
      %1 = spv._address_of @__builtin_var_WorkgroupId__ : !spv.ptr<vector<3xi32>, Input>
      %2 = spv.Load "Input" %1 : vector<3xi32>
      %3 = spv.CompositeExtract %2[0 : i32] : vector<3xi32>
      %4 = spv._address_of @__builtin_var_LocalInvocationId__ : !spv.ptr<vector<3xi32>, Input>
      %5 = spv.Load "Input" %4 : vector<3xi32>
      %6 = spv.CompositeExtract %5[0 : i32] : vector<3xi32>
      %7 = spv.constant 0 : i32
      %8 = spv.constant 0 : i32
      %9 = spv.constant 32 : i32
      %10 = spv.IMul %9, %3 : i32
      %11 = spv.IAdd %8, %10 : i32
      %12 = spv.constant 1 : i32
      %13 = spv.IMul %12, %6 : i32
      %14 = spv.IAdd %11, %13 : i32
      %15 = spv.AccessChain %arg0[%7, %14] : !spv.ptr<!spv.struct<!spv.array<256 x f32, stride=4> [0]>, CrossWorkgroup>, i32, i32
      spv.Store "CrossWorkgroup" %15, %0 : f32
      spv.Return
    }
  }
  gpu.module @gpu_module {
    gpu.func @zero(%arg0: memref<8x32xf32, 11>) kernel attributes {spv.entry_point_abi = {local_size = dense<[32, 1, 1]> : vector<3xi32>}} {
      %cst = constant 0.000000e+00 : f32
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = "gpu.thread_id"() {dimension = "x"} : () -> index
      store %cst, %arg0[%0, %1] : memref<8x32xf32, 11>
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  // CHECK-LABEL: func @one_gpu_func
  //  CHECK-SAME:     %[[DEV:[a-zA-Z0-9]*]]: !llvm.ptr<i8>
  //  CHECK-SAME:     %{{[a-zA-Z0-9]*}}: memref<8x32xf16>
  //       CHECK:   %[[ENV:.*]] = llvm.call @oclCreate(%[[DEV]])
  //       CHECK:   %[[CNT:.*]] = llvm.mlir.constant(512
  //       CHECK:   %[[MEM:.*]] = llvm.call @oclAlloc(%[[ENV]], %[[CNT]], %{{.*}})
  //       CHECK:   %[[CST0:.*]] = llvm.mlir.constant(0
  //       CHECK:   %[[WEV:.*]] = llvm.call @oclWrite(%{{.*}}, %[[MEM]], %[[ENV]], %[[CST0]])
  //       CHECK:   %[[KRNL:.*]] = llvm.call @oclCreateKernel(%[[ENV]]
  //       CHECK:   llvm.call @oclSetKernelArg(%[[KRNL]], {{.*}}, %[[MEM]])
  //       CHECK:   llvm.call @oclAddKernelDep(%[[KRNL]], %[[WEV]])
  //       CHECK:   %[[FEV:.*]] = call @oclScheduleFunc(%[[ENV]], %[[KRNL]]
  //       CHECK:   %[[CST1:.*]] = llvm.mlir.constant(1
  //       CHECK:   %[[REV:.*]] = llvm.call @oclRead(%{{.*}}, %[[MEM]], %[[ENV]], %[[CST1]], %[[FEV]])
  //       CHECK:   %[[CST1:.*]] = llvm.mlir.constant(1
  //       CHECK:   llvm.call @oclWait(%[[CST1]], %[[REV]])
  //       CHECK:   llvm.call @oclDealloc(%[[ENV]], %[[MEM]])
  //       CHECK:   llvm.call @oclDestroy(%[[ENV]])
  func @one_gpu_func(%dev: !comp.device, %arg0: memref<8x32xf16>) {
    %c8 = constant 8 : index
    %c32 = constant 32 : index
    %c1 = constant 1 : index
    %env = comp.create_execenv %dev : (!comp.device) -> !comp.execenv<ocl:0,(11)>
    %mem0 = comp.alloc %env %arg0 : (!comp.execenv<ocl:0,(11)>, memref<8x32xf16>) -> memref<8x32xf16, 11>
    %wev = comp.schedule_write %arg0 to %mem0 on %env : (memref<8x32xf16>, memref<8x32xf16, 11>, !comp.execenv<ocl:0,(11)>) -> !comp.event<ocl>
    %fev = "comp.schedule_func"(%env, %wev) ( {
      "gpu.launch_func"(%c8, %c1, %c1, %c32, %c1, %c1, %mem0) {kernel = @gpu_module::@zero} : (index, index, index, index, index, index, memref<8x32xf16, 11>) -> ()
      "comp.schedule_end"() : () -> ()
    }) : (!comp.execenv<ocl:0,(11)>, !comp.event<ocl>) -> !comp.event<ocl>
    %rev = comp.schedule_read %arg0 from %mem0 on %env wait for %fev : (memref<8x32xf16>, memref<8x32xf16, 11>, !comp.execenv<ocl:0,(11)>, !comp.event<ocl>) -> !comp.event<ocl>
    comp.wait %rev : !comp.event<ocl>
    comp.dealloc %env %mem0 : (!comp.execenv<ocl:0,(11)>, memref<8x32xf16, 11>) -> ()
    comp.destroy_execenv %env : !comp.execenv<ocl:0,(11)>
    return
  }
  // CHECK-NOT: spv.module
  // CHECK-NOT: gpu.module
  spv.module Physical64 OpenCL requires #spv.vce<v1.0, [Addresses, Kernel], []> {
    spv.globalVariable @__builtin_var_LocalInvocationId__ built_in("LocalInvocationId") : !spv.ptr<vector<3xi32>, Input>
    spv.globalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spv.ptr<vector<3xi32>, Input>
    spv.func @zero(%arg0: !spv.ptr<!spv.struct<!spv.array<256 x f16, stride=4> [0]>, CrossWorkgroup>) "None" attributes {spv.entry_point_abi = {local_size = dense<[32, 1, 1]> : vector<3xi32>}, workgroup_attributions = 0 : i64} {
      %0 = spv.constant 0.000000e+00 : f16
      %1 = spv._address_of @__builtin_var_WorkgroupId__ : !spv.ptr<vector<3xi32>, Input>
      %2 = spv.Load "Input" %1 : vector<3xi32>
      %3 = spv.CompositeExtract %2[0 : i32] : vector<3xi32>
      %4 = spv._address_of @__builtin_var_LocalInvocationId__ : !spv.ptr<vector<3xi32>, Input>
      %5 = spv.Load "Input" %4 : vector<3xi32>
      %6 = spv.CompositeExtract %5[0 : i32] : vector<3xi32>
      %7 = spv.constant 0 : i32
      %8 = spv.constant 0 : i32
      %9 = spv.constant 32 : i32
      %10 = spv.IMul %9, %3 : i32
      %11 = spv.IAdd %8, %10 : i32
      %12 = spv.constant 1 : i32
      %13 = spv.IMul %12, %6 : i32
      %14 = spv.IAdd %11, %13 : i32
      %15 = spv.AccessChain %arg0[%7, %14] : !spv.ptr<!spv.struct<!spv.array<256 x f16, stride=4> [0]>, CrossWorkgroup>, i32, i32
      spv.Store "CrossWorkgroup" %15, %0 : f16
      spv.Return
    }
  }
  gpu.module @gpu_module {
    gpu.func @zero(%arg0: memref<8x32xf16, 11>) kernel attributes {spv.entry_point_abi = {local_size = dense<[32, 1, 1]> : vector<3xi32>}} {
      %cst = constant 0.000000e+00 : f16
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = "gpu.thread_id"() {dimension = "x"} : () -> index
      store %cst, %arg0[%0, %1] : memref<8x32xf16, 11>
      gpu.return
    }
  }
}
