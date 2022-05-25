// RUN: pmlc-opt -split-input-file -stdx-split-closure %s | FileCheck %s

func.func @main(%arg0: memref<16x16xf32> {stdx.const}, %arg1: memref<16x16xf32> {stdx.const}) {
  %0 = memref.alloc() : memref<16x16xf32>
  affine.for %arg2 = 0 to 16 {
    affine.for %arg3 = 0 to 16 {
      %1 = affine.load %arg0[%arg2, %arg3] : memref<16x16xf32>
      %2 = affine.load %arg1[%arg2, %arg3] : memref<16x16xf32>
      %3 = arith.addf %1, %2 : f32
      affine.store %3, %0[%arg2, %arg3] : memref<16x16xf32>
    }
  }
  %4 = memref.alloc() : memref<16x16xf32>
  stdx.closure(%arg2: memref<16x16xf32>, %arg3: memref<16x16xf32>) {
    affine.for %arg4 = 0 to 16 {
      affine.for %arg5 = 0 to 16 {
        %1 = affine.load %arg2[%arg4, %arg5] : memref<16x16xf32>
        %2 = affine.load %0[%arg4, %arg5] : memref<16x16xf32>
        %3 = arith.addf %1, %2 : f32
        affine.store %3, %arg3[%arg4, %arg5] : memref<16x16xf32>
      }
    }
    stdx.yield
  }
  memref.dealloc %0 : memref<16x16xf32>
  memref.dealloc %4 : memref<16x16xf32>
  return
}


// CHECK: func.func @init(%[[init_arg0:.*]]: memref<16x16xf32>, %[[init_arg1:.*]]: memref<16x16xf32>) -> tuple<memref<16x16xf32>, memref<16x16xf32>>
// CHECK:   %[[init_tmp0:.*]] = memref.alloc() : memref<16x16xf32>
// CHECK:   affine.for %{{.*}} = 0 to 16
// CHECK:     affine.for %{{.*}} = 0 to 16
// CHECK:       affine.load %[[init_arg0]][%{{.*}}, %{{.*}}] : memref<16x16xf32>
// CHECK:       affine.load %[[init_arg1]][%{{.*}}, %{{.*}}] : memref<16x16xf32>
// CHECK:       addf %{{.*}}, %{{.*}} : f32
// CHECK:       affine.store %{{.*}}, %[[init_tmp0]][%{{.*}}, %{{.*}}] : memref<16x16xf32>
// CHECK:   %[[init_tmp1:.*]] = memref.alloc() : memref<16x16xf32>
// CHECK:   %[[init_tuple:.*]] = stdx.pack(%0, %1) : (memref<16x16xf32>, memref<16x16xf32>) -> tuple<memref<16x16xf32>, memref<16x16xf32>>
// CHECK:   return %[[init_tuple]] : tuple<memref<16x16xf32>, memref<16x16xf32>>

// CHECK: func.func @main(%[[main_arg0:.*]]: tuple<memref<16x16xf32>, memref<16x16xf32>>, %[[main_arg1:.*]]: memref<16x16xf32>, %[[main_arg2:.*]]: memref<16x16xf32>)
// CHECK:   %[[main_tuple:.*]]:2 = stdx.unpack(%[[main_arg0]]) : (tuple<memref<16x16xf32>, memref<16x16xf32>>) -> (memref<16x16xf32>, memref<16x16xf32>)
// CHECK:   affine.for %{{.*}} = 0 to 16
// CHECK:     affine.for %{{.*}} = 0 to 16
// CHECK:       affine.load %[[main_arg1]][%{{.*}}, %{{.*}}] : memref<16x16xf32>
// CHECK:       affine.load %[[main_tuple]]#0[%{{.*}}, %{{.*}}] : memref<16x16xf32>
// CHECK:       addf %{{.*}}, %{{.*}} : f32
// CHECK:       affine.store %{{.*}}, %[[main_arg2]][%{{.*}}, %{{.*}}] : memref<16x16xf32>
// CHECK:   return

// CHECK: func.func @fini(%[[fini_arg0:.*]]: tuple<memref<16x16xf32>, memref<16x16xf32>>)
// CHECK:   %[[fini_tuple:.*]]:2 = stdx.unpack(%[[fini_arg0]]) : (tuple<memref<16x16xf32>, memref<16x16xf32>>) -> (memref<16x16xf32>, memref<16x16xf32>)
// CHECK:   memref.dealloc %[[fini_tuple]]#0 : memref<16x16xf32>
// CHECK:   memref.dealloc %[[fini_tuple]]#1 : memref<16x16xf32>
// CHECK:   return

// -----

func.func @main(%arg0: memref<16x16xf32> {stdx.const}, %arg1: memref<16x16xf32> {stdx.const}) {
  stdx.closure(%arg2: memref<16x16xf32>) {
    affine.for %arg4 = 0 to 16 {
      affine.for %arg5 = 0 to 16 {
        %1 = affine.load %arg0[%arg4, %arg5] : memref<16x16xf32>
        %2 = affine.load %arg1[%arg4, %arg5] : memref<16x16xf32>
        %3 = arith.addf %1, %2 : f32
        affine.store %3, %arg2[%arg4, %arg5] : memref<16x16xf32>
      }
    }
    stdx.yield
  }
  return
}

// CHECK: func.func @init(%[[init_arg0:.*]]: memref<16x16xf32>, %[[init_arg1:.*]]: memref<16x16xf32>) -> tuple<memref<16x16xf32>, memref<16x16xf32>> {
// CHECK:   %[[init_tuple:.*]] = stdx.pack(%[[init_arg0]], %[[init_arg1]]) : (memref<16x16xf32>, memref<16x16xf32>) -> tuple<memref<16x16xf32>, memref<16x16xf32>>
// CHECK:   return %[[init_tuple]] : tuple<memref<16x16xf32>, memref<16x16xf32>>

// CHECK: func.func @main(%[[main_arg0]]: tuple<memref<16x16xf32>, memref<16x16xf32>>, %[[main_arg1:.*]]: memref<16x16xf32>) {
// CHECK:   %[[main_tuple:.*]]:2 = stdx.unpack(%[[main_arg0]]) : (tuple<memref<16x16xf32>, memref<16x16xf32>>) -> (memref<16x16xf32>, memref<16x16xf32>)
// CHECK:   affine.for %arg2 = 0 to 16 {
// CHECK:     affine.for %arg3 = 0 to 16 {
// CHECK:       affine.load %[[main_tuple]]#0[%{{.*}}, %{{.*}}] : memref<16x16xf32>
// CHECK:       affine.load %[[main_tuple]]#1[%{{.*}}, %{{.*}}] : memref<16x16xf32>
// CHECK:       addf %{{.*}}, %{{.*}} : f32
// CHECK:       affine.store %{{.*}}, %[[main_arg1]][%{{.*}}, %{{.*}}] : memref<16x16xf32>
// CHECK:   return

// CHECK: func.func @fini(%[[fini_arg0:.*]]: tuple<memref<16x16xf32>, memref<16x16xf32>>) {
// CHECK:   %[[fini_tuple:.*]]:2 = stdx.unpack(%[[fini_arg0]]) : (tuple<memref<16x16xf32>, memref<16x16xf32>>) -> (memref<16x16xf32>, memref<16x16xf32>)
// CHECK:   return

// -----

func.func private @plaidml_rt_prng(memref<*xi32>, memref<*xf32>, memref<*xi32>)

func.func @main(%arg0: memref<1x3xi32> {stdx.const}) {
  %0 = memref.cast %arg0 : memref<1x3xi32> to memref<*xi32>
  stdx.closure(%arg1: memref<2x3xf32>, %arg2: memref<1x3xi32>) {
    %1 = memref.cast %arg1 : memref<2x3xf32> to memref<*xf32>
    %2 = memref.cast %arg2 : memref<1x3xi32> to memref<*xi32>
    call @plaidml_rt_prng(%0, %1, %2) : (memref<*xi32>, memref<*xf32>, memref<*xi32>) -> ()
    stdx.yield
  }
  return
}

// CHECK: func.func @init(%[[init_arg0:.*]]: memref<1x3xi32>) -> tuple<memref<1x3xi32>>
// CHECK:   %[[init_X0:.*]] = memref.cast %[[init_arg0]] : memref<1x3xi32> to memref<*xi32>
// CHECK:   %[[init_X1:.*]] = stdx.pack(%[[init_arg0]]) : (memref<1x3xi32>) -> tuple<memref<1x3xi32>>
// CHECK:   return %[[init_X1]] : tuple<memref<1x3xi32>>

// CHECK: func.func @main(%[[main_arg0:.*]]: tuple<memref<1x3xi32>>, %[[main_arg1:.*]]: memref<2x3xf32>, %[[main_arg2:.*]]: memref<1x3xi32>)
// CHECK:   %[[main_X0:.*]] = stdx.unpack(%[[main_arg0]]) : (tuple<memref<1x3xi32>>) -> memref<1x3xi32>
// CHECK:   %[[main_X1:.*]] = memref.cast %[[main_X0]] : memref<1x3xi32> to memref<*xi32>
// CHECK:   %[[main_X2:.*]] = memref.cast %[[main_arg1]] : memref<2x3xf32> to memref<*xf32>
// CHECK:   %[[main_X3:.*]] = memref.cast %[[main_arg2]] : memref<1x3xi32> to memref<*xi32>
// CHECK:   call @plaidml_rt_prng(%[[main_X1]], %[[main_X2]], %[[main_X3]]) : (memref<*xi32>, memref<*xf32>, memref<*xi32>) -> ()
// CHECK:   return

// CHECK: func.func @fini(%[[fini_arg0:.*]]: tuple<memref<1x3xi32>>)
// CHECK:   %[[fini_X0:.*]] = stdx.unpack(%[[fini_arg0]]) : (tuple<memref<1x3xi32>>) -> memref<1x3xi32>
// CHECK:   %[[fini_X1:.*]] = memref.cast %[[fini_X0]] : memref<1x3xi32> to memref<*xi32>
// CHECK:   return

// -----

func.func private @print(index)

func.func @main() {
  %c3 = arith.constant 3 : index
  stdx.closure() {
    call @print(%c3) : (index) -> ()
    stdx.yield
  }
  return
}

// CHECK: func.func @init() -> tuple<> {
// CHECK:   %c3 = arith.constant 3 : index
// CHECK:   %0 = stdx.pack() : () -> tuple<>
// CHECK:   return %0 : tuple<>

// CHECK: func.func @main(%[[main_arg0:.*]]: tuple<>) {
// CHECK:   stdx.unpack(%[[main_arg0]]) : (tuple<>) -> ()
// CHECK:   %c3 = arith.constant 3 : index
// CHECK:   call @print(%c3) : (index) -> ()
// CHECK:   return

// CHECK: func.func @fini(%[[fini_arg0:.*]]: tuple<>) {
// CHECK:   stdx.unpack(%[[fini_arg0]]) : (tuple<>) -> ()
// CHECK:   return
// CHECK: }
