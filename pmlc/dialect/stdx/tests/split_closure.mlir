// RUN: pmlc-opt -split-input-file -stdx-split-closure %s | FileCheck %s

func @main(%arg0: memref<16x16xf32> {stdx.const = 0 : index}, %arg1: memref<16x16xf32> {stdx.const = 1 : index}) {
  %0 = memref.alloc() : memref<16x16xf32>
  affine.for %arg2 = 0 to 16 {
    affine.for %arg3 = 0 to 16 {
      %1 = affine.load %arg0[%arg2, %arg3] : memref<16x16xf32>
      %2 = affine.load %arg1[%arg2, %arg3] : memref<16x16xf32>
      %3 = addf %1, %2 : f32
      affine.store %3, %0[%arg2, %arg3] : memref<16x16xf32>
    }
  }
  %4 = memref.alloc() : memref<16x16xf32>
  stdx.closure(%arg2: memref<16x16xf32>, %arg3: memref<16x16xf32>) {
    affine.for %arg4 = 0 to 16 {
      affine.for %arg5 = 0 to 16 {
        %1 = affine.load %arg2[%arg4, %arg5] : memref<16x16xf32>
        %2 = affine.load %0[%arg4, %arg5] : memref<16x16xf32>
        %3 = addf %1, %2 : f32
        affine.store %3, %arg3[%arg4, %arg5] : memref<16x16xf32>
      }
    }
    stdx.yield
  }
  memref.dealloc %0 : memref<16x16xf32>
  memref.dealloc %4 : memref<16x16xf32>
  return
}


// CHECK: func @init(%[[init_arg0:.*]]: memref<16x16xf32>, %[[init_arg1:.*]]: memref<16x16xf32>) -> tuple<memref<16x16xf32>, memref<16x16xf32>>
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

// CHECK: func @main(%[[main_arg0:.*]]: tuple<memref<16x16xf32>, memref<16x16xf32>>, %[[main_arg1:.*]]: memref<16x16xf32>, %[[main_arg2:.*]]: memref<16x16xf32>)
// CHECK:   %[[main_tuple:.*]]:2 = stdx.unpack(%[[main_arg0]]) : (tuple<memref<16x16xf32>, memref<16x16xf32>>) -> (memref<16x16xf32>, memref<16x16xf32>)
// CHECK:   affine.for %{{.*}} = 0 to 16
// CHECK:     affine.for %{{.*}} = 0 to 16
// CHECK:       affine.load %[[main_arg1]][%{{.*}}, %{{.*}}] : memref<16x16xf32>
// CHECK:       affine.load %[[main_tuple]]#0[%{{.*}}, %{{.*}}] : memref<16x16xf32>
// CHECK:       addf %{{.*}}, %{{.*}} : f32
// CHECK:       affine.store %{{.*}}, %[[main_arg2]][%{{.*}}, %{{.*}}] : memref<16x16xf32>
// CHECK:   return

// CHECK: func @fini(%[[fini_arg0:.*]]: tuple<memref<16x16xf32>, memref<16x16xf32>>)
// CHECK:   %[[fini_tuple:.*]]:2 = stdx.unpack(%[[fini_arg0]]) : (tuple<memref<16x16xf32>, memref<16x16xf32>>) -> (memref<16x16xf32>, memref<16x16xf32>)
// CHECK:   memref.dealloc %[[fini_tuple]]#0 : memref<16x16xf32>
// CHECK:   memref.dealloc %[[fini_tuple]]#1 : memref<16x16xf32>
// CHECK:   return

// -----

func @main(%arg0: memref<16x16xf32> {stdx.const = 0 : index}, %arg1: memref<16x16xf32> {stdx.const = 1 : index}) {
  stdx.closure(%arg2: memref<16x16xf32>) {
    affine.for %arg4 = 0 to 16 {
      affine.for %arg5 = 0 to 16 {
        %1 = affine.load %arg0[%arg4, %arg5] : memref<16x16xf32>
        %2 = affine.load %arg1[%arg4, %arg5] : memref<16x16xf32>
        %3 = addf %1, %2 : f32
        affine.store %3, %arg2[%arg4, %arg5] : memref<16x16xf32>
      }
    }
    stdx.yield
  }
  return
}

// CHECK: func @init(%[[init_arg0:.*]]: memref<16x16xf32>, %[[init_arg1:.*]]: memref<16x16xf32>) -> tuple<memref<16x16xf32>, memref<16x16xf32>> {
// CHECK:   %[[init_tuple:.*]] = stdx.pack(%[[init_arg0]], %[[init_arg1]]) : (memref<16x16xf32>, memref<16x16xf32>) -> tuple<memref<16x16xf32>, memref<16x16xf32>>
// CHECK:   return %[[init_tuple]] : tuple<memref<16x16xf32>, memref<16x16xf32>>

// CHECK: func @main(%[[main_arg0]]: tuple<memref<16x16xf32>, memref<16x16xf32>>, %[[main_arg1:.*]]: memref<16x16xf32>) {
// CHECK:   %[[main_tuple:.*]]:2 = stdx.unpack(%[[main_arg0]]) : (tuple<memref<16x16xf32>, memref<16x16xf32>>) -> (memref<16x16xf32>, memref<16x16xf32>)
// CHECK:   affine.for %arg2 = 0 to 16 {
// CHECK:     affine.for %arg3 = 0 to 16 {
// CHECK:       affine.load %[[main_tuple]]#0[%{{.*}}, %{{.*}}] : memref<16x16xf32>
// CHECK:       affine.load %[[main_tuple]]#1[%{{.*}}, %{{.*}}] : memref<16x16xf32>
// CHECK:       addf %{{.*}}, %{{.*}} : f32
// CHECK:       affine.store %{{.*}}, %[[main_arg1]][%{{.*}}, %{{.*}}] : memref<16x16xf32>
// CHECK:   return

// CHECK: func @fini(%[[fini_arg0:.*]]: tuple<memref<16x16xf32>, memref<16x16xf32>>) {
// CHECK:   %[[fini_tuple:.*]]:2 = stdx.unpack(%[[fini_arg0]]) : (tuple<memref<16x16xf32>, memref<16x16xf32>>) -> (memref<16x16xf32>, memref<16x16xf32>)
// CHECK:   return
