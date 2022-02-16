// RUN: pmlc-opt -convert-linalg-to-pxa %s | FileCheck %s

#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

func @closure() {
  %cst = constant 0.0 : f32
  %0 = linalg.init_tensor [8, 32] : tensor<8x32xf32>
  %1 = linalg.fill(%cst, %0) : f32, tensor<8x32xf32> -> tensor<8x32xf32>
  stdx.closure(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>)
      -> tensor<8x32xf32> {
    %2 = linalg.generic {
      indexing_maps = [#map0, #map1, #map2],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%arg0, %arg1 : tensor<8x16xf32>, tensor<16x32xf32>)
      outs(%1 : tensor<8x32xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
      %3 = mulf %arg2, %arg3 : f32
      %4 = addf %arg4, %3 : f32
      linalg.yield %4 : f32
    } -> tensor<8x32xf32>
    stdx.yield %2 : tensor<8x32xf32>
  }
  return
}

// CHECK-LABEL: func @closure()
//       CHECK:   memref.alloc
//       CHECK:   memref.alloc
//       CHECK:   %[[fill:.*]] = affine.parallel
//       CHECK:   stdx.closure(
//  CHECK-SAME:     %[[arg0:.*]]: memref<8x16xf32>
//  CHECK-SAME:     %[[arg1:.*]]: memref<16x32xf32>
//  CHECK-SAME:     %[[arg2:.*]]: memref<8x32xf32>
//       CHECK:     memref.alloc
//       CHECK:     %[[copy:.*]] = affine.parallel
//       CHECK:       pxa.load %[[fill]]
//       CHECK:       pxa.reduce assign %{{.*}}, %[[arg2]]
//       CHECK:     affine.parallel
//       CHECK:       pxa.load
//       CHECK:       pxa.load
//       CHECK:       mulf
//       CHECK:       pxa.reduce addf %{{.*}}, %[[copy]]
//       CHECK:     stdx.yield
//       CHEC:    return


#map = affine_map<(d0, d1) -> (d0, d1)>

func @closure_yield(%arg0: tensor<3x3xf32> {stdx.const}) {
  %0 = linalg.init_tensor [3, 3] : tensor<3x3xf32>
  stdx.closure() -> tensor<3x3xf32> {
    %1 = linalg.generic {indexing_maps = [#map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%arg0 : tensor<3x3xf32>) outs(%0 : tensor<3x3xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
      %2 = stdx.relu(%arg1) : (f32) -> f32
      linalg.yield %2 : f32
    } -> tensor<3x3xf32>
    stdx.yield %1 : tensor<3x3xf32>
  }
  return
}

// CHECK-LABEL: func @closure_yield
//  CHECK-SAME: (%[[arg0:.*]]: memref<3x3xf32> {stdx.const})
//       CHECK:   stdx.closure(%[[arg1:.*]]: memref<3x3xf32>)
//       CHECK:     %[[t0:.*]] = affine.parallel
//       CHECK:       pxa.load %[[arg0]]
//       CHECK:       stdx.relu
//       CHECK:       pxa.reduce assign {{.*}}, %[[arg1]]
//       CHECK:       affine.yield
//       CHECK:     stdx.yield %[[t0]]
//       CHECK:   return
