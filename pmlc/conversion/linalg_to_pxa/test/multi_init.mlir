// RUN: pmlc-opt -cse -convert-linalg-to-pxa %s | FileCheck %s

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func @multi_init() -> tensor<1x112x112x32xf32> {
  %cst = constant 0.00000000 : f32
  %0 = linalg.init_tensor [1, 112, 112, 32] : tensor<1x112x112x32xf32>
  %1 = linalg.generic {
    indexing_maps = [#map0],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } outs(%0 : tensor<1x112x112x32xf32>) {
  ^bb0(%arg0: f32):  // no predecessors
    linalg.yield %cst : f32
  } -> tensor<1x112x112x32xf32>
  %2 = linalg.init_tensor [1, 112, 112, 32] : tensor<1x112x112x32xf32>
  %3 = linalg.generic {
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%1 : tensor<1x112x112x32xf32>) outs(%2 : tensor<1x112x112x32xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
    linalg.yield %arg0 : f32
  } -> tensor<1x112x112x32xf32>
  return %3 : tensor<1x112x112x32xf32>
}

// CHECK-LABEL: func @multi_init
//  CHECK-SAME: (%[[arg0:.*]]: memref<1x112x112x32xf32>) -> memref<1x112x112x32xf32>
//       CHECK:    memref.alloc()
//       CHECK:    %[[buf0:.*]] = memref.alloc() : memref<1x112x112x32xf32>
//       CHECK:    %[[buf1:.*]] = affine.parallel
//       CHECK:      pxa.reduce assign {{.*}}, %[[buf0]]
//       CHECK:      affine.yield
//       CHECK:    %[[buf2:.*]] = affine.parallel
//       CHECK:      %[[tmp:.*]] = pxa.load %[[buf1]]
//       CHECK:      pxa.reduce assign %[[tmp]], %[[arg0]]
//       CHECK:      affine.yield
//       CHECK:    return %[[buf2]]
