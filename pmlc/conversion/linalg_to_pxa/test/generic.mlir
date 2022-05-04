// RUN: pmlc-opt -convert-linalg-to-pxa -cse %s | FileCheck %s

#map0 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>

func @test_conv_2d_input_nhwc_filter_hwcf(%arg0: tensor<1x255x255x3xf32>, %arg1: tensor<3x3x3x32xf32>) -> tensor<1x112x112x32xf32> {
  %0 = linalg.init_tensor [1, 112, 112, 32] : tensor<1x112x112x32xf32>
  %1 = linalg.generic {
    indexing_maps = [#map0, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "parallel"]
  } ins(%arg0, %arg1 : tensor<1x255x255x3xf32>, tensor<3x3x3x32xf32>) outs(%0 : tensor<1x112x112x32xf32>) {
  ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
    %2 = arith.mulf %arg2, %arg3 : f32
    %3 = arith.addf %arg4, %2 : f32
    linalg.yield %3 : f32
  } -> tensor<1x112x112x32xf32>
  return %1 : tensor<1x112x112x32xf32>
}

// CHECK-LABEL: func @test_conv_2d_input_nhwc_filter_hwcf
//  CHECK-SAME: %[[arg0:.*]]: memref<1x255x255x3xf32>, %[[arg1:.*]]: memref<3x3x3x32xf32>, %[[arg2:.*]]: memref<1x112x112x32xf32>
//  CHECK-SAME: -> memref<1x112x112x32xf32>
//       CHECK:   %[[buf1:.*]] = affine.parallel (%[[arg3:.*]], %[[arg4:.*]], %[[arg5:.*]], %[[arg6:.*]], %[[arg7:.*]], %[[arg8:.*]], %[[arg9:.*]]) = (0, 0, 0, 0, 0, 0, 0) to (1, 112, 112, 32, 3, 3, 3) reduce ("assign") -> (memref<1x112x112x32xf32>)
//       CHECK:     %[[t0:.*]] = pxa.load %[[arg0]][%[[arg3]], %[[arg4]] * 2 + %[[arg7]], %[[arg5]] * 2 + %[[arg8]], %[[arg9]]] : memref<1x255x255x3xf32>
//       CHECK:     %[[t1:.*]] = pxa.load %[[arg1]][%[[arg7]], %[[arg8]], %[[arg9]], %[[arg6]]] : memref<3x3x3x32xf32>
//       CHECK:     %[[t2:.*]] = arith.mulf %[[t0]], %[[t1]] : f32
//       CHECK:     %[[t3:.*]] = pxa.reduce addf %[[t2]], %[[arg2]][%[[arg3]], %[[arg4]], %[[arg5]], %[[arg6]]] : memref<1x112x112x32xf32>
//       CHECK:     affine.yield %[[t3]] : memref<1x112x112x32xf32>
//       CHECK:   return %[[buf1]] : memref<1x112x112x32xf32>
