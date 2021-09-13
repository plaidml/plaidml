// RUN: pmlc-opt -split-input-file -pmlc-hoisting %s | FileCheck %s

#map0 = affine_map<(r, s, k0, k1, c0, c1) -> (r, s, k1 * 16 + k0, c1 * 16 + c0)>
#map1 = affine_map<(r, s, k0, k1, c0, c1) -> (k1, c1, r, s, k0, c0)>

func @main(%arg0: tensor<1x1x64x64xf32> {stdx.const}) {
  stdx.closure() -> tensor<4x4x1x1x16x16xf32> {
    %zero = constant 0.000000e+00 : f32

    %0 = linalg.init_tensor [4, 4, 1, 1, 16, 16] : tensor<4x4x1x1x16x16xf32>
    %1 = linalg.generic {
      indexing_maps = [#map0, #map1],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]
    } ins(%arg0 : tensor<1x1x64x64xf32>) outs(%0 : tensor<4x4x1x1x16x16xf32>) {
    ^bb0(%arg3: f32, %arg4 : f32):  // no predecessors
      linalg.yield %arg3 : f32
    } -> tensor<4x4x1x1x16x16xf32>

    stdx.yield %1 : tensor<4x4x1x1x16x16xf32>
  }

  return
}

// CHECK-LABEL: func @main
//       CHECK:   constant
//       CHECK:   linalg.init_tensor
//       CHECK:   linalg.generic
//       CHECK:   stdx.closure()
//       CHECK:     stdx.yield
//       CHECK:   return
