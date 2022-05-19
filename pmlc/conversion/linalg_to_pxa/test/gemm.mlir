// RUN: pmlc-opt -convert-linalg-to-pxa-new %s | FileCheck %s
// XFAIL: *
module @gemm  {
  func @main() {
    stdx.closure(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>, %arg2: tensor<3x3xf32>) -> tensor<3x3xf32> {
      %0 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<3x3xf32>, tensor<3x3xf32>) outs(%arg2 : tensor<3x3xf32>) attrs =  {iterator_ranges = [3, 3, 3]} {
      ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
        %1 = mulf %arg3, %arg4 : f32
        %2 = addf %arg5, %1 : f32
        linalg.yield %2 : f32
      } -> tensor<3x3xf32>
      stdx.yield %0 : tensor<3x3xf32>
    }
    return
  }
}
