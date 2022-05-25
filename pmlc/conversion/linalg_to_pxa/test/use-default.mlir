// RUN: pmlc-opt -convert-linalg-to-pxa %s | FileCheck %s

// CHECK-LABEL: @use_default
module @use_default {
  func.func @main() {
    stdx.closure(%arg0: tensor<1x10x10xf32>, %arg1: tensor<1x7x10x10xf32>) -> tensor<1x7x10x10xf32>{
      %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 3, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x10x10xf32>) outs(%arg1 : tensor<1x7x10x10xf32>) attrs =  {iterator_ranges = [1, 10, 10]} {
      ^bb0(%arg2: f32, %arg3: f32):
        linalg.yield %arg2 : f32
      } -> tensor<1x7x10x10xf32>
      stdx.yield %0 : tensor<1x7x10x10xf32>
    }
    return
  }
}
