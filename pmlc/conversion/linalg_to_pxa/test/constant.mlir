// RUN: pmlc-opt --convert-linalg-to-pxa %s | FileCheck %s

#map0 = affine_map<(d0, d1, d2, d3) -> ()>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>  
func @main(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32> {
  %cst = constant dense<"0x00000000"> : tensor<f32>
  %0 = linalg.init_tensor [1, 224, 224, 3] : tensor<1x224x224x3xf32>
  %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst : tensor<f32>) outs(%0 : tensor<1x224x224x3xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
    linalg.yield %arg1 : f32
  } -> tensor<1x224x224x3xf32>
  return %1 : tensor<1x224x224x3xf32>
}

// CHECK: memref.global "private" constant @[[cst:.*]] : memref<f32> = dense<0.000000e+00>
// CHECK: func @main
// CHECK: memref.get_global @[[cst]] : memref<f32>
