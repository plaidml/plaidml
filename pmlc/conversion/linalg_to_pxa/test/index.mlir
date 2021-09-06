// RUN: pmlc-opt -convert-linalg-to-pxa -cse %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

func @test_index() -> (tensor<16x16xindex>, tensor<16x16xi64>) {
  %out0 = linalg.init_tensor [16, 16] : tensor<16x16xindex>
  %out1 = linalg.init_tensor [16, 16] : tensor<16x16xi64>
  %t0, %t1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} outs(%out0, %out1 : tensor<16x16xindex>, tensor<16x16xi64>) {
    ^bb0(%arg0 : index, %arg1 : i64):
    %i = linalg.index 0 : index
    %j0 = linalg.index 1 : index
    %j1 = index_cast %j0 : index to i64
    linalg.yield %i, %j1 : index, i64
  } -> (tensor<16x16xindex>, tensor<16x16xi64>)
  return %t0, %t1 : tensor<16x16xindex>, tensor<16x16xi64>
}

// CHECK-LABEL: func @test_index
// CHECK-SAME: (%[[arg0:.*]]: memref<16x16xindex>, %[[arg1:.*]]: memref<16x16xi64>) -> (memref<16x16xindex>, memref<16x16xi64>)
// CHECK: %[[out:.*]]:2 = affine.parallel (%[[arg2:.*]], %[[arg3:.*]]) = (0, 0) to (16, 16) reduce ("assign", "assign") -> (memref<16x16xindex>, memref<16x16xi64>)
// CHECK:   %[[t0:.*]] = index_cast %[[arg3]] : index to i64
// CHECK:   %[[t1:.*]] = pxa.reduce assign %[[arg2]], %[[arg0]][%[[arg2]], %[[arg3]]] : memref<16x16xindex>
// CHECK:   %[[t2:.*]] = pxa.reduce assign %[[t0]], %[[arg1]][%[[arg2]], %[[arg3]]] : memref<16x16xi64>
// CHECK:   affine.yield %[[t1]], %[[t2]] : memref<16x16xindex>, memref<16x16xi64>
// CHECK: return %[[out]]#0, %[[out]]#1 : memref<16x16xindex>, memref<16x16xi64>
