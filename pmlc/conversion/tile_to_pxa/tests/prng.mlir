// RUN: pmlc-opt -convert-tile-to-pxa -canonicalize -cse %s | FileCheck %s

module {
  func @prng(%arg0: tensor<3x2048xui32>) -> (tensor<2x3x4x5xf32>, tensor<3x2048xui32>) {
    %c2 = "eltwise.sconst"() {value = 2 : i64} : () -> tensor<si32>
    %c3 = "eltwise.sconst"() {value = 3 : i64} : () -> tensor<si32>
    %c4 = "eltwise.sconst"() {value = 4 : i64} : () -> tensor<si32>
    %c5 = "eltwise.sconst"() {value = 5 : i64} : () -> tensor<si32>
    %result, %new_state = "tile.prng"(%arg0, %c2, %c3, %c4, %c5) : (tensor<3x2048xui32>, tensor<si32>, tensor<si32>, tensor<si32>, tensor<si32>) -> (tensor<2x3x4x5xf32>, tensor<3x2048xui32>)
    return %result, %new_state : tensor<2x3x4x5xf32>, tensor<3x2048xui32>
  }
}

// CHECK-LABEL: func @prng
// CHECK: %{{.*}} = memref_cast %{{.*}}: memref<3x2048xi32> to memref<*xi32>
// CHECK-NEXT %{{.*}} = memref_cast %{{.*}} : memref<2x3x4x5xf32> to memref<*xf32>
// CHECK-NEXT %{{.*}} = memref_cast %{{.*}} : memref<3x2048xi32> to memref<*xi32>
// CHECK-NEXT call @plaidml_rt_prng(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<*xi32>, memref<*xf32>, memref<*xi32>) -> ()
