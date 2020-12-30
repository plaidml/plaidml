// RUN: pmlc-opt -tile-split-main="main-function=split" %s | FileCheck %s

func @split(%arg0: tensor<16x16xf32>) -> tensor<16x16xf32> {
  return %arg0 : tensor<16x16xf32>
}

// CHECK-LABEL: @init
// CHECK-SAME: !stdx.argpack
// CHECK:   stdx.pack
// CHECK:   return
// CHECK-LABEL: @split
// CHECK-SAME: (%[[arg0:.*]]: !stdx.argpack, %[[arg1:.*]]: tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK:   stdx.unpack %[[arg0]]
// CHECK:   return %[[arg1]]
// CHECK-LABEL: @fini
// CHECK-SAME: !stdx.argpack
// CHECK:   stdx.unpack
// CHECK:   return
