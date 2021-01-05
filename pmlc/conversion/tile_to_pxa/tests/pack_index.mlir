// Check if pack/unpack crash for constant op.
// RUN: pmlc-opt -convert-tile-to-pxa %s | FileCheck %s

// CHECK-LABEL: @init
func @init() -> !stdx.argpack {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = stdx.pack %c1, %c0: index, index
  return %0 : !stdx.argpack
}

// CHECK-LABEL: @fini
func @fini(%arg0: !stdx.argpack) {
  %0:2 = stdx.unpack %arg0 : index, index
  return
}
