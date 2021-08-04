// RUN: pmlc-opt %s | pmlc-opt | FileCheck %s

// CHECK-LABEL: func @init
func @init() -> !stdx.argpack {
  // CHECK: stdx.pack() : () -> !stdx.argpack
  %0 = stdx.pack() : () -> !stdx.argpack
  return %0 : !stdx.argpack
}
