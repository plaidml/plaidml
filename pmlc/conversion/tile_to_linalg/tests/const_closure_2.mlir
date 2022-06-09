// RUN: pmlc-opt -convert-tile-to-linalg %s | FileCheck %s

func @main(%arg0: tensor<4xsi32> {stdx.const}) {
  stdx.closure() -> tensor<4xsi32> {
    stdx.yield %arg0 : tensor<4xsi32>
  }
  return
}

// CHECK-LABEL: func @main
