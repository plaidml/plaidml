// RUN: pmlc-opt -convert-tile-to-linalg %s | FileCheck %s

func @closure() {
  %c0 = tile.constant(0 : i64) : tensor<ui64>
  stdx.closure(%arg0: tensor<3xui64>) -> tensor<1xui64> {
    %0 = tile.contract max, none, %c0, %arg0 {lowerBounds = affine_map<() -> (0, 0)>, sink = affine_map<(d0, d1) -> (d0)>, srcs = [affine_map<(d0, d1) -> (d0 * 2 + d1)>], upperBounds = affine_map<() -> (0, 1)>} : tensor<ui64>, tensor<3xui64> -> tensor<1xui64>
    stdx.yield %0 : tensor<1xui64>
  }
  return
}

// CHECK-LABEL: func @closure()
// CHECK: stdx.closure({{.*}}: tensor<3xi64>) -> tensor<1xi64>
// CHECK: linalg.fill
// CHECK: linalg.generic
// CHECK:   cmpi ugt
// CHECK:   select
// CHECK:   linalg.yield
// CHECK: stdx.yield
// CHECK: return
