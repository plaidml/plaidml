// RUN: pmlc-opt -convert-tile-to-linalg -cse %s | FileCheck %s

func.func @max_pool(%arg0: tensor<3xui64>) -> tensor<1xui64> {
  %c0 = tile.constant(0 : i64) : tensor<ui64>
  %0 = tile.contract max, none, %c0, %arg0 {
    sink = affine_map<(d0, d1) -> (d0)>,
    srcs = [affine_map<(d0, d1) -> (d0 * 2 + d1)>],
    lowerBounds = affine_map<() -> (0, 0)>,
    upperBounds = affine_map<() -> (0, 1)>
  } : tensor<ui64>, tensor<3xui64> -> tensor<1xui64>
  return %0 : tensor<1xui64>
}

// CHECK-LABEL: func.func @max_pool
//       CHECK:   linalg.fill
//       CHECK:   linalg.generic
//       CHECK:     cmpi ugt
//       CHECK:     select
//       CHECK:     linalg.yield
//       CHECK:   return
