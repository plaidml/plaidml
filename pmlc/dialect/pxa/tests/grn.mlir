// RUN: pmlc-opt \
// RUN:   -tile-compute-bounds \
// RUN:   -convert-tile-to-pxa \
// RUN:   -canonicalize \
// RUN:   -pxa-fusion \
// RUN:   -canonicalize \
// RUN:   -pxa-dataflow-opt \
// RUN:   -canonicalize \
// RUN:   -pxa-localize \
// RUN:   -pxa-resize-tmps \
// RUN:   %s | FileCheck %s

#map0 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>

func @grn(%arg0: tensor<1x4x4x3xf16>) -> tensor<1x4x4x3xf16> {
  %cst = "eltwise.sconst"() {value = 1.000000e-05 : f64} : () -> tensor<f16>
  %cst_0 = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> tensor<f16>
  %cst_1 = "eltwise.sconst"() {value = 1.000000e+00 : f64} : () -> tensor<f16>
  %0 = "eltwise.mul"(%arg0, %arg0) : (tensor<1x4x4x3xf16>, tensor<1x4x4x3xf16>) -> tensor<1x4x4x3xf16>
  %1 = tile.contract add, none, %cst_0, %0 {sink = #map0, srcs = [#map1]} : tensor<f16>, tensor<1x4x4x3xf16> -> tensor<1x4x4x1xf16>
  %2 = "eltwise.add"(%1, %cst_1) : (tensor<1x4x4x1xf16>, tensor<f16>) -> tensor<1x4x4x1xf16>
  %3 = "eltwise.sqrt"(%2) : (tensor<1x4x4x1xf16>) -> tensor<1x4x4x1xf16>
  %4 = "eltwise.cmp_lt"(%3, %cst) : (tensor<1x4x4x1xf16>, tensor<f16>) -> tensor<1x4x4x1xi1>
  %5 = "eltwise.select"(%4, %cst, %3) : (tensor<1x4x4x1xi1>, tensor<f16>, tensor<1x4x4x1xf16>) -> tensor<1x4x4x1xf16>
  %6 = "eltwise.div"(%arg0, %5) : (tensor<1x4x4x3xf16>, tensor<1x4x4x1xf16>) -> tensor<1x4x4x3xf16>
  return %6 : tensor<1x4x4x3xf16>
}

// CHECK: func @grn(%[[in:.*]]: memref<1x4x4x3xf16>) -> memref<1x4x4x3xf16>
// CHECK:   %[[epsilon:.*]] = constant 1.001360e-05 : f16
// CHECK:   %[[zero:.*]] = constant 0.000000e+00 : f16
// CHECK:   %[[one:.*]] = constant 1.000000e+00 : f16
// CHECK:   %[[out:.*]] = alloc() : memref<1x4x4x3xf16>
// CHECK:   %[[P0:.*]] = affine.parallel (%[[i:.*]], %[[j:.*]]) = (0, 0) to (4, 4) : memref<1x4x4x3xf16>
// CHECK:     %[[T0:.*]] = alloc() : memref<1x1x1x1xf16>
// CHECK:     %[[T1:.*]] = pxa.reduce assign %[[zero]], %[[T0]][0, 0, 0, 0] : memref<1x1x1x1xf16>
// CHECK:     %[[P1:.*]] = affine.parallel (%[[k:.*]]) = (0) to (3) : memref<1x1x1x1xf16>
// CHECK:       %[[X0:.*]] = affine.load %[[in]][0, %[[i]], %[[j]], %[[k]]] : memref<1x4x4x3xf16>
// CHECK:       %[[X1:.*]] = affine.load %[[in]][0, %[[i]], %[[j]], %[[k]]] : memref<1x4x4x3xf16>
// CHECK:       %[[X2:.*]] = mulf %[[X0]], %[[X1]] : f16
// CHECK:       %[[X3:.*]] = pxa.reduce add %[[X2]], %[[T1]][0, 0, 0, 0] : memref<1x1x1x1xf16>
// CHECK:       affine.yield %[[X3]] : memref<1x1x1x1xf16>
// CHECK:     %[[X5:.*]] = affine.load %[[P1]][0, 0, 0, 0] : memref<1x1x1x1xf16>
// CHECK:     %[[X6:.*]] = addf %[[X5]], %[[one]] : f16
// CHECK:     %[[X7:.*]] = sqrt %[[X6]] : f16
// CHECK:     %[[X8:.*]] = cmpf "olt", %[[X7]], %[[epsilon]] : f16
// CHECK:     %[[X9:.*]] = select %[[X8]], %[[epsilon]], %[[X7]] : f16
// CHECK:     %[[P1:.*]] = affine.parallel (%[[k:.*]]) = (0) to (3) : memref<1x4x4x3xf16>
// CHECK:       %[[X11:.*]] = affine.load %[[in]][0, %[[i]], %[[j]], %[[k]]] : memref<1x4x4x3xf16>
// CHECK:       %[[X12:.*]] = divf %[[X11]], %[[X9]] : f16
// CHECK:       %[[X13:.*]] = pxa.reduce assign %[[X12]], %[[out]][0, %[[i]], %[[j]], %[[k]]] : memref<1x4x4x3xf16>
// CHECK:       affine.yield %[[X13]] : memref<1x4x4x3xf16>
// CHECK:     affine.yield %[[P1]] : memref<1x4x4x3xf16>
// CHECK:   return %[[P0]] : memref<1x4x4x3xf16>
