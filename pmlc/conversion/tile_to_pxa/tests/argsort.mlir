// RUN: pmlc-opt -convert-tile-to-pxa -canonicalize -cse %s | FileCheck %s

func @argsort0(%arg0: tensor<20xf32>) -> tensor<20xsi32> {
  %0 = tile.argsort "asc" %arg0[0] : (tensor<20xf32>) -> tensor<20xsi32>
  return %0 : tensor<20xsi32>
}

// CHECK-LABEL: func @argsort0
// CHECK:     %[[RET:.*]] = layer.box "argsort"(%arg1, %arg0) {attrs = {}} : memref<20xi32>, memref<20xf32> -> memref<20xi32> {
// CHECK:       ^bb0(%[[OUT:.*]]: memref<20xi32>, %[[IN:.*]]: memref<20xf32>):
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C20:.*]] = arith.constant 20 : index
// CHECK-DAG:   %[[C19:.*]] = arith.constant 19 : index
// CHECK:       scf.for %[[IDX0:.*]] = %[[C0]] to %[[C20]]
// CHECK:         memref.store %{{.*}}, %[[OUT]][%[[IDX0]]] : memref<20xi32>
// CHECK:       scf.for %[[IDX1:.*]] = %[[C0]] to %[[C19]]
// CHECK:         %[[CHECKIDX:.*]] = memref.load %[[OUT]][%[[IDX1]]] : memref<20xi32>
// CHECK:         %[[CHECKVAL:.*]] = memref.load %[[IN]][%{{.*}}] : memref<20xf32>
// CHECK:         %[[SUBIDX:.*]] = arith.addi %[[IDX1]], %[[C1]] : index
// CHECK:         scf.for %[[IDX2:.*]] = %[[SUBIDX]] to %[[C20]]
// CHECK:           memref.load %[[OUT]][%[[IDX2]]] : memref<20xi32>
// CHECK:           %[[SUBVAL:.*]] = memref.load %[[IN]][%{{.*}}] : memref<20xf32>
// CHECK:           %[[COND:.*]] = arith.cmpf olt
// CHECK:           scf.if %[[COND]]
// CHECK:             memref.store %[[IDX2]]
// CHECK:             memref.store %[[SUBVAL]]
// CHECK:       layer.return %[[OUT]] : memref<20xi32>
// CHECK:     return %[[RET]] : memref<20xi32>

func @argsort1(%arg0: tensor<5x4xf32>) -> tensor<5x4xsi32> {
  %0 = tile.argsort "desc" %arg0[0] : (tensor<5x4xf32>) -> tensor<5x4xsi32>
  return %0 : tensor<5x4xsi32>
}

// CHECK-LABEL: func @argsort1
// CHECK:     %[[RET:.*]] = layer.box "argsort"(%arg1, %arg0) {attrs = {}} : memref<5x4xi32>, memref<5x4xf32> -> memref<5x4xi32> {
// CHECK:       ^bb0(%[[OUT:.*]]: memref<5x4xi32>, %[[IN:.*]]: memref<5x4xf32>):
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK:       scf.for %[[IDX0:.*]] = %[[C0]] to %[[C4]] step %[[C1]]
// CHECK:         scf.for %[[IDX1:.*]] = %[[C0]] to %[[C5]] step %[[C1]]
// CHECK:           memref.store %{{.*}}, %[[OUT]][%[[IDX1]], %[[IDX0]]] : memref<5x4xi32>
// CHECK:       scf.for %{{.*}} = %[[C0]] to %[[C4]] step %[[C1]]
// CHECK:         %[[CHECKIDX:.*]] = memref.load %[[OUT]][%{{.*}}, %{{.*}}] : memref<5x4xi32>
// CHECK:         %[[CHECKVAL:.*]] = memref.load %[[IN]][%{{.*}}, %{{.*}}] : memref<5x4xf32>
// CHECK:         %[[SUBIDX:.*]] = arith.addi %{{.*}}, %[[C1]] : index
// CHECK:           scf.for %[[IDX3:.*]] = %[[SUBIDX]] to %[[C5]] step %[[C1]]
// CHECK:           memref.load %[[OUT]][%{{.*}}, %{{.*}}] : memref<5x4xi32>
// CHECK:           %[[SUBVAL:.*]] = memref.load %[[IN]][%{{.*}}, %{{.*}}] : memref<5x4xf32>
// CHECK:           %[[COND:.*]] = arith.cmpf ogt
// CHECK:           scf.if %[[COND]]
// CHECK:             memref.store %[[IDX3]]
// CHECK:             memref.store %[[SUBVAL]]
// CHECK:       layer.return %[[OUT]] : memref<5x4xi32>
// CHECK:     return %[[RET]] : memref<5x4xi32>
