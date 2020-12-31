// RUN: pmlc-opt -convert-tile-to-pxa -canonicalize -cse %s | FileCheck %s

func @argsort0(%arg0: tensor<20xf32>) -> tensor<20xsi32> {
  %0 = tile.argsort "asc" %arg0[0] : (tensor<20xf32>) -> tensor<20xsi32>
  return %0 : tensor<20xsi32>
}

// CHECK-LABEL: func @argsort0
// CHECK: %[[RET:.*]] = layer.box "argsort" (%[[OUT:.*]], %[[IN:.*]]) = (%arg1, %arg0)
// CHECK: %[[C0:.*]] = constant 0 : index
// CHECK: %[[C1:.*]] = constant 1 : index
// CHECK: %[[C20:.*]] = constant 20 : index
// CHECK: %[[C19:.*]] = constant 19 : index
// CHECK: scf.for %[[IDX0:.*]] = %[[C0]] to %[[C20]]
// CHECK: store %{{.*}}, %[[OUT]][%[[IDX0]]] : memref<20xi32>
// CHECK: scf.for %[[IDX1:.*]] = %[[C0]] to %[[C19]]
// CHECK: %[[CHECKIDX:.*]] = load %[[OUT]][%[[IDX1]]] : memref<20xi32>
// CHECK: %[[CHECKVAL:.*]] = load %[[IN]][%{{.*}}] : memref<20xf32>
// CHECK: %[[SUBIDX:.*]] = addi %[[IDX1]], %[[C1]] : index
// CHECK: scf.for %[[IDX2:.*]] = %[[SUBIDX]] to %[[C20]]
// CHECK: load %[[OUT]][%[[IDX2]]] : memref<20xi32>
// CHECK: %[[SUBVAL:.*]] = load %[[IN]][%{{.*}}] : memref<20xf32>
// CHECK: %[[COND:.*]] = cmpf "olt"
// CHECK: scf.if %[[COND]]
// CHECK: store %[[IDX2]]
// CHECK: store %[[SUBVAL]]
// CHECK: layer.return %[[OUT]] : memref<20xi32>
// CHECK: return %[[RET]] : memref<20xi32>

// -----

func @argsort1(%arg0: tensor<5x4xf32>) -> tensor<5x4xsi32> {
  %0 = tile.argsort "desc" %arg0[0] : (tensor<5x4xf32>) -> tensor<5x4xsi32>
  return %0 : tensor<5x4xsi32>
}

// CHECK-LABEL: func @argsort1
// CHECK: %[[RET:.*]] = layer.box "argsort" (%[[OUT:.*]], %[[IN:.*]]) = (%arg1, %arg0)
// CHECK: %[[C0:.*]] = constant 0 : index
// CHECK: %[[C1:.*]] = constant 1 : index
// CHECK: %[[C5:.*]] = constant 5 : index
// CHECK: %[[C4:.*]] = constant 4 : index
// CHECK scf.for %[[IDX0:.*]] = %[[C0]] to %[[C4]] step %[[C1]]
// CHECK scf.for %[[IDX1:.*]] = %[[C0]] to %[[C5]] step %[[C1]]
// CHECK store %{{.*}}, %[[OUT]][%[[IDX0]], %[[IDX1]]] : memref<5x4xi32>
// CHECK scf.for %{{IDX2:.*}} = %[[C0]] to %[[C5]] step %[[C1]]
// CHECK: %[[CHECKIDX:.*]] = load %[[OUT]][%{{.*}}, %{{.*}}] : memref<5x4xi32>
// CHECK: %[[CHECKVAL:.*]] = load %[[IN]][%{{.*}}, %{{.*}}] : memref<5x4xf32>
// CHECK: %[[SUBIDX:.*]] = addi %{{.*}}, %[[C1]] : index
// CHECK: scf.for %[[IDX3:.*]] = %[[SUBIDX]] to %[[C5]] step %[[C1]]
// CHECK: load %[[OUT]][%{{.*}}, %{{.*}}] : memref<5x4xi32>
// CHECK: %[[SUBVAL:.*]] = load %[[IN]][%{{.*}}, %{{.*}}] : memref<5x4xf32>
// CHECK: %[[COND:.*]] = cmpf "ogt"
// CHECK: scf.if %[[COND]]
// CHECK: store %[[IDX3]]
// CHECK: store %[[SUBVAL]]
// CHECK: layer.return %[[OUT]] : memref<5x4xi32>
// CHECK: return %[[RET]] : memref<5x4xi32>

// -----
