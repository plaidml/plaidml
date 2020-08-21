// RUN: pmlc-opt -convert-stdx-to-llvm -canonicalize -cse %s | FileCheck %s

module {
  func @reshaper0(%arg0: memref<1x1x60xf32>) -> memref<60xf32> {
    %0 = stdx.reshape(%arg0) : (memref<1x1x60xf32>) -> memref<60xf32>
    return %0 : memref<60xf32>
  }
}

// CHECK-LABEL: func @reshaper0
// CHECK: undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK: %[[index0:.*]] = llvm.mlir.constant(60 : index)
// CHECK: insertvalue %[[index0]], %{{.*}}[3, 0]
// CHECK: %[[index1:.*]] = llvm.mlir.constant(1 : index)
// CHECK: insertvalue %[[index1]], %{{.*}}[4, 0]
// CHECK: return %{{.*}} : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>

// -----

module {
  func @reshaper1(%arg0: memref<2x4x5xf32>) -> memref<2x20xf32> {
    %0 = stdx.reshape(%arg0) : (memref<2x4x5xf32>) -> memref<2x20xf32>
    return %0 : memref<2x20xf32>
  }
}

// CHECK-LABEL: func @reshaper1
// CHECK: undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK: %[[index0:.*]] = llvm.mlir.constant(2 : index)
// CHECK: insertvalue %[[index0]], %{{.*}}[3, 0]
// CHECK: %[[index1:.*]] = llvm.mlir.constant(20 : index)
// CHECK: insertvalue %[[index1]], %{{.*}}[3, 1]
// CHECK: insertvalue %[[index1]], %{{.*}}[4, 0]
// CHECK: %[[index2:.*]] = llvm.mlir.constant(1 : index)
// CHECK: insertvalue %[[index2]], %{{.*}}[4, 1]
// CHECK: return %{{.*}} : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>

// -----

module {
  func @reshaper2(%arg0: memref<5x2x3xf32>) -> memref<5x6xf32> {
    %0 = stdx.reshape(%arg0) : (memref<5x2x3xf32>) -> memref<5x6xf32>
    return %0 : memref<5x6xf32>
  }
}

// CHECK-LABEL: func @reshaper2
// CHECK: undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK: %[[index0:.*]] = llvm.mlir.constant(5 : index)
// CHECK: insertvalue %[[index0]], %{{.*}}[3, 0]
// CHECK: %[[index1:.*]] = llvm.mlir.constant(6 : index)
// CHECK: insertvalue %[[index1]], %{{.*}}[3, 1]
// CHECK: insertvalue %[[index1]], %{{.*}}[4, 0]
// CHECK: %[[index2:.*]] = llvm.mlir.constant(1 : index)
// CHECK: insertvalue %[[index2]], %{{.*}}[4, 1]
// CHECK: return %{{.*}} : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>


// -----

module {
  func @squeeze(%arg0: memref<4x2x1x3x2xf32>) -> memref<4x2x3x2xf32> {
    %0 = stdx.reshape(%arg0) : (memref<4x2x1x3x2xf32>) -> memref<4x2x3x2xf32>
    return %0 : memref<4x2x3x2xf32>
  }
}

// CHECK-LABEL: func @squeeze
// CHECK: undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<5 x i64>, array<5 x i64>)>
// CHECK: %[[index0:.*]] = llvm.mlir.constant(4 : index)
// CHECK: insertvalue %[[index0]], %{{.*}}[3, 0]
// CHECK: %[[index1:.*]] = llvm.mlir.constant(2 : index)
// CHECK: insertvalue %[[index1]], %{{.*}}[3, 1]
// CHECK: %[[index2:.*]] = llvm.mlir.constant(3 : index)
// CHECK: insertvalue %[[index2]], %{{.*}}[3, 2]
// CHECK: insertvalue %[[index1]], %{{.*}}[3, 3]
// CHECK: %[[index3:.*]] = llvm.mlir.constant(12 : index)
// CHECK: insertvalue %[[index3]], %{{.*}}[4, 0]
// CHECK: %[[index4:.*]] = llvm.mlir.constant(6 : index)
// CHECK: insertvalue %[[index4]], %{{.*}}[4, 1]
// CHECK: insertvalue %[[index1]], %{{.*}}[4, 2]
// CHECK: %[[index5:.*]] = llvm.mlir.constant(1 : index)
// CHECK: insertvalue %[[index5]], %{{.*}}[4, 3]
// CHECK: return %{{.*}} : !llvm.struct<(ptr<float>, ptr<float>, i64, array<4 x i64>, array<4 x i64>)>

