// RUN: pmlc-opt -stdx-check-bounds -convert-std-to-llvm %s | pmlc-jit | FileCheck %s

module {
  func @simpleLoad(%A: memref<20x10xf32>, %i: index, %j: index) -> (f32) {
    %0 = load %A[%i, %j] : memref<20x10xf32>
    return %0: f32
  }
  
  func @main() {
    %i = constant 0 : index
    %j = constant 10 : index // out of bounds
    %buf = alloc() : memref<20x10xf32>
    call @simpleLoad(%buf, %i, %j) : (memref<20x10xf32>, index, index) -> (f32)
    dealloc %buf : memref<20x10xf32>
    return
  }
  // CHECK: ERROR: Out of bounds index for mlir::LoadOp or mlir::StoreOp
}