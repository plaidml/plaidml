// RUN: pmlc-opt -stdx-check-bounds -convert-std-to-llvm='emit-c-wrappers=1' %s | pmlc-jit | FileCheck %s

func @main() {
  %c0 = constant 0 : index
  %c10 = constant 10 : index // out of bounds
  %buf = alloc() : memref<20x10xf32>
  %0 = load %buf[%c0, %c10] : memref<20x10xf32>
  dealloc %buf : memref<20x10xf32>
  return
}
// CHECK: ERROR: Out of bounds index for mlir::LoadOp or mlir::StoreOp
