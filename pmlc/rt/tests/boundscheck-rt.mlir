// RUN: pmlc-opt -stdx-check-bounds -convert-std-to-llvm='emit-c-wrappers' %s | pmlc-jit | FileCheck %s

func @main() {
  %c0 = constant 0 : index
  %c10 = constant 10 : index // out of bounds
  %buf = memref.alloc() : memref<20x10xf32>
  %0 = memref.load %buf[%c0, %c10] : memref<20x10xf32>
  memref.dealloc %buf : memref<20x10xf32>
  return
}
// CHECK: ERROR: Out of bounds index for mlir::memref::LoadOp or mlir::memref::StoreOp
