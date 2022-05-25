// RUN: pmlc-opt -stdx-check-bounds \
// RUN: -x86-convert-std-to-llvm -convert-arith-to-llvm \
// RUN: -convert-memref-to-llvm -convert-func-to-llvm='emit-c-wrappers=1' \
// RUN: -reconcile-unrealized-casts %s | pmlc-jit | FileCheck %s

func.func @main() attributes {llvm.emit_c_interface} {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index // out of bounds
  %buf = memref.alloc() : memref<20x10xf32>
  %0 = memref.load %buf[%c0, %c10] : memref<20x10xf32>
  memref.dealloc %buf : memref<20x10xf32>
  return
}
// CHECK: ERROR: Out of bounds index for mlir::memref::LoadOp or mlir::memref::StoreOp
