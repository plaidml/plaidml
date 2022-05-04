// RUN: pmlc-opt %s \
// RUN:   -x86-collapse-scf-parallel \
// RUN:   -convert-scf-to-openmp \
// RUN:   -convert-arith-to-llvm \
// RUN:   -convert-openmp-to-llvm \
// RUN:   -reconcile-unrealized-casts \
// RUN:   -x86-convert-std-to-llvm \
// RUN:   | pmlc-jit | FileCheck %s

func private @print_memref_f32(memref<*xf32>)

func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %B = memref.alloc() : memref<2x2xf32>
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
      %cst = arith.constant 42.0 : f32
      memref.store %cst, %B[%i, %j] : memref<2x2xf32>
  }
  %B_u = memref.cast %B : memref<2x2xf32> to memref<*xf32>
  call @print_memref_f32(%B_u) : (memref<*xf32>) -> ()
  return
}

// CHECK: [42, 42],
// CHECK: [42, 42]]
