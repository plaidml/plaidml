// RUN: pmlc-opt -x86-convert-std-to-llvm -x86-trace-linking %s | pmlc-jit | FileCheck %s

func @print_memref_f32(memref<*xf32>)
func @plaidml_rt_thread_num() -> index

func @main() {
  %num_threads = constant 4 : index
  %B = alloc() : memref<4xf32>
  omp.parallel num_threads(%num_threads : index) {
    %tid = call @plaidml_rt_thread_num() : () -> index
    %c0 = constant 0 : index
    %cf0 = constant 42.0 : f32
    store %cf0, %B[%tid] : memref<4xf32>
    omp.terminator
  }
  %B_u = memref_cast %B : memref<4xf32> to memref<*xf32>
  call @print_memref_f32(%B_u) : (memref<*xf32>) -> ()
  return
}

// CHECK: [42,  42,  42,  42]

