// RUN: pmlc-opt -x86-convert-std-to-llvm -x86-trace-linking %s | pmlc-jit | FileCheck %s

func @__trace_0() attributes {id = 0 : i64, msg = "msg\n", trace}

func @main() {
  %c1 = constant 1 : index
  omp.parallel num_threads(%c1 : index) private(%c1: index) {
    call @__trace_0() : () -> ()
    omp.flush
    omp.barrier
    omp.taskwait
    omp.taskyield
    omp.terminator
  }
  return
}

// CHECK: msg
