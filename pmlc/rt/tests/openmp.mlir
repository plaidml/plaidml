// RUN: pmlc-opt -x86-convert-std-to-llvm -x86-trace-linking %s -split-input-file | pmlc-jit | FileCheck %s

func @__trace_0() attributes {id = 0 : i64, msg = "msg\n", trace}

func @main() {
  %cst = constant 1.000000e+00 : f32
  omp.parallel {
    call @__trace_0() : () -> ()
    omp.terminator
  }
  call @__trace_0() : () -> ()
  return
}

// CHECK: msg
