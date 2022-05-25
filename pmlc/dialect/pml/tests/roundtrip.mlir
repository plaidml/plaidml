// RUN: pmlc-opt -split-input-file %s | pmlc-opt | FileCheck %s

// CHECK-LABEL: func.func @schedule
//  CHECK-SAME:   schedule = #pml.schedule<(d0, d1, d2) -> (0, 0, d0, d1, 0, 0, d2), [gemm_m:56, gemm_n:64, gemm_k:64]>
func.func @schedule() attributes {schedule = #pml.schedule<(m, n, k) -> (0, 0, m, n, 0, 0, k), [gemm_m:56, gemm_n:64, gemm_k:64]>} {
  return
}
