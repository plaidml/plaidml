// RUN: pmlc-opt --target-intel_gen %s| FileCheck %s

!tensor = type tensor<2x4xf32>

func @eltwise_add(%A: !tensor, %B: !tensor) -> !tensor {
  %C = tile.add %A, %B : (!tensor, !tensor) -> !tensor
  return %C : !tensor
}

// CHECK: llvm.call @vlk_schedule_compute
