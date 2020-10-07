// RUN: pmlc-opt -pmlc-convert-affine-to-omp --split-input-file %s | FileCheck %s

module {
  // CHECK-LABEL: func @test
  func @test() {
    return
  }
}
