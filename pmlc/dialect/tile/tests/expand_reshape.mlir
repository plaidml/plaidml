// RUN: pmlc-opt -expand-reshape %s | FileCheck %s

func @reshape_4x4x4x4(%arg0: tensor<4x4x4x4xf16>) -> tensor<1x4x4x16xf16> {
  %0 = tile.reshape %arg0 : (tensor<4x4x4x4xf16>) -> tensor<1x4x4x16xf16>
  return %0 : tensor<1x4x4x16xf16>
}
