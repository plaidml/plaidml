// RUN: pmlc-opt -convert-linalg-to-pxa -cse %s | FileCheck %s

func @main(%arg0 : tensor<1x56x56x64xf32>, %arg1: tensor<1x1x64x256xf32> {stdx.const}) -> tensor<1x56x56x256xf32> {
  %0 = linalg.init_tensor [1, 56, 56, 256] : tensor<1x56x56x256xf32>
  %1 = linalg.conv_2d_input_nhwc_filter_hwcf
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%arg0, %arg1 : tensor<1x56x56x64xf32>, tensor<1x1x64x256xf32>)
    outs(%0 : tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
  return %1 : tensor<1x56x56x256xf32>
}

// CHECK-LABEL: func @main
// TODO
