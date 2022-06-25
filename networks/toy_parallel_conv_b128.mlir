module @predict_function {
  func @main(%arg0: tensor<128x224x224x3xf32>, %arg1: tensor<7x7x3x64xf32> {stdx.const}, %arg2: tensor<64xf32> {stdx.const}, %arg3: tensor<1x1x64x256xf32> {stdx.const}, %arg4: tensor<256xf32> {stdx.const}, %arg5: tensor<3x3x64x256xf32> {stdx.const}, %arg6: tensor<256xf32> {stdx.const}) -> tensor<128x56x56x256xf32> {
    %0 = tile.constant(0.000000e+00 : f64) : tensor<f32>
    %1 = tile.contract add, mul, %0, %arg0, %arg1 {sink = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>, srcs = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4 - 2, d2 * 2 + d5 - 2, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>]} : tensor<f32>, tensor<128x224x224x3xf32>, tensor<7x7x3x64xf32> -> tensor<128x112x112x64xf32>
    %2 = tile.add %1, %arg2 : (tensor<128x112x112x64xf32>, tensor<64xf32>) -> tensor<128x112x112x64xf32>
    %3 = tile.relu %2 : (tensor<128x112x112x64xf32>) -> tensor<128x112x112x64xf32>
    %4 = tile.contract add, mul, %0, %3, %arg3 {sink = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>, srcs = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>]} : tensor<f32>, tensor<128x112x112x64xf32>, tensor<1x1x64x256xf32> -> tensor<128x56x56x256xf32>
    %5 = tile.add %4, %arg4 : (tensor<128x56x56x256xf32>, tensor<256xf32>) -> tensor<128x56x56x256xf32>
    %6 = tile.relu %5 : (tensor<128x56x56x256xf32>) -> tensor<128x56x56x256xf32>
    %7 = tile.contract add, mul, %0, %3, %arg5 {sink = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>, srcs = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>]} : tensor<f32>, tensor<128x112x112x64xf32>, tensor<3x3x64x256xf32> -> tensor<128x56x56x256xf32>
    %8 = tile.add %7, %arg6 : (tensor<128x56x56x256xf32>, tensor<256xf32>) -> tensor<128x56x56x256xf32>
    %9 = tile.relu %8 : (tensor<128x56x56x256xf32>) -> tensor<128x56x56x256xf32>
    %10 = tile.add %6, %9 : (tensor<128x56x56x256xf32>, tensor<128x56x56x256xf32>) -> tensor<128x56x56x256xf32>
    %11 = tile.relu %10 : (tensor<128x56x56x256xf32>) -> tensor<128x56x56x256xf32>
    return %11 : tensor<128x56x56x256xf32>
  }
}