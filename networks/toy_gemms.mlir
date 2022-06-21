#map0 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
module @predict_function {
  func @main(%arg0: tensor<1x256xf32>, %arg1: tensor<256x512xf32> {stdx.const}, %arg2: tensor<512xf32> {stdx.const}, %arg3: tensor<512x1024xf32> {stdx.const}, %arg4: tensor<1024xf32> {stdx.const}, %arg5: tensor<1024x2048xf32> {stdx.const}, %arg6: tensor<2048xf32> {stdx.const}, %arg7: tensor<2048x1000xf32> {stdx.const}, %arg8: tensor<1000xf32> {stdx.const}) -> tensor<1x1000xf32> {
    %0 = tile.constant(0.000000e+00 : f64) : tensor<f32>
    %1 = tile.contract add, mul, %0, %arg0, %arg1 {sink = #map0, srcs = [#map1, #map2]} : tensor<f32>, tensor<1x256xf32>, tensor<256x512xf32> -> tensor<1x512xf32>
    %2 = tile.add %1, %arg2 : (tensor<1x512xf32>, tensor<512xf32>) -> tensor<1x512xf32>
    %3 = tile.relu %2 : (tensor<1x512xf32>) -> tensor<1x512xf32>
    %4 = tile.contract add, mul, %0, %3, %arg3 {sink = #map0, srcs = [#map1, #map2]} : tensor<f32>, tensor<1x512xf32>, tensor<512x1024xf32> -> tensor<1x1024xf32>
    %5 = tile.add %4, %arg4 : (tensor<1x1024xf32>, tensor<1024xf32>) -> tensor<1x1024xf32>
    %6 = tile.relu %5 : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %7 = tile.contract add, mul, %0, %6, %arg5 {sink = #map0, srcs = [#map1, #map2]} : tensor<f32>, tensor<1x1024xf32>, tensor<1024x2048xf32> -> tensor<1x2048xf32>
    %8 = tile.add %7, %arg6 : (tensor<1x2048xf32>, tensor<2048xf32>) -> tensor<1x2048xf32>
    %9 = tile.relu %8 : (tensor<1x2048xf32>) -> tensor<1x2048xf32>
    %10 = tile.contract add, mul, %0, %9, %arg7 {sink = #map0, srcs = [#map1, #map2]} : tensor<f32>, tensor<1x2048xf32>, tensor<2048x1000xf32> -> tensor<1x1000xf32>
    %11 = tile.add %10, %arg8 : (tensor<1x1000xf32>, tensor<1000xf32>) -> tensor<1x1000xf32>
    %12 = tile.relu %11 : (tensor<1x1000xf32>) -> tensor<1x1000xf32>
    return %12 : tensor<1x1000xf32>
  }
}
