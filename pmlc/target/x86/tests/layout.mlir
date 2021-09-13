// #filter0 = affine_map<(r, s, k0, k1, c0, c1) -> (r, s, k1 * 16 + k0, c1 * 16 + c0)>
// #filter1 = affine_map<(r, s, k0, k1, c0, c1) -> (k1, c1, r, s, k0, c0)>

// #input0 = affine_map<(n, h, w, c0, c1) -> (n, h, w, c1 * 16 + c0)>
// #input1 = affine_map<(n, h, w, c0, c1) -> (n, c1, h, w, c0)>

// #output0 = affine_map<(n, h, w, c0, c1) -> (n, c1, h, w, c0)>
// #output1 = affine_map<(n, h, w, c0, c1) -> (n, h, w, c1 * 16 + c0)>

// #input  = affine_map<(n, h, w, c0, c1, r, s, k0, k1) -> (n, k1, h + r, w + s, k0)>
// #filter = affine_map<(n, h, w, c0, c1, r, s, k0, k1) -> (k1, c1, r, s, k0, c0)>
// #output = affine_map<(n, h, w, c0, c1, r, s, k0, k1) -> (n, c1, h, w, c0)>

// #map3 = affine_map<(n, h, w, c0, c1) -> (n, c1, h, w, c0)>
// #map4 = affine_map<(n, h, w, c0, c1) -> (c1 * 16 + c0)>

#input = affine_map<(n, h, w, c, r, s, k) -> (n, h + r, w + s, k)>
#filter = affine_map<(n, h, w, c, r, s, k) -> (r, s, k, c)>
#output = affine_map<(n, h, w, c, r, s, k) -> (n, h, w, c)>
#bias   = affine_map<(n, h, w, c) -> (c)>
#act   = affine_map<(n, h, w, c) -> (n, h, w, c)>

// func @main(%arg0: tensor<1x56x56x64xf32>, %arg1: tensor<1x1x64x64xf32> {stdx.const}, %arg2: tensor<64xf32> {stdx.const}) -> tensor<1x56x56x64xf32> {
//   %zero = tile.constant(0.0 : f64) : tensor<f32>
//   %0 = tile.contract add, mul, %zero, %arg0, %arg1 {sink = #output, srcs = [#input, #filter]} : tensor<f32>, tensor<1x56x56x64xf32>, tensor<1x1x64x64xf32> -> tensor<1x56x56x64xf32>
//   %1 = tile.add %0, %arg2 : (tensor<1x56x56x64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
//   %2 = tile.relu %1 : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
//   return %2 : tensor<1x56x56x64xf32>
// }

func @main(%arg0: tensor<1x56x56x64xf32>, %arg1: tensor<1x1x64x64xf32> {stdx.const}, %arg2: tensor<64xf32> {stdx.const}) -> tensor<1x56x56x64xf32> {
  %cst = constant 0.000000e+00 : f32
  %0 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
  %1 = linalg.fill(%cst, %0) : f32, tensor<1x56x56x64xf32> -> tensor<1x56x56x64xf32>
  %2 = linalg.generic {
    indexing_maps = [#input, #filter, #output],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
  } ins(%arg0, %arg1 : tensor<1x56x56x64xf32>, tensor<1x1x64x64xf32>) outs(%1 : tensor<1x56x56x64xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
    %5 = mulf %arg3, %arg4 : f32
    %6 = addf %arg5, %5 : f32
    linalg.yield %6 : f32
  } -> tensor<1x56x56x64xf32>
  %3 = linalg.generic {
    indexing_maps = [#act, #bias, #act],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%2, %arg2 : tensor<1x56x56x64xf32>, tensor<64xf32>) outs(%0 : tensor<1x56x56x64xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
    %5 = addf %arg3, %arg4 : f32
    linalg.yield %5 : f32
  } -> tensor<1x56x56x64xf32>
  %4 = linalg.generic {
    indexing_maps = [#act, #act],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%3 : tensor<1x56x56x64xf32>) outs(%0 : tensor<1x56x56x64xf32>) {
  ^bb0(%arg3: f32, %arg4: f32):  // no predecessors
    %5 = stdx.relu(%arg3) : (f32) -> f32
    linalg.yield %5 : f32
  } -> tensor<1x56x56x64xf32>
  return %4 : tensor<1x56x56x64xf32>
}

// func @main(%I0: tensor<1x56x56x64xf32>, %F0: tensor<1x1x64x64xf32> {stdx.const}, %B0: tensor<64xf32> {stdx.const}) -> tensor<1x56x56x64xf32> {
//   %zero = constant 0.000000e+00 : f32
//   %T0 = linalg.init_tensor [4, 4, 1, 1, 16, 16] : tensor<4x4x1x1x16x16xf32>
//   %F1 = linalg.generic {
//     indexing_maps = [#filter0, #filter1],
//     iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]
//   } ins(%F0 : tensor<1x1x64x64xf32>) outs(%T0 : tensor<4x4x1x1x16x16xf32>) {
//   ^bb0(%arg3: f32, %arg4 : f32):  // no predecessors
//     linalg.yield %arg3 : f32
//   } -> tensor<4x4x1x1x16x16xf32>

//   %T1 = linalg.init_tensor [1, 4, 56, 56, 16] : tensor<1x4x56x56x16xf32>
//   %I1 = linalg.generic {
//     indexing_maps = [#input0, #input1],
//     iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
//   } ins(%I0 : tensor<1x56x56x64xf32>) outs(%T1 : tensor<1x4x56x56x16xf32>) {
//   ^bb0(%arg3: f32, %arg4 : f32):  // no predecessors
//     linalg.yield %arg3 : f32
//   } -> tensor<1x4x56x56x16xf32>

//   %0 = linalg.init_tensor [1, 4, 56, 56, 16] : tensor<1x4x56x56x16xf32>
//   %1 = linalg.fill(%zero, %0) : f32, tensor<1x4x56x56x16xf32> -> tensor<1x4x56x56x16xf32>
//   %2 = linalg.generic {
//     indexing_maps = [#input, #filter, #output],
//     iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]
//   } ins(%I1, %F1 : tensor<1x4x56x56x16xf32>, tensor<4x4x1x1x16x16xf32>) outs(%1 : tensor<1x4x56x56x16xf32>) {
//   ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
//     %5 = mulf %arg3, %arg4 : f32
//     %6 = addf %arg5, %5 : f32
//     linalg.yield %6 : f32
//   } -> tensor<1x4x56x56x16xf32>

//   %T2 = linalg.init_tensor [1, 4, 56, 56, 16] : tensor<1x4x56x56x16xf32>
//   %3 = linalg.generic {
//     indexing_maps = [#map3, #map4, #map3],
//     iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
//   } ins(%2, %B0 : tensor<1x4x56x56x16xf32>, tensor<64xf32>) outs(%T2 : tensor<1x4x56x56x16xf32>) {
//   ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
//     %5 = addf %arg3, %arg4 : f32
//     linalg.yield %5 : f32
//   } -> tensor<1x4x56x56x16xf32>

//   %T3 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
//   %4 = linalg.generic {
//     indexing_maps = [#output0, #output1],
//     iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
//   } ins(%3 : tensor<1x4x56x56x16xf32>) outs(%T3 : tensor<1x56x56x64xf32>) {
//   ^bb0(%arg3: f32, %arg4: f32):  // no predecessors
//     linalg.yield %arg3 : f32
//   } -> tensor<1x56x56x64xf32>

//   return %4 : tensor<1x56x56x64xf32>

