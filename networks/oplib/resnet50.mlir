#map0 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4 - 3, d2 * 2 + d5 - 3, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 * 2 + d4 - 1, d2 * 2 + d5 - 1, d3)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
#map6 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4 - 1, d2 + d5 - 1, d6)>
#map7 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d6)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map10 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map11 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map12 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map13 = affine_map<(d0, d1) -> (d0, 0)>
#map14 = affine_map<(d0, d1) -> (d0, d1)>

#set0 = affine_set<(d0, d1, d2, d3, d4, d5) : (d4 >= 0, -d4 + 2 >= 0, d5 >= 0, -d5 + 2 >= 0)>

module @resnet50 {
  func @main(%arg0: tensor<1x224x224x3xf32>, %arg1: tensor<7x7x3x64xf32>, %arg2: tensor<1x1x64x64xf32>, %arg3: tensor<3x3x64x64xf32>, %arg4: tensor<1x1x64x256xf32>, %arg5: tensor<1x1x64x256xf32>, %arg6: tensor<1x1x256x64xf32>, %arg7: tensor<3x3x64x64xf32>, %arg8: tensor<1x1x64x256xf32>, %arg9: tensor<1x1x256x64xf32>, %arg10: tensor<3x3x64x64xf32>, %arg11: tensor<1x1x64x256xf32>, %arg12: tensor<1x1x256x128xf32>, %arg13: tensor<3x3x128x128xf32>, %arg14: tensor<1x1x128x512xf32>, %arg15: tensor<1x1x256x512xf32>, %arg16: tensor<1x1x512x128xf32>, %arg17: tensor<3x3x128x128xf32>, %arg18: tensor<1x1x128x512xf32>, %arg19: tensor<1x1x512x128xf32>, %arg20: tensor<3x3x128x128xf32>, %arg21: tensor<1x1x128x512xf32>, %arg22: tensor<1x1x512x128xf32>, %arg23: tensor<3x3x128x128xf32>, %arg24: tensor<1x1x128x512xf32>, %arg25: tensor<1x1x512x256xf32>, %arg26: tensor<3x3x256x256xf32>, %arg27: tensor<1x1x256x1024xf32>, %arg28: tensor<1x1x512x1024xf32>, %arg29: tensor<1x1x1024x256xf32>, %arg30: tensor<3x3x256x256xf32>, %arg31: tensor<1x1x256x1024xf32>, %arg32: tensor<1x1x1024x256xf32>, %arg33: tensor<3x3x256x256xf32>, %arg34: tensor<1x1x256x1024xf32>, %arg35: tensor<1x1x1024x256xf32>, %arg36: tensor<3x3x256x256xf32>, %arg37: tensor<1x1x256x1024xf32>, %arg38: tensor<1x1x1024x256xf32>, %arg39: tensor<3x3x256x256xf32>, %arg40: tensor<1x1x256x1024xf32>, %arg41: tensor<1x1x1024x256xf32>, %arg42: tensor<3x3x256x256xf32>, %arg43: tensor<1x1x256x1024xf32>, %arg44: tensor<1x1x1024x512xf32>, %arg45: tensor<3x3x512x512xf32>, %arg46: tensor<1x1x512x2048xf32>, %arg47: tensor<1x1x1024x2048xf32>, %arg48: tensor<1x1x2048x512xf32>, %arg49: tensor<3x3x512x512xf32>, %arg50: tensor<1x1x512x2048xf32>, %arg51: tensor<1x1x2048x512xf32>, %arg52: tensor<3x3x512x512xf32>, %arg53: tensor<1x1x512x2048xf32>, %arg54: tensor<2048x1000xf32>, %arg55: tensor<64xf32>, %arg56: tensor<64xf32>, %arg57: tensor<64xf32>, %arg58: tensor<256xf32>, %arg59: tensor<256xf32>, %arg60: tensor<64xf32>, %arg61: tensor<64xf32>, %arg62: tensor<256xf32>, %arg63: tensor<64xf32>, %arg64: tensor<64xf32>, %arg65: tensor<256xf32>, %arg66: tensor<128xf32>, %arg67: tensor<128xf32>, %arg68: tensor<512xf32>, %arg69: tensor<512xf32>, %arg70: tensor<128xf32>, %arg71: tensor<128xf32>, %arg72: tensor<512xf32>, %arg73: tensor<128xf32>, %arg74: tensor<128xf32>, %arg75: tensor<512xf32>, %arg76: tensor<128xf32>, %arg77: tensor<128xf32>, %arg78: tensor<512xf32>, %arg79: tensor<256xf32>, %arg80: tensor<256xf32>, %arg81: tensor<1024xf32>, %arg82: tensor<1024xf32>, %arg83: tensor<256xf32>, %arg84: tensor<256xf32>, %arg85: tensor<1024xf32>, %arg86: tensor<256xf32>, %arg87: tensor<256xf32>, %arg88: tensor<1024xf32>, %arg89: tensor<256xf32>, %arg90: tensor<256xf32>, %arg91: tensor<1024xf32>, %arg92: tensor<256xf32>, %arg93: tensor<256xf32>, %arg94: tensor<1024xf32>, %arg95: tensor<256xf32>, %arg96: tensor<256xf32>, %arg97: tensor<1024xf32>, %arg98: tensor<512xf32>, %arg99: tensor<512xf32>, %arg100: tensor<2048xf32>, %arg101: tensor<2048xf32>, %arg102: tensor<512xf32>, %arg103: tensor<512xf32>, %arg104: tensor<2048xf32>, %arg105: tensor<512xf32>, %arg106: tensor<512xf32>, %arg107: tensor<2048xf32>, %arg108: tensor<1000xf32>) -> tensor<1x1000xf32> {
    %c49 = tile.constant(49 : i64) : tensor<si64>
    %cst = tile.constant(0xFFF0000000000000 : f64) : tensor<f32>
    %cst_0 = tile.constant(0.000000e+00 : f64) : tensor<f32>
    %conv1 = tile.contract add, mul, %cst_0, %arg0, %arg1 {sink = #map0, srcs = [#map1, #map2]} : tensor<f32>, tensor<1x224x224x3xf32>, tensor<7x7x3x64xf32> -> tensor<1x112x112x64xf32>
    %0 = tile.add %conv1, %arg55 : (tensor<1x112x112x64xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>
    %1 = tile.cmp_lt %0, %cst_0 : (tensor<1x112x112x64xf32>, tensor<f32>) -> tensor<1x112x112x64xi1>
    %2 = tile.select %1, %cst_0, %0 : (tensor<1x112x112x64xi1>, tensor<f32>, tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
    %3 = tile.contract max, none, %cst, %2 {cons = #set0, sink = #map3, srcs = [#map4]} : tensor<f32>, tensor<1x112x112x64xf32> -> tensor<1x56x56x64xf32>
    %4 = tile.pragma %3 "trace" {msg = "res2a"} : tensor<1x56x56x64xf32>
    %res2a_branch2a = tile.contract add, mul, %cst_0, %4, %arg2 {sink = #map0, srcs = [#map5, #map2]} : tensor<f32>, tensor<1x56x56x64xf32>, tensor<1x1x64x64xf32> -> tensor<1x56x56x64xf32>
    %5 = tile.add %res2a_branch2a, %arg56 : (tensor<1x56x56x64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
    %6 = tile.cmp_lt %5, %cst_0 : (tensor<1x56x56x64xf32>, tensor<f32>) -> tensor<1x56x56x64xi1>
    %7 = tile.select %6, %cst_0, %5 : (tensor<1x56x56x64xi1>, tensor<f32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %res2a_branch2b = tile.contract add, mul, %cst_0, %7, %arg3 {sink = #map0, srcs = [#map6, #map2]} : tensor<f32>, tensor<1x56x56x64xf32>, tensor<3x3x64x64xf32> -> tensor<1x56x56x64xf32>
    %8 = tile.add %res2a_branch2b, %arg57 : (tensor<1x56x56x64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
    %9 = tile.cmp_lt %8, %cst_0 : (tensor<1x56x56x64xf32>, tensor<f32>) -> tensor<1x56x56x64xi1>
    %10 = tile.select %9, %cst_0, %8 : (tensor<1x56x56x64xi1>, tensor<f32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %res2a_branch2b_1 = tile.contract add, mul, %cst_0, %10, %arg4 {sink = #map0, srcs = [#map5, #map2]} : tensor<f32>, tensor<1x56x56x64xf32>, tensor<1x1x64x256xf32> -> tensor<1x56x56x256xf32>
    %11 = tile.add %res2a_branch2b_1, %arg58 : (tensor<1x56x56x256xf32>, tensor<256xf32>) -> tensor<1x56x56x256xf32>
    %res2a_branch1 = tile.contract add, mul, %cst_0, %4, %arg5 {sink = #map0, srcs = [#map5, #map2]} : tensor<f32>, tensor<1x56x56x64xf32>, tensor<1x1x64x256xf32> -> tensor<1x56x56x256xf32>
    %12 = tile.add %11, %res2a_branch1 : (tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %13 = tile.add %12, %arg59 : (tensor<1x56x56x256xf32>, tensor<256xf32>) -> tensor<1x56x56x256xf32>
    %14 = tile.cmp_lt %13, %cst_0 : (tensor<1x56x56x256xf32>, tensor<f32>) -> tensor<1x56x56x256xi1>
    %15 = tile.select %14, %cst_0, %13 : (tensor<1x56x56x256xi1>, tensor<f32>, tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %16 = tile.pragma %15 "trace" {msg = "res2b"} : tensor<1x56x56x256xf32>
    %res2b_branch2a = tile.contract add, mul, %cst_0, %16, %arg6 {sink = #map0, srcs = [#map5, #map2]} : tensor<f32>, tensor<1x56x56x256xf32>, tensor<1x1x256x64xf32> -> tensor<1x56x56x64xf32>
    %17 = tile.add %res2b_branch2a, %arg60 : (tensor<1x56x56x64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
    %18 = tile.cmp_lt %17, %cst_0 : (tensor<1x56x56x64xf32>, tensor<f32>) -> tensor<1x56x56x64xi1>
    %19 = tile.select %18, %cst_0, %17 : (tensor<1x56x56x64xi1>, tensor<f32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %res2b_branch2b = tile.contract add, mul, %cst_0, %19, %arg7 {sink = #map0, srcs = [#map6, #map2]} : tensor<f32>, tensor<1x56x56x64xf32>, tensor<3x3x64x64xf32> -> tensor<1x56x56x64xf32>
    %20 = tile.add %res2b_branch2b, %arg61 : (tensor<1x56x56x64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
    %21 = tile.cmp_lt %20, %cst_0 : (tensor<1x56x56x64xf32>, tensor<f32>) -> tensor<1x56x56x64xi1>
    %22 = tile.select %21, %cst_0, %20 : (tensor<1x56x56x64xi1>, tensor<f32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %res2b_branch2b_2 = tile.contract add, mul, %cst_0, %22, %arg8 {sink = #map0, srcs = [#map5, #map2]} : tensor<f32>, tensor<1x56x56x64xf32>, tensor<1x1x64x256xf32> -> tensor<1x56x56x256xf32>
    %23 = tile.add %res2b_branch2b_2, %arg62 : (tensor<1x56x56x256xf32>, tensor<256xf32>) -> tensor<1x56x56x256xf32>
    %24 = tile.add %23, %16 : (tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %25 = tile.cmp_lt %24, %cst_0 : (tensor<1x56x56x256xf32>, tensor<f32>) -> tensor<1x56x56x256xi1>
    %26 = tile.select %25, %cst_0, %24 : (tensor<1x56x56x256xi1>, tensor<f32>, tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %27 = tile.pragma %26 "trace" {msg = "res2c"} : tensor<1x56x56x256xf32>
    %res2c_branch2a = tile.contract add, mul, %cst_0, %27, %arg9 {sink = #map0, srcs = [#map5, #map2]} : tensor<f32>, tensor<1x56x56x256xf32>, tensor<1x1x256x64xf32> -> tensor<1x56x56x64xf32>
    %28 = tile.add %res2c_branch2a, %arg63 : (tensor<1x56x56x64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
    %29 = tile.cmp_lt %28, %cst_0 : (tensor<1x56x56x64xf32>, tensor<f32>) -> tensor<1x56x56x64xi1>
    %30 = tile.select %29, %cst_0, %28 : (tensor<1x56x56x64xi1>, tensor<f32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %res2c_branch2b = tile.contract add, mul, %cst_0, %30, %arg10 {sink = #map0, srcs = [#map6, #map2]} : tensor<f32>, tensor<1x56x56x64xf32>, tensor<3x3x64x64xf32> -> tensor<1x56x56x64xf32>
    %31 = tile.add %res2c_branch2b, %arg64 : (tensor<1x56x56x64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
    %32 = tile.cmp_lt %31, %cst_0 : (tensor<1x56x56x64xf32>, tensor<f32>) -> tensor<1x56x56x64xi1>
    %33 = tile.select %32, %cst_0, %31 : (tensor<1x56x56x64xi1>, tensor<f32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %res2c_branch2b_3 = tile.contract add, mul, %cst_0, %33, %arg11 {sink = #map0, srcs = [#map5, #map2]} : tensor<f32>, tensor<1x56x56x64xf32>, tensor<1x1x64x256xf32> -> tensor<1x56x56x256xf32>
    %34 = tile.add %res2c_branch2b_3, %arg65 : (tensor<1x56x56x256xf32>, tensor<256xf32>) -> tensor<1x56x56x256xf32>
    %35 = tile.add %34, %27 : (tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %36 = tile.cmp_lt %35, %cst_0 : (tensor<1x56x56x256xf32>, tensor<f32>) -> tensor<1x56x56x256xi1>
    %37 = tile.select %36, %cst_0, %35 : (tensor<1x56x56x256xi1>, tensor<f32>, tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %38 = tile.pragma %37 "trace" {msg = "res3a"} : tensor<1x56x56x256xf32>
    %res3a_branch2a = tile.contract add, mul, %cst_0, %38, %arg12 {sink = #map0, srcs = [#map7, #map2]} : tensor<f32>, tensor<1x56x56x256xf32>, tensor<1x1x256x128xf32> -> tensor<1x28x28x128xf32>
    %39 = tile.add %res3a_branch2a, %arg66 : (tensor<1x28x28x128xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
    %40 = tile.cmp_lt %39, %cst_0 : (tensor<1x28x28x128xf32>, tensor<f32>) -> tensor<1x28x28x128xi1>
    %41 = tile.select %40, %cst_0, %39 : (tensor<1x28x28x128xi1>, tensor<f32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %res3a_branch2b = tile.contract add, mul, %cst_0, %41, %arg13 {sink = #map0, srcs = [#map6, #map2]} : tensor<f32>, tensor<1x28x28x128xf32>, tensor<3x3x128x128xf32> -> tensor<1x28x28x128xf32>
    %42 = tile.add %res3a_branch2b, %arg67 : (tensor<1x28x28x128xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
    %43 = tile.cmp_lt %42, %cst_0 : (tensor<1x28x28x128xf32>, tensor<f32>) -> tensor<1x28x28x128xi1>
    %44 = tile.select %43, %cst_0, %42 : (tensor<1x28x28x128xi1>, tensor<f32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %res3a_branch2b_4 = tile.contract add, mul, %cst_0, %44, %arg14 {sink = #map0, srcs = [#map5, #map2]} : tensor<f32>, tensor<1x28x28x128xf32>, tensor<1x1x128x512xf32> -> tensor<1x28x28x512xf32>
    %45 = tile.add %res3a_branch2b_4, %arg68 : (tensor<1x28x28x512xf32>, tensor<512xf32>) -> tensor<1x28x28x512xf32>
    %res3a_branch1 = tile.contract add, mul, %cst_0, %38, %arg15 {sink = #map0, srcs = [#map7, #map2]} : tensor<f32>, tensor<1x56x56x256xf32>, tensor<1x1x256x512xf32> -> tensor<1x28x28x512xf32>
    %46 = tile.add %45, %res3a_branch1 : (tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %47 = tile.add %46, %arg69 : (tensor<1x28x28x512xf32>, tensor<512xf32>) -> tensor<1x28x28x512xf32>
    %48 = tile.cmp_lt %47, %cst_0 : (tensor<1x28x28x512xf32>, tensor<f32>) -> tensor<1x28x28x512xi1>
    %49 = tile.select %48, %cst_0, %47 : (tensor<1x28x28x512xi1>, tensor<f32>, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %50 = tile.pragma %49 "trace" {msg = "res3b"} : tensor<1x28x28x512xf32>
    %res3b_branch2a = tile.contract add, mul, %cst_0, %50, %arg16 {sink = #map0, srcs = [#map5, #map2]} : tensor<f32>, tensor<1x28x28x512xf32>, tensor<1x1x512x128xf32> -> tensor<1x28x28x128xf32>
    %51 = tile.add %res3b_branch2a, %arg70 : (tensor<1x28x28x128xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
    %52 = tile.cmp_lt %51, %cst_0 : (tensor<1x28x28x128xf32>, tensor<f32>) -> tensor<1x28x28x128xi1>
    %53 = tile.select %52, %cst_0, %51 : (tensor<1x28x28x128xi1>, tensor<f32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %res3b_branch2b = tile.contract add, mul, %cst_0, %53, %arg17 {sink = #map0, srcs = [#map6, #map2]} : tensor<f32>, tensor<1x28x28x128xf32>, tensor<3x3x128x128xf32> -> tensor<1x28x28x128xf32>
    %54 = tile.add %res3b_branch2b, %arg71 : (tensor<1x28x28x128xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
    %55 = tile.cmp_lt %54, %cst_0 : (tensor<1x28x28x128xf32>, tensor<f32>) -> tensor<1x28x28x128xi1>
    %56 = tile.select %55, %cst_0, %54 : (tensor<1x28x28x128xi1>, tensor<f32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %res3b_branch2b_5 = tile.contract add, mul, %cst_0, %56, %arg18 {sink = #map0, srcs = [#map5, #map2]} : tensor<f32>, tensor<1x28x28x128xf32>, tensor<1x1x128x512xf32> -> tensor<1x28x28x512xf32>
    %57 = tile.add %res3b_branch2b_5, %arg72 : (tensor<1x28x28x512xf32>, tensor<512xf32>) -> tensor<1x28x28x512xf32>
    %58 = tile.add %57, %50 : (tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %59 = tile.cmp_lt %58, %cst_0 : (tensor<1x28x28x512xf32>, tensor<f32>) -> tensor<1x28x28x512xi1>
    %60 = tile.select %59, %cst_0, %58 : (tensor<1x28x28x512xi1>, tensor<f32>, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %61 = tile.pragma %60 "trace" {msg = "res3c"} : tensor<1x28x28x512xf32>
    %res3c_branch2a = tile.contract add, mul, %cst_0, %61, %arg19 {sink = #map0, srcs = [#map5, #map2]} : tensor<f32>, tensor<1x28x28x512xf32>, tensor<1x1x512x128xf32> -> tensor<1x28x28x128xf32>
    %62 = tile.add %res3c_branch2a, %arg73 : (tensor<1x28x28x128xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
    %63 = tile.cmp_lt %62, %cst_0 : (tensor<1x28x28x128xf32>, tensor<f32>) -> tensor<1x28x28x128xi1>
    %64 = tile.select %63, %cst_0, %62 : (tensor<1x28x28x128xi1>, tensor<f32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %res3c_branch2b = tile.contract add, mul, %cst_0, %64, %arg20 {sink = #map0, srcs = [#map6, #map2]} : tensor<f32>, tensor<1x28x28x128xf32>, tensor<3x3x128x128xf32> -> tensor<1x28x28x128xf32>
    %65 = tile.add %res3c_branch2b, %arg74 : (tensor<1x28x28x128xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
    %66 = tile.cmp_lt %65, %cst_0 : (tensor<1x28x28x128xf32>, tensor<f32>) -> tensor<1x28x28x128xi1>
    %67 = tile.select %66, %cst_0, %65 : (tensor<1x28x28x128xi1>, tensor<f32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %res3c_branch2b_6 = tile.contract add, mul, %cst_0, %67, %arg21 {sink = #map0, srcs = [#map5, #map2]} : tensor<f32>, tensor<1x28x28x128xf32>, tensor<1x1x128x512xf32> -> tensor<1x28x28x512xf32>
    %68 = tile.add %res3c_branch2b_6, %arg75 : (tensor<1x28x28x512xf32>, tensor<512xf32>) -> tensor<1x28x28x512xf32>
    %69 = tile.add %68, %61 : (tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %70 = tile.cmp_lt %69, %cst_0 : (tensor<1x28x28x512xf32>, tensor<f32>) -> tensor<1x28x28x512xi1>
    %71 = tile.select %70, %cst_0, %69 : (tensor<1x28x28x512xi1>, tensor<f32>, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %72 = tile.pragma %71 "trace" {msg = "res3d"} : tensor<1x28x28x512xf32>
    %res3d_branch2a = tile.contract add, mul, %cst_0, %72, %arg22 {sink = #map0, srcs = [#map5, #map2]} : tensor<f32>, tensor<1x28x28x512xf32>, tensor<1x1x512x128xf32> -> tensor<1x28x28x128xf32>
    %73 = tile.add %res3d_branch2a, %arg76 : (tensor<1x28x28x128xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
    %74 = tile.cmp_lt %73, %cst_0 : (tensor<1x28x28x128xf32>, tensor<f32>) -> tensor<1x28x28x128xi1>
    %75 = tile.select %74, %cst_0, %73 : (tensor<1x28x28x128xi1>, tensor<f32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %res3d_branch2b = tile.contract add, mul, %cst_0, %75, %arg23 {sink = #map0, srcs = [#map6, #map2]} : tensor<f32>, tensor<1x28x28x128xf32>, tensor<3x3x128x128xf32> -> tensor<1x28x28x128xf32>
    %76 = tile.add %res3d_branch2b, %arg77 : (tensor<1x28x28x128xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
    %77 = tile.cmp_lt %76, %cst_0 : (tensor<1x28x28x128xf32>, tensor<f32>) -> tensor<1x28x28x128xi1>
    %78 = tile.select %77, %cst_0, %76 : (tensor<1x28x28x128xi1>, tensor<f32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %res3d_branch2b_7 = tile.contract add, mul, %cst_0, %78, %arg24 {sink = #map0, srcs = [#map5, #map2]} : tensor<f32>, tensor<1x28x28x128xf32>, tensor<1x1x128x512xf32> -> tensor<1x28x28x512xf32>
    %79 = tile.add %res3d_branch2b_7, %arg78 : (tensor<1x28x28x512xf32>, tensor<512xf32>) -> tensor<1x28x28x512xf32>
    %80 = tile.add %79, %72 : (tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %81 = tile.cmp_lt %80, %cst_0 : (tensor<1x28x28x512xf32>, tensor<f32>) -> tensor<1x28x28x512xi1>
    %82 = tile.select %81, %cst_0, %80 : (tensor<1x28x28x512xi1>, tensor<f32>, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %83 = tile.pragma %82 "trace" {msg = "res4a"} : tensor<1x28x28x512xf32>
    %res4a_branch2a = tile.contract add, mul, %cst_0, %83, %arg25 {sink = #map0, srcs = [#map7, #map2]} : tensor<f32>, tensor<1x28x28x512xf32>, tensor<1x1x512x256xf32> -> tensor<1x14x14x256xf32>
    %84 = tile.add %res4a_branch2a, %arg79 : (tensor<1x14x14x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
    %85 = tile.cmp_lt %84, %cst_0 : (tensor<1x14x14x256xf32>, tensor<f32>) -> tensor<1x14x14x256xi1>
    %86 = tile.select %85, %cst_0, %84 : (tensor<1x14x14x256xi1>, tensor<f32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %res4a_branch2b = tile.contract add, mul, %cst_0, %86, %arg26 {sink = #map0, srcs = [#map6, #map2]} : tensor<f32>, tensor<1x14x14x256xf32>, tensor<3x3x256x256xf32> -> tensor<1x14x14x256xf32>
    %87 = tile.add %res4a_branch2b, %arg80 : (tensor<1x14x14x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
    %88 = tile.cmp_lt %87, %cst_0 : (tensor<1x14x14x256xf32>, tensor<f32>) -> tensor<1x14x14x256xi1>
    %89 = tile.select %88, %cst_0, %87 : (tensor<1x14x14x256xi1>, tensor<f32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %res4a_branch2b_8 = tile.contract add, mul, %cst_0, %89, %arg27 {sink = #map0, srcs = [#map5, #map2]} : tensor<f32>, tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32> -> tensor<1x14x14x1024xf32>
    %90 = tile.add %res4a_branch2b_8, %arg81 : (tensor<1x14x14x1024xf32>, tensor<1024xf32>) -> tensor<1x14x14x1024xf32>
    %res4a_branch1 = tile.contract add, mul, %cst_0, %83, %arg28 {sink = #map0, srcs = [#map7, #map2]} : tensor<f32>, tensor<1x28x28x512xf32>, tensor<1x1x512x1024xf32> -> tensor<1x14x14x1024xf32>
    %91 = tile.add %90, %res4a_branch1 : (tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %92 = tile.add %91, %arg82 : (tensor<1x14x14x1024xf32>, tensor<1024xf32>) -> tensor<1x14x14x1024xf32>
    %93 = tile.cmp_lt %92, %cst_0 : (tensor<1x14x14x1024xf32>, tensor<f32>) -> tensor<1x14x14x1024xi1>
    %94 = tile.select %93, %cst_0, %92 : (tensor<1x14x14x1024xi1>, tensor<f32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %95 = tile.pragma %94 "trace" {msg = "res4b"} : tensor<1x14x14x1024xf32>
    %res4b_branch2a = tile.contract add, mul, %cst_0, %95, %arg29 {sink = #map0, srcs = [#map5, #map2]} : tensor<f32>, tensor<1x14x14x1024xf32>, tensor<1x1x1024x256xf32> -> tensor<1x14x14x256xf32>
    %96 = tile.add %res4b_branch2a, %arg83 : (tensor<1x14x14x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
    %97 = tile.cmp_lt %96, %cst_0 : (tensor<1x14x14x256xf32>, tensor<f32>) -> tensor<1x14x14x256xi1>
    %98 = tile.select %97, %cst_0, %96 : (tensor<1x14x14x256xi1>, tensor<f32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %res4b_branch2b = tile.contract add, mul, %cst_0, %98, %arg30 {sink = #map0, srcs = [#map6, #map2]} : tensor<f32>, tensor<1x14x14x256xf32>, tensor<3x3x256x256xf32> -> tensor<1x14x14x256xf32>
    %99 = tile.add %res4b_branch2b, %arg84 : (tensor<1x14x14x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
    %100 = tile.cmp_lt %99, %cst_0 : (tensor<1x14x14x256xf32>, tensor<f32>) -> tensor<1x14x14x256xi1>
    %101 = tile.select %100, %cst_0, %99 : (tensor<1x14x14x256xi1>, tensor<f32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %res4b_branch2b_9 = tile.contract add, mul, %cst_0, %101, %arg31 {sink = #map0, srcs = [#map5, #map2]} : tensor<f32>, tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32> -> tensor<1x14x14x1024xf32>
    %102 = tile.add %res4b_branch2b_9, %arg85 : (tensor<1x14x14x1024xf32>, tensor<1024xf32>) -> tensor<1x14x14x1024xf32>
    %103 = tile.add %102, %95 : (tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %104 = tile.cmp_lt %103, %cst_0 : (tensor<1x14x14x1024xf32>, tensor<f32>) -> tensor<1x14x14x1024xi1>
    %105 = tile.select %104, %cst_0, %103 : (tensor<1x14x14x1024xi1>, tensor<f32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %106 = tile.pragma %105 "trace" {msg = "res4c"} : tensor<1x14x14x1024xf32>
    %res4c_branch2a = tile.contract add, mul, %cst_0, %106, %arg32 {sink = #map0, srcs = [#map5, #map2]} : tensor<f32>, tensor<1x14x14x1024xf32>, tensor<1x1x1024x256xf32> -> tensor<1x14x14x256xf32>
    %107 = tile.add %res4c_branch2a, %arg86 : (tensor<1x14x14x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
    %108 = tile.cmp_lt %107, %cst_0 : (tensor<1x14x14x256xf32>, tensor<f32>) -> tensor<1x14x14x256xi1>
    %109 = tile.select %108, %cst_0, %107 : (tensor<1x14x14x256xi1>, tensor<f32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %res4c_branch2b = tile.contract add, mul, %cst_0, %109, %arg33 {sink = #map0, srcs = [#map6, #map2]} : tensor<f32>, tensor<1x14x14x256xf32>, tensor<3x3x256x256xf32> -> tensor<1x14x14x256xf32>
    %110 = tile.add %res4c_branch2b, %arg87 : (tensor<1x14x14x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
    %111 = tile.cmp_lt %110, %cst_0 : (tensor<1x14x14x256xf32>, tensor<f32>) -> tensor<1x14x14x256xi1>
    %112 = tile.select %111, %cst_0, %110 : (tensor<1x14x14x256xi1>, tensor<f32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %res4c_branch2b_10 = tile.contract add, mul, %cst_0, %112, %arg34 {sink = #map0, srcs = [#map5, #map2]} : tensor<f32>, tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32> -> tensor<1x14x14x1024xf32>
    %113 = tile.add %res4c_branch2b_10, %arg88 : (tensor<1x14x14x1024xf32>, tensor<1024xf32>) -> tensor<1x14x14x1024xf32>
    %114 = tile.add %113, %106 : (tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %115 = tile.cmp_lt %114, %cst_0 : (tensor<1x14x14x1024xf32>, tensor<f32>) -> tensor<1x14x14x1024xi1>
    %116 = tile.select %115, %cst_0, %114 : (tensor<1x14x14x1024xi1>, tensor<f32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %117 = tile.pragma %116 "trace" {msg = "res4d"} : tensor<1x14x14x1024xf32>
    %res4d_branch2a = tile.contract add, mul, %cst_0, %117, %arg35 {sink = #map0, srcs = [#map5, #map2]} : tensor<f32>, tensor<1x14x14x1024xf32>, tensor<1x1x1024x256xf32> -> tensor<1x14x14x256xf32>
    %118 = tile.add %res4d_branch2a, %arg89 : (tensor<1x14x14x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
    %119 = tile.cmp_lt %118, %cst_0 : (tensor<1x14x14x256xf32>, tensor<f32>) -> tensor<1x14x14x256xi1>
    %120 = tile.select %119, %cst_0, %118 : (tensor<1x14x14x256xi1>, tensor<f32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %res4d_branch2b = tile.contract add, mul, %cst_0, %120, %arg36 {sink = #map0, srcs = [#map6, #map2]} : tensor<f32>, tensor<1x14x14x256xf32>, tensor<3x3x256x256xf32> -> tensor<1x14x14x256xf32>
    %121 = tile.add %res4d_branch2b, %arg90 : (tensor<1x14x14x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
    %122 = tile.cmp_lt %121, %cst_0 : (tensor<1x14x14x256xf32>, tensor<f32>) -> tensor<1x14x14x256xi1>
    %123 = tile.select %122, %cst_0, %121 : (tensor<1x14x14x256xi1>, tensor<f32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %res4d_branch2b_11 = tile.contract add, mul, %cst_0, %123, %arg37 {sink = #map0, srcs = [#map5, #map2]} : tensor<f32>, tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32> -> tensor<1x14x14x1024xf32>
    %124 = tile.add %res4d_branch2b_11, %arg91 : (tensor<1x14x14x1024xf32>, tensor<1024xf32>) -> tensor<1x14x14x1024xf32>
    %125 = tile.add %124, %117 : (tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %126 = tile.cmp_lt %125, %cst_0 : (tensor<1x14x14x1024xf32>, tensor<f32>) -> tensor<1x14x14x1024xi1>
    %127 = tile.select %126, %cst_0, %125 : (tensor<1x14x14x1024xi1>, tensor<f32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %128 = tile.pragma %127 "trace" {msg = "res4e"} : tensor<1x14x14x1024xf32>
    %res4e_branch2a = tile.contract add, mul, %cst_0, %128, %arg38 {sink = #map0, srcs = [#map5, #map2]} : tensor<f32>, tensor<1x14x14x1024xf32>, tensor<1x1x1024x256xf32> -> tensor<1x14x14x256xf32>
    %129 = tile.add %res4e_branch2a, %arg92 : (tensor<1x14x14x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
    %130 = tile.cmp_lt %129, %cst_0 : (tensor<1x14x14x256xf32>, tensor<f32>) -> tensor<1x14x14x256xi1>
    %131 = tile.select %130, %cst_0, %129 : (tensor<1x14x14x256xi1>, tensor<f32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %res4e_branch2b = tile.contract add, mul, %cst_0, %131, %arg39 {sink = #map0, srcs = [#map6, #map2]} : tensor<f32>, tensor<1x14x14x256xf32>, tensor<3x3x256x256xf32> -> tensor<1x14x14x256xf32>
    %132 = tile.add %res4e_branch2b, %arg93 : (tensor<1x14x14x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
    %133 = tile.cmp_lt %132, %cst_0 : (tensor<1x14x14x256xf32>, tensor<f32>) -> tensor<1x14x14x256xi1>
    %134 = tile.select %133, %cst_0, %132 : (tensor<1x14x14x256xi1>, tensor<f32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %res4e_branch2b_12 = tile.contract add, mul, %cst_0, %134, %arg40 {sink = #map0, srcs = [#map5, #map2]} : tensor<f32>, tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32> -> tensor<1x14x14x1024xf32>
    %135 = tile.add %res4e_branch2b_12, %arg94 : (tensor<1x14x14x1024xf32>, tensor<1024xf32>) -> tensor<1x14x14x1024xf32>
    %136 = tile.add %135, %128 : (tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %137 = tile.cmp_lt %136, %cst_0 : (tensor<1x14x14x1024xf32>, tensor<f32>) -> tensor<1x14x14x1024xi1>
    %138 = tile.select %137, %cst_0, %136 : (tensor<1x14x14x1024xi1>, tensor<f32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %139 = tile.pragma %138 "trace" {msg = "res4f"} : tensor<1x14x14x1024xf32>
    %res4f_branch2a = tile.contract add, mul, %cst_0, %139, %arg41 {sink = #map0, srcs = [#map5, #map2]} : tensor<f32>, tensor<1x14x14x1024xf32>, tensor<1x1x1024x256xf32> -> tensor<1x14x14x256xf32>
    %140 = tile.add %res4f_branch2a, %arg95 : (tensor<1x14x14x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
    %141 = tile.cmp_lt %140, %cst_0 : (tensor<1x14x14x256xf32>, tensor<f32>) -> tensor<1x14x14x256xi1>
    %142 = tile.select %141, %cst_0, %140 : (tensor<1x14x14x256xi1>, tensor<f32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %res4f_branch2b = tile.contract add, mul, %cst_0, %142, %arg42 {sink = #map0, srcs = [#map6, #map2]} : tensor<f32>, tensor<1x14x14x256xf32>, tensor<3x3x256x256xf32> -> tensor<1x14x14x256xf32>
    %143 = tile.add %res4f_branch2b, %arg96 : (tensor<1x14x14x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
    %144 = tile.cmp_lt %143, %cst_0 : (tensor<1x14x14x256xf32>, tensor<f32>) -> tensor<1x14x14x256xi1>
    %145 = tile.select %144, %cst_0, %143 : (tensor<1x14x14x256xi1>, tensor<f32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %res4f_branch2b_13 = tile.contract add, mul, %cst_0, %145, %arg43 {sink = #map0, srcs = [#map5, #map2]} : tensor<f32>, tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32> -> tensor<1x14x14x1024xf32>
    %146 = tile.add %res4f_branch2b_13, %arg97 : (tensor<1x14x14x1024xf32>, tensor<1024xf32>) -> tensor<1x14x14x1024xf32>
    %147 = tile.add %146, %139 : (tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %148 = tile.cmp_lt %147, %cst_0 : (tensor<1x14x14x1024xf32>, tensor<f32>) -> tensor<1x14x14x1024xi1>
    %149 = tile.select %148, %cst_0, %147 : (tensor<1x14x14x1024xi1>, tensor<f32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %150 = tile.pragma %149 "trace" {msg = "res5a"} : tensor<1x14x14x1024xf32>
    %res5a_branch2a = tile.contract add, mul, %cst_0, %150, %arg44 {sink = #map0, srcs = [#map7, #map2]} : tensor<f32>, tensor<1x14x14x1024xf32>, tensor<1x1x1024x512xf32> -> tensor<1x7x7x512xf32>
    %151 = tile.add %res5a_branch2a, %arg98 : (tensor<1x7x7x512xf32>, tensor<512xf32>) -> tensor<1x7x7x512xf32>
    %152 = tile.cmp_lt %151, %cst_0 : (tensor<1x7x7x512xf32>, tensor<f32>) -> tensor<1x7x7x512xi1>
    %153 = tile.select %152, %cst_0, %151 : (tensor<1x7x7x512xi1>, tensor<f32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %res5a_branch2b = tile.contract add, mul, %cst_0, %153, %arg45 {sink = #map0, srcs = [#map6, #map2]} : tensor<f32>, tensor<1x7x7x512xf32>, tensor<3x3x512x512xf32> -> tensor<1x7x7x512xf32>
    %154 = tile.add %res5a_branch2b, %arg99 : (tensor<1x7x7x512xf32>, tensor<512xf32>) -> tensor<1x7x7x512xf32>
    %155 = tile.cmp_lt %154, %cst_0 : (tensor<1x7x7x512xf32>, tensor<f32>) -> tensor<1x7x7x512xi1>
    %156 = tile.select %155, %cst_0, %154 : (tensor<1x7x7x512xi1>, tensor<f32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %res5a_branch2b_14 = tile.contract add, mul, %cst_0, %156, %arg46 {sink = #map0, srcs = [#map5, #map2]} : tensor<f32>, tensor<1x7x7x512xf32>, tensor<1x1x512x2048xf32> -> tensor<1x7x7x2048xf32>
    %157 = tile.add %res5a_branch2b_14, %arg100 : (tensor<1x7x7x2048xf32>, tensor<2048xf32>) -> tensor<1x7x7x2048xf32>
    %res5a_branch1 = tile.contract add, mul, %cst_0, %150, %arg47 {sink = #map0, srcs = [#map7, #map2]} : tensor<f32>, tensor<1x14x14x1024xf32>, tensor<1x1x1024x2048xf32> -> tensor<1x7x7x2048xf32>
    %158 = tile.add %157, %res5a_branch1 : (tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    %159 = tile.add %158, %arg101 : (tensor<1x7x7x2048xf32>, tensor<2048xf32>) -> tensor<1x7x7x2048xf32>
    %160 = tile.cmp_lt %159, %cst_0 : (tensor<1x7x7x2048xf32>, tensor<f32>) -> tensor<1x7x7x2048xi1>
    %161 = tile.select %160, %cst_0, %159 : (tensor<1x7x7x2048xi1>, tensor<f32>, tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    %162 = tile.pragma %161 "trace" {msg = "res5b"} : tensor<1x7x7x2048xf32>
    %res5b_branch2a = tile.contract add, mul, %cst_0, %162, %arg48 {sink = #map0, srcs = [#map5, #map2]} : tensor<f32>, tensor<1x7x7x2048xf32>, tensor<1x1x2048x512xf32> -> tensor<1x7x7x512xf32>
    %163 = tile.add %res5b_branch2a, %arg102 : (tensor<1x7x7x512xf32>, tensor<512xf32>) -> tensor<1x7x7x512xf32>
    %164 = tile.cmp_lt %163, %cst_0 : (tensor<1x7x7x512xf32>, tensor<f32>) -> tensor<1x7x7x512xi1>
    %165 = tile.select %164, %cst_0, %163 : (tensor<1x7x7x512xi1>, tensor<f32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %res5b_branch2b = tile.contract add, mul, %cst_0, %165, %arg49 {sink = #map0, srcs = [#map6, #map2]} : tensor<f32>, tensor<1x7x7x512xf32>, tensor<3x3x512x512xf32> -> tensor<1x7x7x512xf32>
    %166 = tile.add %res5b_branch2b, %arg103 : (tensor<1x7x7x512xf32>, tensor<512xf32>) -> tensor<1x7x7x512xf32>
    %167 = tile.cmp_lt %166, %cst_0 : (tensor<1x7x7x512xf32>, tensor<f32>) -> tensor<1x7x7x512xi1>
    %168 = tile.select %167, %cst_0, %166 : (tensor<1x7x7x512xi1>, tensor<f32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %res5b_branch2b_15 = tile.contract add, mul, %cst_0, %168, %arg50 {sink = #map0, srcs = [#map5, #map2]} : tensor<f32>, tensor<1x7x7x512xf32>, tensor<1x1x512x2048xf32> -> tensor<1x7x7x2048xf32>
    %169 = tile.add %res5b_branch2b_15, %arg104 : (tensor<1x7x7x2048xf32>, tensor<2048xf32>) -> tensor<1x7x7x2048xf32>
    %170 = tile.add %169, %162 : (tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    %171 = tile.cmp_lt %170, %cst_0 : (tensor<1x7x7x2048xf32>, tensor<f32>) -> tensor<1x7x7x2048xi1>
    %172 = tile.select %171, %cst_0, %170 : (tensor<1x7x7x2048xi1>, tensor<f32>, tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    %173 = tile.pragma %172 "trace" {msg = "res5c"} : tensor<1x7x7x2048xf32>
    %res5c_branch2a = tile.contract add, mul, %cst_0, %173, %arg51 {sink = #map0, srcs = [#map5, #map2]} : tensor<f32>, tensor<1x7x7x2048xf32>, tensor<1x1x2048x512xf32> -> tensor<1x7x7x512xf32>
    %174 = tile.add %res5c_branch2a, %arg105 : (tensor<1x7x7x512xf32>, tensor<512xf32>) -> tensor<1x7x7x512xf32>
    %175 = tile.cmp_lt %174, %cst_0 : (tensor<1x7x7x512xf32>, tensor<f32>) -> tensor<1x7x7x512xi1>
    %176 = tile.select %175, %cst_0, %174 : (tensor<1x7x7x512xi1>, tensor<f32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %res5c_branch2b = tile.contract add, mul, %cst_0, %176, %arg52 {sink = #map0, srcs = [#map6, #map2]} : tensor<f32>, tensor<1x7x7x512xf32>, tensor<3x3x512x512xf32> -> tensor<1x7x7x512xf32>
    %177 = tile.add %res5c_branch2b, %arg106 : (tensor<1x7x7x512xf32>, tensor<512xf32>) -> tensor<1x7x7x512xf32>
    %178 = tile.cmp_lt %177, %cst_0 : (tensor<1x7x7x512xf32>, tensor<f32>) -> tensor<1x7x7x512xi1>
    %179 = tile.select %178, %cst_0, %177 : (tensor<1x7x7x512xi1>, tensor<f32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %res5c_branch2b_16 = tile.contract add, mul, %cst_0, %179, %arg53 {sink = #map0, srcs = [#map5, #map2]} : tensor<f32>, tensor<1x7x7x512xf32>, tensor<1x1x512x2048xf32> -> tensor<1x7x7x2048xf32>
    %180 = tile.add %res5c_branch2b_16, %arg107 : (tensor<1x7x7x2048xf32>, tensor<2048xf32>) -> tensor<1x7x7x2048xf32>
    %181 = tile.add %180, %173 : (tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    %182 = tile.cmp_lt %181, %cst_0 : (tensor<1x7x7x2048xf32>, tensor<f32>) -> tensor<1x7x7x2048xi1>
    %183 = tile.select %182, %cst_0, %181 : (tensor<1x7x7x2048xi1>, tensor<f32>, tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    %184 = tile.contract add, none, %cst_0, %183 {sink = #map8, srcs = [#map9]} : tensor<f32>, tensor<1x7x7x2048xf32> -> tensor<1x2048xf32>
    %185 = tile.div %184, %c49 : (tensor<1x2048xf32>, tensor<si64>) -> tensor<1x2048xf32>
    %186 = tile.contract add, mul, %cst_0, %185, %arg54 {sink = #map10, srcs = [#map11, #map12]} : tensor<f32>, tensor<1x2048xf32>, tensor<2048x1000xf32> -> tensor<1x1000xf32>
    %187 = tile.add %186, %arg108 : (tensor<1x1000xf32>, tensor<1000xf32>) -> tensor<1x1000xf32>
    %188 = tile.contract max, none, %cst, %187 {sink = #map13, srcs = [#map14]} : tensor<f32>, tensor<1x1000xf32> -> tensor<1x1xf32>
    %189 = tile.sub %187, %188 : (tensor<1x1000xf32>, tensor<1x1xf32>) -> tensor<1x1000xf32>
    %190 = tile.exp %189 : (tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %191 = tile.contract add, none, %cst_0, %190 {sink = #map13, srcs = [#map14]} : tensor<f32>, tensor<1x1000xf32> -> tensor<1x1xf32>
    %192 = tile.div %190, %191 : (tensor<1x1000xf32>, tensor<1x1xf32>) -> tensor<1x1000xf32>
    %193 = tile.pragma %192 "trace" {msg = "done"} : tensor<1x1000xf32>
    return %193 : tensor<1x1000xf32>
  }
}
