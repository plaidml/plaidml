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

module {
  func @resnet50(%arg0: tensor<1000xf32>, %arg1: tensor<2048x1000xf32>, %arg2: tensor<2048xf32>, %arg3: tensor<1x1x1024x2048xf32>, %arg4: tensor<1024xf32>, %arg5: tensor<1x1x512x1024xf32>, %arg6: tensor<512xf32>, %arg7: tensor<1x1x256x512xf32>, %arg8: tensor<256xf32>, %arg9: tensor<1x1x64x256xf32>, %arg10: tensor<64xf32>, %arg11: tensor<7x7x3x64xf32>, %arg12: tensor<1x224x224x3xf32>, %arg13: tensor<256xf32>, %arg14: tensor<1x1x64x256xf32>, %arg15: tensor<64xf32>, %arg16: tensor<3x3x64x64xf32>, %arg17: tensor<64xf32>, %arg18: tensor<1x1x64x64xf32>, %arg19: tensor<256xf32>, %arg20: tensor<1x1x64x256xf32>, %arg21: tensor<64xf32>, %arg22: tensor<3x3x64x64xf32>, %arg23: tensor<64xf32>, %arg24: tensor<1x1x256x64xf32>, %arg25: tensor<256xf32>, %arg26: tensor<1x1x64x256xf32>, %arg27: tensor<64xf32>, %arg28: tensor<3x3x64x64xf32>, %arg29: tensor<64xf32>, %arg30: tensor<1x1x256x64xf32>, %arg31: tensor<512xf32>, %arg32: tensor<1x1x128x512xf32>, %arg33: tensor<128xf32>, %arg34: tensor<3x3x128x128xf32>, %arg35: tensor<128xf32>, %arg36: tensor<1x1x256x128xf32>, %arg37: tensor<512xf32>, %arg38: tensor<1x1x128x512xf32>, %arg39: tensor<128xf32>, %arg40: tensor<3x3x128x128xf32>, %arg41: tensor<128xf32>, %arg42: tensor<1x1x512x128xf32>, %arg43: tensor<512xf32>, %arg44: tensor<1x1x128x512xf32>, %arg45: tensor<128xf32>, %arg46: tensor<3x3x128x128xf32>, %arg47: tensor<128xf32>, %arg48: tensor<1x1x512x128xf32>, %arg49: tensor<512xf32>, %arg50: tensor<1x1x128x512xf32>, %arg51: tensor<128xf32>, %arg52: tensor<3x3x128x128xf32>, %arg53: tensor<128xf32>, %arg54: tensor<1x1x512x128xf32>, %arg55: tensor<1024xf32>, %arg56: tensor<1x1x256x1024xf32>, %arg57: tensor<256xf32>, %arg58: tensor<3x3x256x256xf32>, %arg59: tensor<256xf32>, %arg60: tensor<1x1x512x256xf32>, %arg61: tensor<1024xf32>, %arg62: tensor<1x1x256x1024xf32>, %arg63: tensor<256xf32>, %arg64: tensor<3x3x256x256xf32>, %arg65: tensor<256xf32>, %arg66: tensor<1x1x1024x256xf32>, %arg67: tensor<1024xf32>, %arg68: tensor<1x1x256x1024xf32>, %arg69: tensor<256xf32>, %arg70: tensor<3x3x256x256xf32>, %arg71: tensor<256xf32>, %arg72: tensor<1x1x1024x256xf32>, %arg73: tensor<1024xf32>, %arg74: tensor<1x1x256x1024xf32>, %arg75: tensor<256xf32>, %arg76: tensor<3x3x256x256xf32>, %arg77: tensor<256xf32>, %arg78: tensor<1x1x1024x256xf32>, %arg79: tensor<1024xf32>, %arg80: tensor<1x1x256x1024xf32>, %arg81: tensor<256xf32>, %arg82: tensor<3x3x256x256xf32>, %arg83: tensor<256xf32>, %arg84: tensor<1x1x1024x256xf32>, %arg85: tensor<1024xf32>, %arg86: tensor<1x1x256x1024xf32>, %arg87: tensor<256xf32>, %arg88: tensor<3x3x256x256xf32>, %arg89: tensor<256xf32>, %arg90: tensor<1x1x1024x256xf32>, %arg91: tensor<2048xf32>, %arg92: tensor<1x1x512x2048xf32>, %arg93: tensor<512xf32>, %arg94: tensor<3x3x512x512xf32>, %arg95: tensor<512xf32>, %arg96: tensor<1x1x1024x512xf32>, %arg97: tensor<2048xf32>, %arg98: tensor<1x1x512x2048xf32>, %arg99: tensor<512xf32>, %arg100: tensor<3x3x512x512xf32>, %arg101: tensor<512xf32>, %arg102: tensor<1x1x2048x512xf32>, %arg103: tensor<2048xf32>, %arg104: tensor<1x1x512x2048xf32>, %arg105: tensor<512xf32>, %arg106: tensor<3x3x512x512xf32>, %arg107: tensor<512xf32>, %arg108: tensor<1x1x2048x512xf32>) -> tensor<1x1000xf32> {
    %c49 = "eltwise.sconst"() {value = 49 : i64} : () -> si32
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %conv1 = tile.contract add, mul, %cst, %arg12, %arg11 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map1, #map2]} : f32, tensor<1x224x224x3xf32>, tensor<7x7x3x64xf32> -> tensor<1x112x112x64xf32>
    %0 = "eltwise.add"(%conv1, %arg10) : (tensor<1x112x112x64xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>
    %1 = "eltwise.cmp_lt"(%0, %cst) : (tensor<1x112x112x64xf32>, f32) -> tensor<1x112x112x64xi1>
    %2 = "eltwise.select"(%1, %cst, %0) : (tensor<1x112x112x64xi1>, f32, tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
    %3 = tile.contract max, none, %cst, %2 {cons = #set0, sink = #map3, srcs = [#map4]} : f32, tensor<1x112x112x64xf32> -> tensor<1x56x56x64xf32>
    %4 = "tile.trace"(%3) {msg = "res2a"} : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %res2a_branch1 = tile.contract add, mul, %cst, %4, %arg9 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : f32, tensor<1x56x56x64xf32>, tensor<1x1x64x256xf32> -> tensor<1x56x56x256xf32>
    %res2a_branch2a = tile.contract add, mul, %cst, %4, %arg18 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : f32, tensor<1x56x56x64xf32>, tensor<1x1x64x64xf32> -> tensor<1x56x56x64xf32>
    %5 = "eltwise.add"(%res2a_branch2a, %arg17) : (tensor<1x56x56x64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
    %6 = "eltwise.cmp_lt"(%5, %cst) : (tensor<1x56x56x64xf32>, f32) -> tensor<1x56x56x64xi1>
    %7 = "eltwise.select"(%6, %cst, %5) : (tensor<1x56x56x64xi1>, f32, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %res2a_branch2b = tile.contract add, mul, %cst, %7, %arg16 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : f32, tensor<1x56x56x64xf32>, tensor<3x3x64x64xf32> -> tensor<1x56x56x64xf32>
    %8 = "eltwise.add"(%res2a_branch2b, %arg15) : (tensor<1x56x56x64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
    %9 = "eltwise.cmp_lt"(%8, %cst) : (tensor<1x56x56x64xf32>, f32) -> tensor<1x56x56x64xi1>
    %10 = "eltwise.select"(%9, %cst, %8) : (tensor<1x56x56x64xi1>, f32, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %res2a_branch2c = tile.contract add, mul, %cst, %10, %arg14 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : f32, tensor<1x56x56x64xf32>, tensor<1x1x64x256xf32> -> tensor<1x56x56x256xf32>
    %11 = "eltwise.add"(%res2a_branch2c, %arg13) : (tensor<1x56x56x256xf32>, tensor<256xf32>) -> tensor<1x56x56x256xf32>
    %12 = "eltwise.add"(%11, %res2a_branch1) : (tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %13 = "eltwise.add"(%12, %arg8) : (tensor<1x56x56x256xf32>, tensor<256xf32>) -> tensor<1x56x56x256xf32>
    %14 = "eltwise.cmp_lt"(%13, %cst) : (tensor<1x56x56x256xf32>, f32) -> tensor<1x56x56x256xi1>
    %15 = "eltwise.select"(%14, %cst, %13) : (tensor<1x56x56x256xi1>, f32, tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %16 = "tile.trace"(%15) {msg = "res2b"} : (tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %res2b_branch2a = tile.contract add, mul, %cst, %16, %arg24 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : f32, tensor<1x56x56x256xf32>, tensor<1x1x256x64xf32> -> tensor<1x56x56x64xf32>
    %17 = "eltwise.add"(%res2b_branch2a, %arg23) : (tensor<1x56x56x64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
    %18 = "eltwise.cmp_lt"(%17, %cst) : (tensor<1x56x56x64xf32>, f32) -> tensor<1x56x56x64xi1>
    %19 = "eltwise.select"(%18, %cst, %17) : (tensor<1x56x56x64xi1>, f32, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %res2b_branch2b = tile.contract add, mul, %cst, %19, %arg22 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : f32, tensor<1x56x56x64xf32>, tensor<3x3x64x64xf32> -> tensor<1x56x56x64xf32>
    %20 = "eltwise.add"(%res2b_branch2b, %arg21) : (tensor<1x56x56x64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
    %21 = "eltwise.cmp_lt"(%20, %cst) : (tensor<1x56x56x64xf32>, f32) -> tensor<1x56x56x64xi1>
    %22 = "eltwise.select"(%21, %cst, %20) : (tensor<1x56x56x64xi1>, f32, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %res2b_branch2c = tile.contract add, mul, %cst, %22, %arg20 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : f32, tensor<1x56x56x64xf32>, tensor<1x1x64x256xf32> -> tensor<1x56x56x256xf32>
    %23 = "eltwise.add"(%res2b_branch2c, %arg19) : (tensor<1x56x56x256xf32>, tensor<256xf32>) -> tensor<1x56x56x256xf32>
    %24 = "eltwise.add"(%23, %16) : (tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %25 = "eltwise.cmp_lt"(%24, %cst) : (tensor<1x56x56x256xf32>, f32) -> tensor<1x56x56x256xi1>
    %26 = "eltwise.select"(%25, %cst, %24) : (tensor<1x56x56x256xi1>, f32, tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %27 = "tile.trace"(%26) {msg = "res2c"} : (tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %res2c_branch2a = tile.contract add, mul, %cst, %27, %arg30 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : f32, tensor<1x56x56x256xf32>, tensor<1x1x256x64xf32> -> tensor<1x56x56x64xf32>
    %28 = "eltwise.add"(%res2c_branch2a, %arg29) : (tensor<1x56x56x64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
    %29 = "eltwise.cmp_lt"(%28, %cst) : (tensor<1x56x56x64xf32>, f32) -> tensor<1x56x56x64xi1>
    %30 = "eltwise.select"(%29, %cst, %28) : (tensor<1x56x56x64xi1>, f32, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %res2c_branch2b = tile.contract add, mul, %cst, %30, %arg28 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : f32, tensor<1x56x56x64xf32>, tensor<3x3x64x64xf32> -> tensor<1x56x56x64xf32>
    %31 = "eltwise.add"(%res2c_branch2b, %arg27) : (tensor<1x56x56x64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
    %32 = "eltwise.cmp_lt"(%31, %cst) : (tensor<1x56x56x64xf32>, f32) -> tensor<1x56x56x64xi1>
    %33 = "eltwise.select"(%32, %cst, %31) : (tensor<1x56x56x64xi1>, f32, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %res2c_branch2c = tile.contract add, mul, %cst, %33, %arg26 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : f32, tensor<1x56x56x64xf32>, tensor<1x1x64x256xf32> -> tensor<1x56x56x256xf32>
    %34 = "eltwise.add"(%res2c_branch2c, %arg25) : (tensor<1x56x56x256xf32>, tensor<256xf32>) -> tensor<1x56x56x256xf32>
    %35 = "eltwise.add"(%34, %27) : (tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %36 = "eltwise.cmp_lt"(%35, %cst) : (tensor<1x56x56x256xf32>, f32) -> tensor<1x56x56x256xi1>
    %37 = "eltwise.select"(%36, %cst, %35) : (tensor<1x56x56x256xi1>, f32, tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %38 = "tile.trace"(%37) {msg = "res3a"} : (tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %res3a_branch1 = tile.contract add, mul, %cst, %38, %arg7 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map7, #map2]} : f32, tensor<1x56x56x256xf32>, tensor<1x1x256x512xf32> -> tensor<1x28x28x512xf32>
    %res3a_branch2a = tile.contract add, mul, %cst, %38, %arg36 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map7, #map2]} : f32, tensor<1x56x56x256xf32>, tensor<1x1x256x128xf32> -> tensor<1x28x28x128xf32>
    %39 = "eltwise.add"(%res3a_branch2a, %arg35) : (tensor<1x28x28x128xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
    %40 = "eltwise.cmp_lt"(%39, %cst) : (tensor<1x28x28x128xf32>, f32) -> tensor<1x28x28x128xi1>
    %41 = "eltwise.select"(%40, %cst, %39) : (tensor<1x28x28x128xi1>, f32, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %res3a_branch2b = tile.contract add, mul, %cst, %41, %arg34 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : f32, tensor<1x28x28x128xf32>, tensor<3x3x128x128xf32> -> tensor<1x28x28x128xf32>
    %42 = "eltwise.add"(%res3a_branch2b, %arg33) : (tensor<1x28x28x128xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
    %43 = "eltwise.cmp_lt"(%42, %cst) : (tensor<1x28x28x128xf32>, f32) -> tensor<1x28x28x128xi1>
    %44 = "eltwise.select"(%43, %cst, %42) : (tensor<1x28x28x128xi1>, f32, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %res3a_branch2c = tile.contract add, mul, %cst, %44, %arg32 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : f32, tensor<1x28x28x128xf32>, tensor<1x1x128x512xf32> -> tensor<1x28x28x512xf32>
    %45 = "eltwise.add"(%res3a_branch2c, %arg31) : (tensor<1x28x28x512xf32>, tensor<512xf32>) -> tensor<1x28x28x512xf32>
    %46 = "eltwise.add"(%45, %res3a_branch1) : (tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %47 = "eltwise.add"(%46, %arg6) : (tensor<1x28x28x512xf32>, tensor<512xf32>) -> tensor<1x28x28x512xf32>
    %48 = "eltwise.cmp_lt"(%47, %cst) : (tensor<1x28x28x512xf32>, f32) -> tensor<1x28x28x512xi1>
    %49 = "eltwise.select"(%48, %cst, %47) : (tensor<1x28x28x512xi1>, f32, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %50 = "tile.trace"(%49) {msg = "res3b"} : (tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %res3b_branch2a = tile.contract add, mul, %cst, %50, %arg42 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : f32, tensor<1x28x28x512xf32>, tensor<1x1x512x128xf32> -> tensor<1x28x28x128xf32>
    %51 = "eltwise.add"(%res3b_branch2a, %arg41) : (tensor<1x28x28x128xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
    %52 = "eltwise.cmp_lt"(%51, %cst) : (tensor<1x28x28x128xf32>, f32) -> tensor<1x28x28x128xi1>
    %53 = "eltwise.select"(%52, %cst, %51) : (tensor<1x28x28x128xi1>, f32, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %res3b_branch2b = tile.contract add, mul, %cst, %53, %arg40 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : f32, tensor<1x28x28x128xf32>, tensor<3x3x128x128xf32> -> tensor<1x28x28x128xf32>
    %54 = "eltwise.add"(%res3b_branch2b, %arg39) : (tensor<1x28x28x128xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
    %55 = "eltwise.cmp_lt"(%54, %cst) : (tensor<1x28x28x128xf32>, f32) -> tensor<1x28x28x128xi1>
    %56 = "eltwise.select"(%55, %cst, %54) : (tensor<1x28x28x128xi1>, f32, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %res3b_branch2c = tile.contract add, mul, %cst, %56, %arg38 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : f32, tensor<1x28x28x128xf32>, tensor<1x1x128x512xf32> -> tensor<1x28x28x512xf32>
    %57 = "eltwise.add"(%res3b_branch2c, %arg37) : (tensor<1x28x28x512xf32>, tensor<512xf32>) -> tensor<1x28x28x512xf32>
    %58 = "eltwise.add"(%57, %50) : (tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %59 = "eltwise.cmp_lt"(%58, %cst) : (tensor<1x28x28x512xf32>, f32) -> tensor<1x28x28x512xi1>
    %60 = "eltwise.select"(%59, %cst, %58) : (tensor<1x28x28x512xi1>, f32, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %61 = "tile.trace"(%60) {msg = "res3c"} : (tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %res3c_branch2a = tile.contract add, mul, %cst, %61, %arg48 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : f32, tensor<1x28x28x512xf32>, tensor<1x1x512x128xf32> -> tensor<1x28x28x128xf32>
    %62 = "eltwise.add"(%res3c_branch2a, %arg47) : (tensor<1x28x28x128xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
    %63 = "eltwise.cmp_lt"(%62, %cst) : (tensor<1x28x28x128xf32>, f32) -> tensor<1x28x28x128xi1>
    %64 = "eltwise.select"(%63, %cst, %62) : (tensor<1x28x28x128xi1>, f32, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %res3c_branch2b = tile.contract add, mul, %cst, %64, %arg46 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : f32, tensor<1x28x28x128xf32>, tensor<3x3x128x128xf32> -> tensor<1x28x28x128xf32>
    %65 = "eltwise.add"(%res3c_branch2b, %arg45) : (tensor<1x28x28x128xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
    %66 = "eltwise.cmp_lt"(%65, %cst) : (tensor<1x28x28x128xf32>, f32) -> tensor<1x28x28x128xi1>
    %67 = "eltwise.select"(%66, %cst, %65) : (tensor<1x28x28x128xi1>, f32, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %res3c_branch2c = tile.contract add, mul, %cst, %67, %arg44 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : f32, tensor<1x28x28x128xf32>, tensor<1x1x128x512xf32> -> tensor<1x28x28x512xf32>
    %68 = "eltwise.add"(%res3c_branch2c, %arg43) : (tensor<1x28x28x512xf32>, tensor<512xf32>) -> tensor<1x28x28x512xf32>
    %69 = "eltwise.add"(%68, %61) : (tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %70 = "eltwise.cmp_lt"(%69, %cst) : (tensor<1x28x28x512xf32>, f32) -> tensor<1x28x28x512xi1>
    %71 = "eltwise.select"(%70, %cst, %69) : (tensor<1x28x28x512xi1>, f32, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %72 = "tile.trace"(%71) {msg = "res3d"} : (tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %res3d_branch2a = tile.contract add, mul, %cst, %72, %arg54 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : f32, tensor<1x28x28x512xf32>, tensor<1x1x512x128xf32> -> tensor<1x28x28x128xf32>
    %73 = "eltwise.add"(%res3d_branch2a, %arg53) : (tensor<1x28x28x128xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
    %74 = "eltwise.cmp_lt"(%73, %cst) : (tensor<1x28x28x128xf32>, f32) -> tensor<1x28x28x128xi1>
    %75 = "eltwise.select"(%74, %cst, %73) : (tensor<1x28x28x128xi1>, f32, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %res3d_branch2b = tile.contract add, mul, %cst, %75, %arg52 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : f32, tensor<1x28x28x128xf32>, tensor<3x3x128x128xf32> -> tensor<1x28x28x128xf32>
    %76 = "eltwise.add"(%res3d_branch2b, %arg51) : (tensor<1x28x28x128xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
    %77 = "eltwise.cmp_lt"(%76, %cst) : (tensor<1x28x28x128xf32>, f32) -> tensor<1x28x28x128xi1>
    %78 = "eltwise.select"(%77, %cst, %76) : (tensor<1x28x28x128xi1>, f32, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %res3d_branch2c = tile.contract add, mul, %cst, %78, %arg50 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : f32, tensor<1x28x28x128xf32>, tensor<1x1x128x512xf32> -> tensor<1x28x28x512xf32>
    %79 = "eltwise.add"(%res3d_branch2c, %arg49) : (tensor<1x28x28x512xf32>, tensor<512xf32>) -> tensor<1x28x28x512xf32>
    %80 = "eltwise.add"(%79, %72) : (tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %81 = "eltwise.cmp_lt"(%80, %cst) : (tensor<1x28x28x512xf32>, f32) -> tensor<1x28x28x512xi1>
    %82 = "eltwise.select"(%81, %cst, %80) : (tensor<1x28x28x512xi1>, f32, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %83 = "tile.trace"(%82) {msg = "res4a"} : (tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %res4a_branch1 = tile.contract add, mul, %cst, %83, %arg5 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map7, #map2]} : f32, tensor<1x28x28x512xf32>, tensor<1x1x512x1024xf32> -> tensor<1x14x14x1024xf32>
    %res4a_branch2a = tile.contract add, mul, %cst, %83, %arg60 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map7, #map2]} : f32, tensor<1x28x28x512xf32>, tensor<1x1x512x256xf32> -> tensor<1x14x14x256xf32>
    %84 = "eltwise.add"(%res4a_branch2a, %arg59) : (tensor<1x14x14x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
    %85 = "eltwise.cmp_lt"(%84, %cst) : (tensor<1x14x14x256xf32>, f32) -> tensor<1x14x14x256xi1>
    %86 = "eltwise.select"(%85, %cst, %84) : (tensor<1x14x14x256xi1>, f32, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %res4a_branch2b = tile.contract add, mul, %cst, %86, %arg58 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : f32, tensor<1x14x14x256xf32>, tensor<3x3x256x256xf32> -> tensor<1x14x14x256xf32>
    %87 = "eltwise.add"(%res4a_branch2b, %arg57) : (tensor<1x14x14x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
    %88 = "eltwise.cmp_lt"(%87, %cst) : (tensor<1x14x14x256xf32>, f32) -> tensor<1x14x14x256xi1>
    %89 = "eltwise.select"(%88, %cst, %87) : (tensor<1x14x14x256xi1>, f32, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %res4a_branch2c = tile.contract add, mul, %cst, %89, %arg56 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : f32, tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32> -> tensor<1x14x14x1024xf32>
    %90 = "eltwise.add"(%res4a_branch2c, %arg55) : (tensor<1x14x14x1024xf32>, tensor<1024xf32>) -> tensor<1x14x14x1024xf32>
    %91 = "eltwise.add"(%90, %res4a_branch1) : (tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %92 = "eltwise.add"(%91, %arg4) : (tensor<1x14x14x1024xf32>, tensor<1024xf32>) -> tensor<1x14x14x1024xf32>
    %93 = "eltwise.cmp_lt"(%92, %cst) : (tensor<1x14x14x1024xf32>, f32) -> tensor<1x14x14x1024xi1>
    %94 = "eltwise.select"(%93, %cst, %92) : (tensor<1x14x14x1024xi1>, f32, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %95 = "tile.trace"(%94) {msg = "res4b"} : (tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %res4b_branch2a = tile.contract add, mul, %cst, %95, %arg66 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : f32, tensor<1x14x14x1024xf32>, tensor<1x1x1024x256xf32> -> tensor<1x14x14x256xf32>
    %96 = "eltwise.add"(%res4b_branch2a, %arg65) : (tensor<1x14x14x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
    %97 = "eltwise.cmp_lt"(%96, %cst) : (tensor<1x14x14x256xf32>, f32) -> tensor<1x14x14x256xi1>
    %98 = "eltwise.select"(%97, %cst, %96) : (tensor<1x14x14x256xi1>, f32, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %res4b_branch2b = tile.contract add, mul, %cst, %98, %arg64 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : f32, tensor<1x14x14x256xf32>, tensor<3x3x256x256xf32> -> tensor<1x14x14x256xf32>
    %99 = "eltwise.add"(%res4b_branch2b, %arg63) : (tensor<1x14x14x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
    %100 = "eltwise.cmp_lt"(%99, %cst) : (tensor<1x14x14x256xf32>, f32) -> tensor<1x14x14x256xi1>
    %101 = "eltwise.select"(%100, %cst, %99) : (tensor<1x14x14x256xi1>, f32, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %res4b_branch2c = tile.contract add, mul, %cst, %101, %arg62 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : f32, tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32> -> tensor<1x14x14x1024xf32>
    %102 = "eltwise.add"(%res4b_branch2c, %arg61) : (tensor<1x14x14x1024xf32>, tensor<1024xf32>) -> tensor<1x14x14x1024xf32>
    %103 = "eltwise.add"(%102, %95) : (tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %104 = "eltwise.cmp_lt"(%103, %cst) : (tensor<1x14x14x1024xf32>, f32) -> tensor<1x14x14x1024xi1>
    %105 = "eltwise.select"(%104, %cst, %103) : (tensor<1x14x14x1024xi1>, f32, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %106 = "tile.trace"(%105) {msg = "res4c"} : (tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %res4c_branch2a = tile.contract add, mul, %cst, %106, %arg72 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : f32, tensor<1x14x14x1024xf32>, tensor<1x1x1024x256xf32> -> tensor<1x14x14x256xf32>
    %107 = "eltwise.add"(%res4c_branch2a, %arg71) : (tensor<1x14x14x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
    %108 = "eltwise.cmp_lt"(%107, %cst) : (tensor<1x14x14x256xf32>, f32) -> tensor<1x14x14x256xi1>
    %109 = "eltwise.select"(%108, %cst, %107) : (tensor<1x14x14x256xi1>, f32, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %res4c_branch2b = tile.contract add, mul, %cst, %109, %arg70 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : f32, tensor<1x14x14x256xf32>, tensor<3x3x256x256xf32> -> tensor<1x14x14x256xf32>
    %110 = "eltwise.add"(%res4c_branch2b, %arg69) : (tensor<1x14x14x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
    %111 = "eltwise.cmp_lt"(%110, %cst) : (tensor<1x14x14x256xf32>, f32) -> tensor<1x14x14x256xi1>
    %112 = "eltwise.select"(%111, %cst, %110) : (tensor<1x14x14x256xi1>, f32, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %res4c_branch2c = tile.contract add, mul, %cst, %112, %arg68 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : f32, tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32> -> tensor<1x14x14x1024xf32>
    %113 = "eltwise.add"(%res4c_branch2c, %arg67) : (tensor<1x14x14x1024xf32>, tensor<1024xf32>) -> tensor<1x14x14x1024xf32>
    %114 = "eltwise.add"(%113, %106) : (tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %115 = "eltwise.cmp_lt"(%114, %cst) : (tensor<1x14x14x1024xf32>, f32) -> tensor<1x14x14x1024xi1>
    %116 = "eltwise.select"(%115, %cst, %114) : (tensor<1x14x14x1024xi1>, f32, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %117 = "tile.trace"(%116) {msg = "res4d"} : (tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %res4d_branch2a = tile.contract add, mul, %cst, %117, %arg78 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : f32, tensor<1x14x14x1024xf32>, tensor<1x1x1024x256xf32> -> tensor<1x14x14x256xf32>
    %118 = "eltwise.add"(%res4d_branch2a, %arg77) : (tensor<1x14x14x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
    %119 = "eltwise.cmp_lt"(%118, %cst) : (tensor<1x14x14x256xf32>, f32) -> tensor<1x14x14x256xi1>
    %120 = "eltwise.select"(%119, %cst, %118) : (tensor<1x14x14x256xi1>, f32, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %res4d_branch2b = tile.contract add, mul, %cst, %120, %arg76 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : f32, tensor<1x14x14x256xf32>, tensor<3x3x256x256xf32> -> tensor<1x14x14x256xf32>
    %121 = "eltwise.add"(%res4d_branch2b, %arg75) : (tensor<1x14x14x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
    %122 = "eltwise.cmp_lt"(%121, %cst) : (tensor<1x14x14x256xf32>, f32) -> tensor<1x14x14x256xi1>
    %123 = "eltwise.select"(%122, %cst, %121) : (tensor<1x14x14x256xi1>, f32, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %res4d_branch2c = tile.contract add, mul, %cst, %123, %arg74 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : f32, tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32> -> tensor<1x14x14x1024xf32>
    %124 = "eltwise.add"(%res4d_branch2c, %arg73) : (tensor<1x14x14x1024xf32>, tensor<1024xf32>) -> tensor<1x14x14x1024xf32>
    %125 = "eltwise.add"(%124, %117) : (tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %126 = "eltwise.cmp_lt"(%125, %cst) : (tensor<1x14x14x1024xf32>, f32) -> tensor<1x14x14x1024xi1>
    %127 = "eltwise.select"(%126, %cst, %125) : (tensor<1x14x14x1024xi1>, f32, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %128 = "tile.trace"(%127) {msg = "res4e"} : (tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %res4e_branch2a = tile.contract add, mul, %cst, %128, %arg84 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : f32, tensor<1x14x14x1024xf32>, tensor<1x1x1024x256xf32> -> tensor<1x14x14x256xf32>
    %129 = "eltwise.add"(%res4e_branch2a, %arg83) : (tensor<1x14x14x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
    %130 = "eltwise.cmp_lt"(%129, %cst) : (tensor<1x14x14x256xf32>, f32) -> tensor<1x14x14x256xi1>
    %131 = "eltwise.select"(%130, %cst, %129) : (tensor<1x14x14x256xi1>, f32, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %res4e_branch2b = tile.contract add, mul, %cst, %131, %arg82 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : f32, tensor<1x14x14x256xf32>, tensor<3x3x256x256xf32> -> tensor<1x14x14x256xf32>
    %132 = "eltwise.add"(%res4e_branch2b, %arg81) : (tensor<1x14x14x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
    %133 = "eltwise.cmp_lt"(%132, %cst) : (tensor<1x14x14x256xf32>, f32) -> tensor<1x14x14x256xi1>
    %134 = "eltwise.select"(%133, %cst, %132) : (tensor<1x14x14x256xi1>, f32, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %res4e_branch2c = tile.contract add, mul, %cst, %134, %arg80 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : f32, tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32> -> tensor<1x14x14x1024xf32>
    %135 = "eltwise.add"(%res4e_branch2c, %arg79) : (tensor<1x14x14x1024xf32>, tensor<1024xf32>) -> tensor<1x14x14x1024xf32>
    %136 = "eltwise.add"(%135, %128) : (tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %137 = "eltwise.cmp_lt"(%136, %cst) : (tensor<1x14x14x1024xf32>, f32) -> tensor<1x14x14x1024xi1>
    %138 = "eltwise.select"(%137, %cst, %136) : (tensor<1x14x14x1024xi1>, f32, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %139 = "tile.trace"(%138) {msg = "res4f"} : (tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %res4f_branch2a = tile.contract add, mul, %cst, %139, %arg90 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : f32, tensor<1x14x14x1024xf32>, tensor<1x1x1024x256xf32> -> tensor<1x14x14x256xf32>
    %140 = "eltwise.add"(%res4f_branch2a, %arg89) : (tensor<1x14x14x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
    %141 = "eltwise.cmp_lt"(%140, %cst) : (tensor<1x14x14x256xf32>, f32) -> tensor<1x14x14x256xi1>
    %142 = "eltwise.select"(%141, %cst, %140) : (tensor<1x14x14x256xi1>, f32, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %res4f_branch2b = tile.contract add, mul, %cst, %142, %arg88 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : f32, tensor<1x14x14x256xf32>, tensor<3x3x256x256xf32> -> tensor<1x14x14x256xf32>
    %143 = "eltwise.add"(%res4f_branch2b, %arg87) : (tensor<1x14x14x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
    %144 = "eltwise.cmp_lt"(%143, %cst) : (tensor<1x14x14x256xf32>, f32) -> tensor<1x14x14x256xi1>
    %145 = "eltwise.select"(%144, %cst, %143) : (tensor<1x14x14x256xi1>, f32, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %res4f_branch2c = tile.contract add, mul, %cst, %145, %arg86 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : f32, tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32> -> tensor<1x14x14x1024xf32>
    %146 = "eltwise.add"(%res4f_branch2c, %arg85) : (tensor<1x14x14x1024xf32>, tensor<1024xf32>) -> tensor<1x14x14x1024xf32>
    %147 = "eltwise.add"(%146, %139) : (tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %148 = "eltwise.cmp_lt"(%147, %cst) : (tensor<1x14x14x1024xf32>, f32) -> tensor<1x14x14x1024xi1>
    %149 = "eltwise.select"(%148, %cst, %147) : (tensor<1x14x14x1024xi1>, f32, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %150 = "tile.trace"(%149) {msg = "res5a"} : (tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %res5a_branch1 = tile.contract add, mul, %cst, %150, %arg3 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map7, #map2]} : f32, tensor<1x14x14x1024xf32>, tensor<1x1x1024x2048xf32> -> tensor<1x7x7x2048xf32>
    %res5a_branch2a = tile.contract add, mul, %cst, %150, %arg96 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map7, #map2]} : f32, tensor<1x14x14x1024xf32>, tensor<1x1x1024x512xf32> -> tensor<1x7x7x512xf32>
    %151 = "eltwise.add"(%res5a_branch2a, %arg95) : (tensor<1x7x7x512xf32>, tensor<512xf32>) -> tensor<1x7x7x512xf32>
    %152 = "eltwise.cmp_lt"(%151, %cst) : (tensor<1x7x7x512xf32>, f32) -> tensor<1x7x7x512xi1>
    %153 = "eltwise.select"(%152, %cst, %151) : (tensor<1x7x7x512xi1>, f32, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %res5a_branch2b = tile.contract add, mul, %cst, %153, %arg94 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : f32, tensor<1x7x7x512xf32>, tensor<3x3x512x512xf32> -> tensor<1x7x7x512xf32>
    %154 = "eltwise.add"(%res5a_branch2b, %arg93) : (tensor<1x7x7x512xf32>, tensor<512xf32>) -> tensor<1x7x7x512xf32>
    %155 = "eltwise.cmp_lt"(%154, %cst) : (tensor<1x7x7x512xf32>, f32) -> tensor<1x7x7x512xi1>
    %156 = "eltwise.select"(%155, %cst, %154) : (tensor<1x7x7x512xi1>, f32, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %res5a_branch2c = tile.contract add, mul, %cst, %156, %arg92 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : f32, tensor<1x7x7x512xf32>, tensor<1x1x512x2048xf32> -> tensor<1x7x7x2048xf32>
    %157 = "eltwise.add"(%res5a_branch2c, %arg91) : (tensor<1x7x7x2048xf32>, tensor<2048xf32>) -> tensor<1x7x7x2048xf32>
    %158 = "eltwise.add"(%157, %res5a_branch1) : (tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    %159 = "eltwise.add"(%158, %arg2) : (tensor<1x7x7x2048xf32>, tensor<2048xf32>) -> tensor<1x7x7x2048xf32>
    %160 = "eltwise.cmp_lt"(%159, %cst) : (tensor<1x7x7x2048xf32>, f32) -> tensor<1x7x7x2048xi1>
    %161 = "eltwise.select"(%160, %cst, %159) : (tensor<1x7x7x2048xi1>, f32, tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    %162 = "tile.trace"(%161) {msg = "res5b"} : (tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    %res5b_branch2a = tile.contract add, mul, %cst, %162, %arg102 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : f32, tensor<1x7x7x2048xf32>, tensor<1x1x2048x512xf32> -> tensor<1x7x7x512xf32>
    %163 = "eltwise.add"(%res5b_branch2a, %arg101) : (tensor<1x7x7x512xf32>, tensor<512xf32>) -> tensor<1x7x7x512xf32>
    %164 = "eltwise.cmp_lt"(%163, %cst) : (tensor<1x7x7x512xf32>, f32) -> tensor<1x7x7x512xi1>
    %165 = "eltwise.select"(%164, %cst, %163) : (tensor<1x7x7x512xi1>, f32, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %res5b_branch2b = tile.contract add, mul, %cst, %165, %arg100 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : f32, tensor<1x7x7x512xf32>, tensor<3x3x512x512xf32> -> tensor<1x7x7x512xf32>
    %166 = "eltwise.add"(%res5b_branch2b, %arg99) : (tensor<1x7x7x512xf32>, tensor<512xf32>) -> tensor<1x7x7x512xf32>
    %167 = "eltwise.cmp_lt"(%166, %cst) : (tensor<1x7x7x512xf32>, f32) -> tensor<1x7x7x512xi1>
    %168 = "eltwise.select"(%167, %cst, %166) : (tensor<1x7x7x512xi1>, f32, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %res5b_branch2c = tile.contract add, mul, %cst, %168, %arg98 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : f32, tensor<1x7x7x512xf32>, tensor<1x1x512x2048xf32> -> tensor<1x7x7x2048xf32>
    %169 = "eltwise.add"(%res5b_branch2c, %arg97) : (tensor<1x7x7x2048xf32>, tensor<2048xf32>) -> tensor<1x7x7x2048xf32>
    %170 = "eltwise.add"(%169, %162) : (tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    %171 = "eltwise.cmp_lt"(%170, %cst) : (tensor<1x7x7x2048xf32>, f32) -> tensor<1x7x7x2048xi1>
    %172 = "eltwise.select"(%171, %cst, %170) : (tensor<1x7x7x2048xi1>, f32, tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    %173 = "tile.trace"(%172) {msg = "res5c"} : (tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    %res5c_branch2a = tile.contract add, mul, %cst, %173, %arg108 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : f32, tensor<1x7x7x2048xf32>, tensor<1x1x2048x512xf32> -> tensor<1x7x7x512xf32>
    %174 = "eltwise.add"(%res5c_branch2a, %arg107) : (tensor<1x7x7x512xf32>, tensor<512xf32>) -> tensor<1x7x7x512xf32>
    %175 = "eltwise.cmp_lt"(%174, %cst) : (tensor<1x7x7x512xf32>, f32) -> tensor<1x7x7x512xi1>
    %176 = "eltwise.select"(%175, %cst, %174) : (tensor<1x7x7x512xi1>, f32, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %res5c_branch2b = tile.contract add, mul, %cst, %176, %arg106 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : f32, tensor<1x7x7x512xf32>, tensor<3x3x512x512xf32> -> tensor<1x7x7x512xf32>
    %177 = "eltwise.add"(%res5c_branch2b, %arg105) : (tensor<1x7x7x512xf32>, tensor<512xf32>) -> tensor<1x7x7x512xf32>
    %178 = "eltwise.cmp_lt"(%177, %cst) : (tensor<1x7x7x512xf32>, f32) -> tensor<1x7x7x512xi1>
    %179 = "eltwise.select"(%178, %cst, %177) : (tensor<1x7x7x512xi1>, f32, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %res5c_branch2c = tile.contract add, mul, %cst, %179, %arg104 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : f32, tensor<1x7x7x512xf32>, tensor<1x1x512x2048xf32> -> tensor<1x7x7x2048xf32>
    %180 = "eltwise.add"(%res5c_branch2c, %arg103) : (tensor<1x7x7x2048xf32>, tensor<2048xf32>) -> tensor<1x7x7x2048xf32>
    %181 = "eltwise.add"(%180, %173) : (tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    %182 = "eltwise.cmp_lt"(%181, %cst) : (tensor<1x7x7x2048xf32>, f32) -> tensor<1x7x7x2048xi1>
    %183 = "eltwise.select"(%182, %cst, %181) : (tensor<1x7x7x2048xi1>, f32, tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    %184 = tile.contract add, none, %cst, %183 {sink = #map8, srcs = [#map9]} : f32, tensor<1x7x7x2048xf32> -> tensor<1x2048xf32>
    %185 = "eltwise.div"(%184, %c49) : (tensor<1x2048xf32>, si32) -> tensor<1x2048xf32>
    %186 = tile.contract add, mul, %cst, %185, %arg1 {sink = #map10, srcs = [#map11, #map12]} : f32, tensor<1x2048xf32>, tensor<2048x1000xf32> -> tensor<1x1000xf32>
    %187 = "eltwise.add"(%186, %arg0) : (tensor<1x1000xf32>, tensor<1000xf32>) -> tensor<1x1000xf32>
    %188 = "eltwise.ident"(%187) : (tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %189 = tile.contract max, none, %cst, %188 {sink = #map13, srcs = [#map14]} : f32, tensor<1x1000xf32> -> tensor<1x1xf32>
    %190 = "eltwise.sub"(%188, %189) : (tensor<1x1000xf32>, tensor<1x1xf32>) -> tensor<1x1000xf32>
    %191 = "eltwise.exp"(%190) : (tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %192 = tile.contract add, none, %cst, %191 {sink = #map13, srcs = [#map14]} : f32, tensor<1x1000xf32> -> tensor<1x1xf32>
    %193 = "eltwise.div"(%191, %192) : (tensor<1x1000xf32>, tensor<1x1xf32>) -> tensor<1x1000xf32>
    return %193 : tensor<1x1000xf32>
  }
}
