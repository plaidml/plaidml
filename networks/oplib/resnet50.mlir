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

!i32 = type tensor<!eltwise.i32>
!f32 = type tensor<!eltwise.f32>
module {
  func @resnet50(%arg0: tensor<1000x!eltwise.f32>, %arg1: tensor<2048x1000x!eltwise.f32>, %arg2: tensor<2048x!eltwise.f32>, %arg3: tensor<1x1x1024x2048x!eltwise.f32>, %arg4: tensor<1024x!eltwise.f32>, %arg5: tensor<1x1x512x1024x!eltwise.f32>, %arg6: tensor<512x!eltwise.f32>, %arg7: tensor<1x1x256x512x!eltwise.f32>, %arg8: tensor<256x!eltwise.f32>, %arg9: tensor<1x1x64x256x!eltwise.f32>, %arg10: tensor<64x!eltwise.f32>, %arg11: tensor<7x7x3x64x!eltwise.f32>, %arg12: tensor<1x224x224x3x!eltwise.f32>, %arg13: tensor<256x!eltwise.f32>, %arg14: tensor<1x1x64x256x!eltwise.f32>, %arg15: tensor<64x!eltwise.f32>, %arg16: tensor<3x3x64x64x!eltwise.f32>, %arg17: tensor<64x!eltwise.f32>, %arg18: tensor<1x1x64x64x!eltwise.f32>, %arg19: tensor<256x!eltwise.f32>, %arg20: tensor<1x1x64x256x!eltwise.f32>, %arg21: tensor<64x!eltwise.f32>, %arg22: tensor<3x3x64x64x!eltwise.f32>, %arg23: tensor<64x!eltwise.f32>, %arg24: tensor<1x1x256x64x!eltwise.f32>, %arg25: tensor<256x!eltwise.f32>, %arg26: tensor<1x1x64x256x!eltwise.f32>, %arg27: tensor<64x!eltwise.f32>, %arg28: tensor<3x3x64x64x!eltwise.f32>, %arg29: tensor<64x!eltwise.f32>, %arg30: tensor<1x1x256x64x!eltwise.f32>, %arg31: tensor<512x!eltwise.f32>, %arg32: tensor<1x1x128x512x!eltwise.f32>, %arg33: tensor<128x!eltwise.f32>, %arg34: tensor<3x3x128x128x!eltwise.f32>, %arg35: tensor<128x!eltwise.f32>, %arg36: tensor<1x1x256x128x!eltwise.f32>, %arg37: tensor<512x!eltwise.f32>, %arg38: tensor<1x1x128x512x!eltwise.f32>, %arg39: tensor<128x!eltwise.f32>, %arg40: tensor<3x3x128x128x!eltwise.f32>, %arg41: tensor<128x!eltwise.f32>, %arg42: tensor<1x1x512x128x!eltwise.f32>, %arg43: tensor<512x!eltwise.f32>, %arg44: tensor<1x1x128x512x!eltwise.f32>, %arg45: tensor<128x!eltwise.f32>, %arg46: tensor<3x3x128x128x!eltwise.f32>, %arg47: tensor<128x!eltwise.f32>, %arg48: tensor<1x1x512x128x!eltwise.f32>, %arg49: tensor<512x!eltwise.f32>, %arg50: tensor<1x1x128x512x!eltwise.f32>, %arg51: tensor<128x!eltwise.f32>, %arg52: tensor<3x3x128x128x!eltwise.f32>, %arg53: tensor<128x!eltwise.f32>, %arg54: tensor<1x1x512x128x!eltwise.f32>, %arg55: tensor<1024x!eltwise.f32>, %arg56: tensor<1x1x256x1024x!eltwise.f32>, %arg57: tensor<256x!eltwise.f32>, %arg58: tensor<3x3x256x256x!eltwise.f32>, %arg59: tensor<256x!eltwise.f32>, %arg60: tensor<1x1x512x256x!eltwise.f32>, %arg61: tensor<1024x!eltwise.f32>, %arg62: tensor<1x1x256x1024x!eltwise.f32>, %arg63: tensor<256x!eltwise.f32>, %arg64: tensor<3x3x256x256x!eltwise.f32>, %arg65: tensor<256x!eltwise.f32>, %arg66: tensor<1x1x1024x256x!eltwise.f32>, %arg67: tensor<1024x!eltwise.f32>, %arg68: tensor<1x1x256x1024x!eltwise.f32>, %arg69: tensor<256x!eltwise.f32>, %arg70: tensor<3x3x256x256x!eltwise.f32>, %arg71: tensor<256x!eltwise.f32>, %arg72: tensor<1x1x1024x256x!eltwise.f32>, %arg73: tensor<1024x!eltwise.f32>, %arg74: tensor<1x1x256x1024x!eltwise.f32>, %arg75: tensor<256x!eltwise.f32>, %arg76: tensor<3x3x256x256x!eltwise.f32>, %arg77: tensor<256x!eltwise.f32>, %arg78: tensor<1x1x1024x256x!eltwise.f32>, %arg79: tensor<1024x!eltwise.f32>, %arg80: tensor<1x1x256x1024x!eltwise.f32>, %arg81: tensor<256x!eltwise.f32>, %arg82: tensor<3x3x256x256x!eltwise.f32>, %arg83: tensor<256x!eltwise.f32>, %arg84: tensor<1x1x1024x256x!eltwise.f32>, %arg85: tensor<1024x!eltwise.f32>, %arg86: tensor<1x1x256x1024x!eltwise.f32>, %arg87: tensor<256x!eltwise.f32>, %arg88: tensor<3x3x256x256x!eltwise.f32>, %arg89: tensor<256x!eltwise.f32>, %arg90: tensor<1x1x1024x256x!eltwise.f32>, %arg91: tensor<2048x!eltwise.f32>, %arg92: tensor<1x1x512x2048x!eltwise.f32>, %arg93: tensor<512x!eltwise.f32>, %arg94: tensor<3x3x512x512x!eltwise.f32>, %arg95: tensor<512x!eltwise.f32>, %arg96: tensor<1x1x1024x512x!eltwise.f32>, %arg97: tensor<2048x!eltwise.f32>, %arg98: tensor<1x1x512x2048x!eltwise.f32>, %arg99: tensor<512x!eltwise.f32>, %arg100: tensor<3x3x512x512x!eltwise.f32>, %arg101: tensor<512x!eltwise.f32>, %arg102: tensor<1x1x2048x512x!eltwise.f32>, %arg103: tensor<2048x!eltwise.f32>, %arg104: tensor<1x1x512x2048x!eltwise.f32>, %arg105: tensor<512x!eltwise.f32>, %arg106: tensor<3x3x512x512x!eltwise.f32>, %arg107: tensor<512x!eltwise.f32>, %arg108: tensor<1x1x2048x512x!eltwise.f32>) -> tensor<1x1000x!eltwise.f32> {
    %c49 = "eltwise.sconst"() {value = 49 : i64} : () -> !i32
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> !f32
    %conv1 = tile.cion add, mul, %cst, %arg12, %arg11 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map1, #map2]} : !f32, tensor<1x224x224x3x!eltwise.f32>, tensor<7x7x3x64x!eltwise.f32> -> tensor<1x112x112x64x!eltwise.f32>
    %0 = "eltwise.add"(%conv1, %arg10) : (tensor<1x112x112x64x!eltwise.f32>, tensor<64x!eltwise.f32>) -> tensor<1x112x112x64x!eltwise.f32>
    %1 = "eltwise.cmp_lt"(%0, %cst) : (tensor<1x112x112x64x!eltwise.f32>, !f32) -> tensor<1x112x112x64x!eltwise.u1>
    %2 = "eltwise.select"(%1, %cst, %0) : (tensor<1x112x112x64x!eltwise.u1>, !f32, tensor<1x112x112x64x!eltwise.f32>) -> tensor<1x112x112x64x!eltwise.f32>
    %3 = tile.cion max, none, %cst, %2 {cons = #set0, sink = #map3, srcs = [#map4]} : !f32, tensor<1x112x112x64x!eltwise.f32> -> tensor<1x56x56x64x!eltwise.f32>
    %4 = "tile.trace"(%3) {msg = "res2a"} : (tensor<1x56x56x64x!eltwise.f32>) -> tensor<1x56x56x64x!eltwise.f32>
    %res2a_branch1 = tile.cion add, mul, %cst, %4, %arg9 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x56x56x64x!eltwise.f32>, tensor<1x1x64x256x!eltwise.f32> -> tensor<1x56x56x256x!eltwise.f32>
    %res2a_branch2a = tile.cion add, mul, %cst, %4, %arg18 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x56x56x64x!eltwise.f32>, tensor<1x1x64x64x!eltwise.f32> -> tensor<1x56x56x64x!eltwise.f32>
    %5 = "eltwise.add"(%res2a_branch2a, %arg17) : (tensor<1x56x56x64x!eltwise.f32>, tensor<64x!eltwise.f32>) -> tensor<1x56x56x64x!eltwise.f32>
    %6 = "eltwise.cmp_lt"(%5, %cst) : (tensor<1x56x56x64x!eltwise.f32>, !f32) -> tensor<1x56x56x64x!eltwise.u1>
    %7 = "eltwise.select"(%6, %cst, %5) : (tensor<1x56x56x64x!eltwise.u1>, !f32, tensor<1x56x56x64x!eltwise.f32>) -> tensor<1x56x56x64x!eltwise.f32>
    %res2a_branch2b = tile.cion add, mul, %cst, %7, %arg16 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : !f32, tensor<1x56x56x64x!eltwise.f32>, tensor<3x3x64x64x!eltwise.f32> -> tensor<1x56x56x64x!eltwise.f32>
    %8 = "eltwise.add"(%res2a_branch2b, %arg15) : (tensor<1x56x56x64x!eltwise.f32>, tensor<64x!eltwise.f32>) -> tensor<1x56x56x64x!eltwise.f32>
    %9 = "eltwise.cmp_lt"(%8, %cst) : (tensor<1x56x56x64x!eltwise.f32>, !f32) -> tensor<1x56x56x64x!eltwise.u1>
    %10 = "eltwise.select"(%9, %cst, %8) : (tensor<1x56x56x64x!eltwise.u1>, !f32, tensor<1x56x56x64x!eltwise.f32>) -> tensor<1x56x56x64x!eltwise.f32>
    %res2a_branch2c = tile.cion add, mul, %cst, %10, %arg14 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x56x56x64x!eltwise.f32>, tensor<1x1x64x256x!eltwise.f32> -> tensor<1x56x56x256x!eltwise.f32>
    %11 = "eltwise.add"(%res2a_branch2c, %arg13) : (tensor<1x56x56x256x!eltwise.f32>, tensor<256x!eltwise.f32>) -> tensor<1x56x56x256x!eltwise.f32>
    %12 = "eltwise.add"(%11, %res2a_branch1) : (tensor<1x56x56x256x!eltwise.f32>, tensor<1x56x56x256x!eltwise.f32>) -> tensor<1x56x56x256x!eltwise.f32>
    %13 = "eltwise.add"(%12, %arg8) : (tensor<1x56x56x256x!eltwise.f32>, tensor<256x!eltwise.f32>) -> tensor<1x56x56x256x!eltwise.f32>
    %14 = "eltwise.cmp_lt"(%13, %cst) : (tensor<1x56x56x256x!eltwise.f32>, !f32) -> tensor<1x56x56x256x!eltwise.u1>
    %15 = "eltwise.select"(%14, %cst, %13) : (tensor<1x56x56x256x!eltwise.u1>, !f32, tensor<1x56x56x256x!eltwise.f32>) -> tensor<1x56x56x256x!eltwise.f32>
    %16 = "tile.trace"(%15) {msg = "res2b"} : (tensor<1x56x56x256x!eltwise.f32>) -> tensor<1x56x56x256x!eltwise.f32>
    %res2b_branch2a = tile.cion add, mul, %cst, %16, %arg24 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x56x56x256x!eltwise.f32>, tensor<1x1x256x64x!eltwise.f32> -> tensor<1x56x56x64x!eltwise.f32>
    %17 = "eltwise.add"(%res2b_branch2a, %arg23) : (tensor<1x56x56x64x!eltwise.f32>, tensor<64x!eltwise.f32>) -> tensor<1x56x56x64x!eltwise.f32>
    %18 = "eltwise.cmp_lt"(%17, %cst) : (tensor<1x56x56x64x!eltwise.f32>, !f32) -> tensor<1x56x56x64x!eltwise.u1>
    %19 = "eltwise.select"(%18, %cst, %17) : (tensor<1x56x56x64x!eltwise.u1>, !f32, tensor<1x56x56x64x!eltwise.f32>) -> tensor<1x56x56x64x!eltwise.f32>
    %res2b_branch2b = tile.cion add, mul, %cst, %19, %arg22 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : !f32, tensor<1x56x56x64x!eltwise.f32>, tensor<3x3x64x64x!eltwise.f32> -> tensor<1x56x56x64x!eltwise.f32>
    %20 = "eltwise.add"(%res2b_branch2b, %arg21) : (tensor<1x56x56x64x!eltwise.f32>, tensor<64x!eltwise.f32>) -> tensor<1x56x56x64x!eltwise.f32>
    %21 = "eltwise.cmp_lt"(%20, %cst) : (tensor<1x56x56x64x!eltwise.f32>, !f32) -> tensor<1x56x56x64x!eltwise.u1>
    %22 = "eltwise.select"(%21, %cst, %20) : (tensor<1x56x56x64x!eltwise.u1>, !f32, tensor<1x56x56x64x!eltwise.f32>) -> tensor<1x56x56x64x!eltwise.f32>
    %res2b_branch2c = tile.cion add, mul, %cst, %22, %arg20 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x56x56x64x!eltwise.f32>, tensor<1x1x64x256x!eltwise.f32> -> tensor<1x56x56x256x!eltwise.f32>
    %23 = "eltwise.add"(%res2b_branch2c, %arg19) : (tensor<1x56x56x256x!eltwise.f32>, tensor<256x!eltwise.f32>) -> tensor<1x56x56x256x!eltwise.f32>
    %24 = "eltwise.add"(%23, %16) : (tensor<1x56x56x256x!eltwise.f32>, tensor<1x56x56x256x!eltwise.f32>) -> tensor<1x56x56x256x!eltwise.f32>
    %25 = "eltwise.cmp_lt"(%24, %cst) : (tensor<1x56x56x256x!eltwise.f32>, !f32) -> tensor<1x56x56x256x!eltwise.u1>
    %26 = "eltwise.select"(%25, %cst, %24) : (tensor<1x56x56x256x!eltwise.u1>, !f32, tensor<1x56x56x256x!eltwise.f32>) -> tensor<1x56x56x256x!eltwise.f32>
    %27 = "tile.trace"(%26) {msg = "res2c"} : (tensor<1x56x56x256x!eltwise.f32>) -> tensor<1x56x56x256x!eltwise.f32>
    %res2c_branch2a = tile.cion add, mul, %cst, %27, %arg30 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x56x56x256x!eltwise.f32>, tensor<1x1x256x64x!eltwise.f32> -> tensor<1x56x56x64x!eltwise.f32>
    %28 = "eltwise.add"(%res2c_branch2a, %arg29) : (tensor<1x56x56x64x!eltwise.f32>, tensor<64x!eltwise.f32>) -> tensor<1x56x56x64x!eltwise.f32>
    %29 = "eltwise.cmp_lt"(%28, %cst) : (tensor<1x56x56x64x!eltwise.f32>, !f32) -> tensor<1x56x56x64x!eltwise.u1>
    %30 = "eltwise.select"(%29, %cst, %28) : (tensor<1x56x56x64x!eltwise.u1>, !f32, tensor<1x56x56x64x!eltwise.f32>) -> tensor<1x56x56x64x!eltwise.f32>
    %res2c_branch2b = tile.cion add, mul, %cst, %30, %arg28 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : !f32, tensor<1x56x56x64x!eltwise.f32>, tensor<3x3x64x64x!eltwise.f32> -> tensor<1x56x56x64x!eltwise.f32>
    %31 = "eltwise.add"(%res2c_branch2b, %arg27) : (tensor<1x56x56x64x!eltwise.f32>, tensor<64x!eltwise.f32>) -> tensor<1x56x56x64x!eltwise.f32>
    %32 = "eltwise.cmp_lt"(%31, %cst) : (tensor<1x56x56x64x!eltwise.f32>, !f32) -> tensor<1x56x56x64x!eltwise.u1>
    %33 = "eltwise.select"(%32, %cst, %31) : (tensor<1x56x56x64x!eltwise.u1>, !f32, tensor<1x56x56x64x!eltwise.f32>) -> tensor<1x56x56x64x!eltwise.f32>
    %res2c_branch2c = tile.cion add, mul, %cst, %33, %arg26 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x56x56x64x!eltwise.f32>, tensor<1x1x64x256x!eltwise.f32> -> tensor<1x56x56x256x!eltwise.f32>
    %34 = "eltwise.add"(%res2c_branch2c, %arg25) : (tensor<1x56x56x256x!eltwise.f32>, tensor<256x!eltwise.f32>) -> tensor<1x56x56x256x!eltwise.f32>
    %35 = "eltwise.add"(%34, %27) : (tensor<1x56x56x256x!eltwise.f32>, tensor<1x56x56x256x!eltwise.f32>) -> tensor<1x56x56x256x!eltwise.f32>
    %36 = "eltwise.cmp_lt"(%35, %cst) : (tensor<1x56x56x256x!eltwise.f32>, !f32) -> tensor<1x56x56x256x!eltwise.u1>
    %37 = "eltwise.select"(%36, %cst, %35) : (tensor<1x56x56x256x!eltwise.u1>, !f32, tensor<1x56x56x256x!eltwise.f32>) -> tensor<1x56x56x256x!eltwise.f32>
    %38 = "tile.trace"(%37) {msg = "res3a"} : (tensor<1x56x56x256x!eltwise.f32>) -> tensor<1x56x56x256x!eltwise.f32>
    %res3a_branch1 = tile.cion add, mul, %cst, %38, %arg7 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map7, #map2]} : !f32, tensor<1x56x56x256x!eltwise.f32>, tensor<1x1x256x512x!eltwise.f32> -> tensor<1x28x28x512x!eltwise.f32>
    %res3a_branch2a = tile.cion add, mul, %cst, %38, %arg36 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map7, #map2]} : !f32, tensor<1x56x56x256x!eltwise.f32>, tensor<1x1x256x128x!eltwise.f32> -> tensor<1x28x28x128x!eltwise.f32>
    %39 = "eltwise.add"(%res3a_branch2a, %arg35) : (tensor<1x28x28x128x!eltwise.f32>, tensor<128x!eltwise.f32>) -> tensor<1x28x28x128x!eltwise.f32>
    %40 = "eltwise.cmp_lt"(%39, %cst) : (tensor<1x28x28x128x!eltwise.f32>, !f32) -> tensor<1x28x28x128x!eltwise.u1>
    %41 = "eltwise.select"(%40, %cst, %39) : (tensor<1x28x28x128x!eltwise.u1>, !f32, tensor<1x28x28x128x!eltwise.f32>) -> tensor<1x28x28x128x!eltwise.f32>
    %res3a_branch2b = tile.cion add, mul, %cst, %41, %arg34 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : !f32, tensor<1x28x28x128x!eltwise.f32>, tensor<3x3x128x128x!eltwise.f32> -> tensor<1x28x28x128x!eltwise.f32>
    %42 = "eltwise.add"(%res3a_branch2b, %arg33) : (tensor<1x28x28x128x!eltwise.f32>, tensor<128x!eltwise.f32>) -> tensor<1x28x28x128x!eltwise.f32>
    %43 = "eltwise.cmp_lt"(%42, %cst) : (tensor<1x28x28x128x!eltwise.f32>, !f32) -> tensor<1x28x28x128x!eltwise.u1>
    %44 = "eltwise.select"(%43, %cst, %42) : (tensor<1x28x28x128x!eltwise.u1>, !f32, tensor<1x28x28x128x!eltwise.f32>) -> tensor<1x28x28x128x!eltwise.f32>
    %res3a_branch2c = tile.cion add, mul, %cst, %44, %arg32 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x28x28x128x!eltwise.f32>, tensor<1x1x128x512x!eltwise.f32> -> tensor<1x28x28x512x!eltwise.f32>
    %45 = "eltwise.add"(%res3a_branch2c, %arg31) : (tensor<1x28x28x512x!eltwise.f32>, tensor<512x!eltwise.f32>) -> tensor<1x28x28x512x!eltwise.f32>
    %46 = "eltwise.add"(%45, %res3a_branch1) : (tensor<1x28x28x512x!eltwise.f32>, tensor<1x28x28x512x!eltwise.f32>) -> tensor<1x28x28x512x!eltwise.f32>
    %47 = "eltwise.add"(%46, %arg6) : (tensor<1x28x28x512x!eltwise.f32>, tensor<512x!eltwise.f32>) -> tensor<1x28x28x512x!eltwise.f32>
    %48 = "eltwise.cmp_lt"(%47, %cst) : (tensor<1x28x28x512x!eltwise.f32>, !f32) -> tensor<1x28x28x512x!eltwise.u1>
    %49 = "eltwise.select"(%48, %cst, %47) : (tensor<1x28x28x512x!eltwise.u1>, !f32, tensor<1x28x28x512x!eltwise.f32>) -> tensor<1x28x28x512x!eltwise.f32>
    %50 = "tile.trace"(%49) {msg = "res3b"} : (tensor<1x28x28x512x!eltwise.f32>) -> tensor<1x28x28x512x!eltwise.f32>
    %res3b_branch2a = tile.cion add, mul, %cst, %50, %arg42 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x28x28x512x!eltwise.f32>, tensor<1x1x512x128x!eltwise.f32> -> tensor<1x28x28x128x!eltwise.f32>
    %51 = "eltwise.add"(%res3b_branch2a, %arg41) : (tensor<1x28x28x128x!eltwise.f32>, tensor<128x!eltwise.f32>) -> tensor<1x28x28x128x!eltwise.f32>
    %52 = "eltwise.cmp_lt"(%51, %cst) : (tensor<1x28x28x128x!eltwise.f32>, !f32) -> tensor<1x28x28x128x!eltwise.u1>
    %53 = "eltwise.select"(%52, %cst, %51) : (tensor<1x28x28x128x!eltwise.u1>, !f32, tensor<1x28x28x128x!eltwise.f32>) -> tensor<1x28x28x128x!eltwise.f32>
    %res3b_branch2b = tile.cion add, mul, %cst, %53, %arg40 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : !f32, tensor<1x28x28x128x!eltwise.f32>, tensor<3x3x128x128x!eltwise.f32> -> tensor<1x28x28x128x!eltwise.f32>
    %54 = "eltwise.add"(%res3b_branch2b, %arg39) : (tensor<1x28x28x128x!eltwise.f32>, tensor<128x!eltwise.f32>) -> tensor<1x28x28x128x!eltwise.f32>
    %55 = "eltwise.cmp_lt"(%54, %cst) : (tensor<1x28x28x128x!eltwise.f32>, !f32) -> tensor<1x28x28x128x!eltwise.u1>
    %56 = "eltwise.select"(%55, %cst, %54) : (tensor<1x28x28x128x!eltwise.u1>, !f32, tensor<1x28x28x128x!eltwise.f32>) -> tensor<1x28x28x128x!eltwise.f32>
    %res3b_branch2c = tile.cion add, mul, %cst, %56, %arg38 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x28x28x128x!eltwise.f32>, tensor<1x1x128x512x!eltwise.f32> -> tensor<1x28x28x512x!eltwise.f32>
    %57 = "eltwise.add"(%res3b_branch2c, %arg37) : (tensor<1x28x28x512x!eltwise.f32>, tensor<512x!eltwise.f32>) -> tensor<1x28x28x512x!eltwise.f32>
    %58 = "eltwise.add"(%57, %50) : (tensor<1x28x28x512x!eltwise.f32>, tensor<1x28x28x512x!eltwise.f32>) -> tensor<1x28x28x512x!eltwise.f32>
    %59 = "eltwise.cmp_lt"(%58, %cst) : (tensor<1x28x28x512x!eltwise.f32>, !f32) -> tensor<1x28x28x512x!eltwise.u1>
    %60 = "eltwise.select"(%59, %cst, %58) : (tensor<1x28x28x512x!eltwise.u1>, !f32, tensor<1x28x28x512x!eltwise.f32>) -> tensor<1x28x28x512x!eltwise.f32>
    %61 = "tile.trace"(%60) {msg = "res3c"} : (tensor<1x28x28x512x!eltwise.f32>) -> tensor<1x28x28x512x!eltwise.f32>
    %res3c_branch2a = tile.cion add, mul, %cst, %61, %arg48 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x28x28x512x!eltwise.f32>, tensor<1x1x512x128x!eltwise.f32> -> tensor<1x28x28x128x!eltwise.f32>
    %62 = "eltwise.add"(%res3c_branch2a, %arg47) : (tensor<1x28x28x128x!eltwise.f32>, tensor<128x!eltwise.f32>) -> tensor<1x28x28x128x!eltwise.f32>
    %63 = "eltwise.cmp_lt"(%62, %cst) : (tensor<1x28x28x128x!eltwise.f32>, !f32) -> tensor<1x28x28x128x!eltwise.u1>
    %64 = "eltwise.select"(%63, %cst, %62) : (tensor<1x28x28x128x!eltwise.u1>, !f32, tensor<1x28x28x128x!eltwise.f32>) -> tensor<1x28x28x128x!eltwise.f32>
    %res3c_branch2b = tile.cion add, mul, %cst, %64, %arg46 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : !f32, tensor<1x28x28x128x!eltwise.f32>, tensor<3x3x128x128x!eltwise.f32> -> tensor<1x28x28x128x!eltwise.f32>
    %65 = "eltwise.add"(%res3c_branch2b, %arg45) : (tensor<1x28x28x128x!eltwise.f32>, tensor<128x!eltwise.f32>) -> tensor<1x28x28x128x!eltwise.f32>
    %66 = "eltwise.cmp_lt"(%65, %cst) : (tensor<1x28x28x128x!eltwise.f32>, !f32) -> tensor<1x28x28x128x!eltwise.u1>
    %67 = "eltwise.select"(%66, %cst, %65) : (tensor<1x28x28x128x!eltwise.u1>, !f32, tensor<1x28x28x128x!eltwise.f32>) -> tensor<1x28x28x128x!eltwise.f32>
    %res3c_branch2c = tile.cion add, mul, %cst, %67, %arg44 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x28x28x128x!eltwise.f32>, tensor<1x1x128x512x!eltwise.f32> -> tensor<1x28x28x512x!eltwise.f32>
    %68 = "eltwise.add"(%res3c_branch2c, %arg43) : (tensor<1x28x28x512x!eltwise.f32>, tensor<512x!eltwise.f32>) -> tensor<1x28x28x512x!eltwise.f32>
    %69 = "eltwise.add"(%68, %61) : (tensor<1x28x28x512x!eltwise.f32>, tensor<1x28x28x512x!eltwise.f32>) -> tensor<1x28x28x512x!eltwise.f32>
    %70 = "eltwise.cmp_lt"(%69, %cst) : (tensor<1x28x28x512x!eltwise.f32>, !f32) -> tensor<1x28x28x512x!eltwise.u1>
    %71 = "eltwise.select"(%70, %cst, %69) : (tensor<1x28x28x512x!eltwise.u1>, !f32, tensor<1x28x28x512x!eltwise.f32>) -> tensor<1x28x28x512x!eltwise.f32>
    %72 = "tile.trace"(%71) {msg = "res3d"} : (tensor<1x28x28x512x!eltwise.f32>) -> tensor<1x28x28x512x!eltwise.f32>
    %res3d_branch2a = tile.cion add, mul, %cst, %72, %arg54 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x28x28x512x!eltwise.f32>, tensor<1x1x512x128x!eltwise.f32> -> tensor<1x28x28x128x!eltwise.f32>
    %73 = "eltwise.add"(%res3d_branch2a, %arg53) : (tensor<1x28x28x128x!eltwise.f32>, tensor<128x!eltwise.f32>) -> tensor<1x28x28x128x!eltwise.f32>
    %74 = "eltwise.cmp_lt"(%73, %cst) : (tensor<1x28x28x128x!eltwise.f32>, !f32) -> tensor<1x28x28x128x!eltwise.u1>
    %75 = "eltwise.select"(%74, %cst, %73) : (tensor<1x28x28x128x!eltwise.u1>, !f32, tensor<1x28x28x128x!eltwise.f32>) -> tensor<1x28x28x128x!eltwise.f32>
    %res3d_branch2b = tile.cion add, mul, %cst, %75, %arg52 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : !f32, tensor<1x28x28x128x!eltwise.f32>, tensor<3x3x128x128x!eltwise.f32> -> tensor<1x28x28x128x!eltwise.f32>
    %76 = "eltwise.add"(%res3d_branch2b, %arg51) : (tensor<1x28x28x128x!eltwise.f32>, tensor<128x!eltwise.f32>) -> tensor<1x28x28x128x!eltwise.f32>
    %77 = "eltwise.cmp_lt"(%76, %cst) : (tensor<1x28x28x128x!eltwise.f32>, !f32) -> tensor<1x28x28x128x!eltwise.u1>
    %78 = "eltwise.select"(%77, %cst, %76) : (tensor<1x28x28x128x!eltwise.u1>, !f32, tensor<1x28x28x128x!eltwise.f32>) -> tensor<1x28x28x128x!eltwise.f32>
    %res3d_branch2c = tile.cion add, mul, %cst, %78, %arg50 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x28x28x128x!eltwise.f32>, tensor<1x1x128x512x!eltwise.f32> -> tensor<1x28x28x512x!eltwise.f32>
    %79 = "eltwise.add"(%res3d_branch2c, %arg49) : (tensor<1x28x28x512x!eltwise.f32>, tensor<512x!eltwise.f32>) -> tensor<1x28x28x512x!eltwise.f32>
    %80 = "eltwise.add"(%79, %72) : (tensor<1x28x28x512x!eltwise.f32>, tensor<1x28x28x512x!eltwise.f32>) -> tensor<1x28x28x512x!eltwise.f32>
    %81 = "eltwise.cmp_lt"(%80, %cst) : (tensor<1x28x28x512x!eltwise.f32>, !f32) -> tensor<1x28x28x512x!eltwise.u1>
    %82 = "eltwise.select"(%81, %cst, %80) : (tensor<1x28x28x512x!eltwise.u1>, !f32, tensor<1x28x28x512x!eltwise.f32>) -> tensor<1x28x28x512x!eltwise.f32>
    %83 = "tile.trace"(%82) {msg = "res4a"} : (tensor<1x28x28x512x!eltwise.f32>) -> tensor<1x28x28x512x!eltwise.f32>
    %res4a_branch1 = tile.cion add, mul, %cst, %83, %arg5 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map7, #map2]} : !f32, tensor<1x28x28x512x!eltwise.f32>, tensor<1x1x512x1024x!eltwise.f32> -> tensor<1x14x14x1024x!eltwise.f32>
    %res4a_branch2a = tile.cion add, mul, %cst, %83, %arg60 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map7, #map2]} : !f32, tensor<1x28x28x512x!eltwise.f32>, tensor<1x1x512x256x!eltwise.f32> -> tensor<1x14x14x256x!eltwise.f32>
    %84 = "eltwise.add"(%res4a_branch2a, %arg59) : (tensor<1x14x14x256x!eltwise.f32>, tensor<256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %85 = "eltwise.cmp_lt"(%84, %cst) : (tensor<1x14x14x256x!eltwise.f32>, !f32) -> tensor<1x14x14x256x!eltwise.u1>
    %86 = "eltwise.select"(%85, %cst, %84) : (tensor<1x14x14x256x!eltwise.u1>, !f32, tensor<1x14x14x256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %res4a_branch2b = tile.cion add, mul, %cst, %86, %arg58 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : !f32, tensor<1x14x14x256x!eltwise.f32>, tensor<3x3x256x256x!eltwise.f32> -> tensor<1x14x14x256x!eltwise.f32>
    %87 = "eltwise.add"(%res4a_branch2b, %arg57) : (tensor<1x14x14x256x!eltwise.f32>, tensor<256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %88 = "eltwise.cmp_lt"(%87, %cst) : (tensor<1x14x14x256x!eltwise.f32>, !f32) -> tensor<1x14x14x256x!eltwise.u1>
    %89 = "eltwise.select"(%88, %cst, %87) : (tensor<1x14x14x256x!eltwise.u1>, !f32, tensor<1x14x14x256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %res4a_branch2c = tile.cion add, mul, %cst, %89, %arg56 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x14x14x256x!eltwise.f32>, tensor<1x1x256x1024x!eltwise.f32> -> tensor<1x14x14x1024x!eltwise.f32>
    %90 = "eltwise.add"(%res4a_branch2c, %arg55) : (tensor<1x14x14x1024x!eltwise.f32>, tensor<1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %91 = "eltwise.add"(%90, %res4a_branch1) : (tensor<1x14x14x1024x!eltwise.f32>, tensor<1x14x14x1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %92 = "eltwise.add"(%91, %arg4) : (tensor<1x14x14x1024x!eltwise.f32>, tensor<1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %93 = "eltwise.cmp_lt"(%92, %cst) : (tensor<1x14x14x1024x!eltwise.f32>, !f32) -> tensor<1x14x14x1024x!eltwise.u1>
    %94 = "eltwise.select"(%93, %cst, %92) : (tensor<1x14x14x1024x!eltwise.u1>, !f32, tensor<1x14x14x1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %95 = "tile.trace"(%94) {msg = "res4b"} : (tensor<1x14x14x1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %res4b_branch2a = tile.cion add, mul, %cst, %95, %arg66 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x14x14x1024x!eltwise.f32>, tensor<1x1x1024x256x!eltwise.f32> -> tensor<1x14x14x256x!eltwise.f32>
    %96 = "eltwise.add"(%res4b_branch2a, %arg65) : (tensor<1x14x14x256x!eltwise.f32>, tensor<256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %97 = "eltwise.cmp_lt"(%96, %cst) : (tensor<1x14x14x256x!eltwise.f32>, !f32) -> tensor<1x14x14x256x!eltwise.u1>
    %98 = "eltwise.select"(%97, %cst, %96) : (tensor<1x14x14x256x!eltwise.u1>, !f32, tensor<1x14x14x256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %res4b_branch2b = tile.cion add, mul, %cst, %98, %arg64 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : !f32, tensor<1x14x14x256x!eltwise.f32>, tensor<3x3x256x256x!eltwise.f32> -> tensor<1x14x14x256x!eltwise.f32>
    %99 = "eltwise.add"(%res4b_branch2b, %arg63) : (tensor<1x14x14x256x!eltwise.f32>, tensor<256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %100 = "eltwise.cmp_lt"(%99, %cst) : (tensor<1x14x14x256x!eltwise.f32>, !f32) -> tensor<1x14x14x256x!eltwise.u1>
    %101 = "eltwise.select"(%100, %cst, %99) : (tensor<1x14x14x256x!eltwise.u1>, !f32, tensor<1x14x14x256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %res4b_branch2c = tile.cion add, mul, %cst, %101, %arg62 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x14x14x256x!eltwise.f32>, tensor<1x1x256x1024x!eltwise.f32> -> tensor<1x14x14x1024x!eltwise.f32>
    %102 = "eltwise.add"(%res4b_branch2c, %arg61) : (tensor<1x14x14x1024x!eltwise.f32>, tensor<1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %103 = "eltwise.add"(%102, %95) : (tensor<1x14x14x1024x!eltwise.f32>, tensor<1x14x14x1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %104 = "eltwise.cmp_lt"(%103, %cst) : (tensor<1x14x14x1024x!eltwise.f32>, !f32) -> tensor<1x14x14x1024x!eltwise.u1>
    %105 = "eltwise.select"(%104, %cst, %103) : (tensor<1x14x14x1024x!eltwise.u1>, !f32, tensor<1x14x14x1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %106 = "tile.trace"(%105) {msg = "res4c"} : (tensor<1x14x14x1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %res4c_branch2a = tile.cion add, mul, %cst, %106, %arg72 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x14x14x1024x!eltwise.f32>, tensor<1x1x1024x256x!eltwise.f32> -> tensor<1x14x14x256x!eltwise.f32>
    %107 = "eltwise.add"(%res4c_branch2a, %arg71) : (tensor<1x14x14x256x!eltwise.f32>, tensor<256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %108 = "eltwise.cmp_lt"(%107, %cst) : (tensor<1x14x14x256x!eltwise.f32>, !f32) -> tensor<1x14x14x256x!eltwise.u1>
    %109 = "eltwise.select"(%108, %cst, %107) : (tensor<1x14x14x256x!eltwise.u1>, !f32, tensor<1x14x14x256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %res4c_branch2b = tile.cion add, mul, %cst, %109, %arg70 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : !f32, tensor<1x14x14x256x!eltwise.f32>, tensor<3x3x256x256x!eltwise.f32> -> tensor<1x14x14x256x!eltwise.f32>
    %110 = "eltwise.add"(%res4c_branch2b, %arg69) : (tensor<1x14x14x256x!eltwise.f32>, tensor<256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %111 = "eltwise.cmp_lt"(%110, %cst) : (tensor<1x14x14x256x!eltwise.f32>, !f32) -> tensor<1x14x14x256x!eltwise.u1>
    %112 = "eltwise.select"(%111, %cst, %110) : (tensor<1x14x14x256x!eltwise.u1>, !f32, tensor<1x14x14x256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %res4c_branch2c = tile.cion add, mul, %cst, %112, %arg68 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x14x14x256x!eltwise.f32>, tensor<1x1x256x1024x!eltwise.f32> -> tensor<1x14x14x1024x!eltwise.f32>
    %113 = "eltwise.add"(%res4c_branch2c, %arg67) : (tensor<1x14x14x1024x!eltwise.f32>, tensor<1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %114 = "eltwise.add"(%113, %106) : (tensor<1x14x14x1024x!eltwise.f32>, tensor<1x14x14x1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %115 = "eltwise.cmp_lt"(%114, %cst) : (tensor<1x14x14x1024x!eltwise.f32>, !f32) -> tensor<1x14x14x1024x!eltwise.u1>
    %116 = "eltwise.select"(%115, %cst, %114) : (tensor<1x14x14x1024x!eltwise.u1>, !f32, tensor<1x14x14x1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %117 = "tile.trace"(%116) {msg = "res4d"} : (tensor<1x14x14x1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %res4d_branch2a = tile.cion add, mul, %cst, %117, %arg78 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x14x14x1024x!eltwise.f32>, tensor<1x1x1024x256x!eltwise.f32> -> tensor<1x14x14x256x!eltwise.f32>
    %118 = "eltwise.add"(%res4d_branch2a, %arg77) : (tensor<1x14x14x256x!eltwise.f32>, tensor<256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %119 = "eltwise.cmp_lt"(%118, %cst) : (tensor<1x14x14x256x!eltwise.f32>, !f32) -> tensor<1x14x14x256x!eltwise.u1>
    %120 = "eltwise.select"(%119, %cst, %118) : (tensor<1x14x14x256x!eltwise.u1>, !f32, tensor<1x14x14x256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %res4d_branch2b = tile.cion add, mul, %cst, %120, %arg76 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : !f32, tensor<1x14x14x256x!eltwise.f32>, tensor<3x3x256x256x!eltwise.f32> -> tensor<1x14x14x256x!eltwise.f32>
    %121 = "eltwise.add"(%res4d_branch2b, %arg75) : (tensor<1x14x14x256x!eltwise.f32>, tensor<256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %122 = "eltwise.cmp_lt"(%121, %cst) : (tensor<1x14x14x256x!eltwise.f32>, !f32) -> tensor<1x14x14x256x!eltwise.u1>
    %123 = "eltwise.select"(%122, %cst, %121) : (tensor<1x14x14x256x!eltwise.u1>, !f32, tensor<1x14x14x256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %res4d_branch2c = tile.cion add, mul, %cst, %123, %arg74 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x14x14x256x!eltwise.f32>, tensor<1x1x256x1024x!eltwise.f32> -> tensor<1x14x14x1024x!eltwise.f32>
    %124 = "eltwise.add"(%res4d_branch2c, %arg73) : (tensor<1x14x14x1024x!eltwise.f32>, tensor<1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %125 = "eltwise.add"(%124, %117) : (tensor<1x14x14x1024x!eltwise.f32>, tensor<1x14x14x1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %126 = "eltwise.cmp_lt"(%125, %cst) : (tensor<1x14x14x1024x!eltwise.f32>, !f32) -> tensor<1x14x14x1024x!eltwise.u1>
    %127 = "eltwise.select"(%126, %cst, %125) : (tensor<1x14x14x1024x!eltwise.u1>, !f32, tensor<1x14x14x1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %128 = "tile.trace"(%127) {msg = "res4e"} : (tensor<1x14x14x1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %res4e_branch2a = tile.cion add, mul, %cst, %128, %arg84 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x14x14x1024x!eltwise.f32>, tensor<1x1x1024x256x!eltwise.f32> -> tensor<1x14x14x256x!eltwise.f32>
    %129 = "eltwise.add"(%res4e_branch2a, %arg83) : (tensor<1x14x14x256x!eltwise.f32>, tensor<256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %130 = "eltwise.cmp_lt"(%129, %cst) : (tensor<1x14x14x256x!eltwise.f32>, !f32) -> tensor<1x14x14x256x!eltwise.u1>
    %131 = "eltwise.select"(%130, %cst, %129) : (tensor<1x14x14x256x!eltwise.u1>, !f32, tensor<1x14x14x256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %res4e_branch2b = tile.cion add, mul, %cst, %131, %arg82 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : !f32, tensor<1x14x14x256x!eltwise.f32>, tensor<3x3x256x256x!eltwise.f32> -> tensor<1x14x14x256x!eltwise.f32>
    %132 = "eltwise.add"(%res4e_branch2b, %arg81) : (tensor<1x14x14x256x!eltwise.f32>, tensor<256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %133 = "eltwise.cmp_lt"(%132, %cst) : (tensor<1x14x14x256x!eltwise.f32>, !f32) -> tensor<1x14x14x256x!eltwise.u1>
    %134 = "eltwise.select"(%133, %cst, %132) : (tensor<1x14x14x256x!eltwise.u1>, !f32, tensor<1x14x14x256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %res4e_branch2c = tile.cion add, mul, %cst, %134, %arg80 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x14x14x256x!eltwise.f32>, tensor<1x1x256x1024x!eltwise.f32> -> tensor<1x14x14x1024x!eltwise.f32>
    %135 = "eltwise.add"(%res4e_branch2c, %arg79) : (tensor<1x14x14x1024x!eltwise.f32>, tensor<1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %136 = "eltwise.add"(%135, %128) : (tensor<1x14x14x1024x!eltwise.f32>, tensor<1x14x14x1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %137 = "eltwise.cmp_lt"(%136, %cst) : (tensor<1x14x14x1024x!eltwise.f32>, !f32) -> tensor<1x14x14x1024x!eltwise.u1>
    %138 = "eltwise.select"(%137, %cst, %136) : (tensor<1x14x14x1024x!eltwise.u1>, !f32, tensor<1x14x14x1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %139 = "tile.trace"(%138) {msg = "res4f"} : (tensor<1x14x14x1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %res4f_branch2a = tile.cion add, mul, %cst, %139, %arg90 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x14x14x1024x!eltwise.f32>, tensor<1x1x1024x256x!eltwise.f32> -> tensor<1x14x14x256x!eltwise.f32>
    %140 = "eltwise.add"(%res4f_branch2a, %arg89) : (tensor<1x14x14x256x!eltwise.f32>, tensor<256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %141 = "eltwise.cmp_lt"(%140, %cst) : (tensor<1x14x14x256x!eltwise.f32>, !f32) -> tensor<1x14x14x256x!eltwise.u1>
    %142 = "eltwise.select"(%141, %cst, %140) : (tensor<1x14x14x256x!eltwise.u1>, !f32, tensor<1x14x14x256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %res4f_branch2b = tile.cion add, mul, %cst, %142, %arg88 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : !f32, tensor<1x14x14x256x!eltwise.f32>, tensor<3x3x256x256x!eltwise.f32> -> tensor<1x14x14x256x!eltwise.f32>
    %143 = "eltwise.add"(%res4f_branch2b, %arg87) : (tensor<1x14x14x256x!eltwise.f32>, tensor<256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %144 = "eltwise.cmp_lt"(%143, %cst) : (tensor<1x14x14x256x!eltwise.f32>, !f32) -> tensor<1x14x14x256x!eltwise.u1>
    %145 = "eltwise.select"(%144, %cst, %143) : (tensor<1x14x14x256x!eltwise.u1>, !f32, tensor<1x14x14x256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %res4f_branch2c = tile.cion add, mul, %cst, %145, %arg86 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x14x14x256x!eltwise.f32>, tensor<1x1x256x1024x!eltwise.f32> -> tensor<1x14x14x1024x!eltwise.f32>
    %146 = "eltwise.add"(%res4f_branch2c, %arg85) : (tensor<1x14x14x1024x!eltwise.f32>, tensor<1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %147 = "eltwise.add"(%146, %139) : (tensor<1x14x14x1024x!eltwise.f32>, tensor<1x14x14x1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %148 = "eltwise.cmp_lt"(%147, %cst) : (tensor<1x14x14x1024x!eltwise.f32>, !f32) -> tensor<1x14x14x1024x!eltwise.u1>
    %149 = "eltwise.select"(%148, %cst, %147) : (tensor<1x14x14x1024x!eltwise.u1>, !f32, tensor<1x14x14x1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %150 = "tile.trace"(%149) {msg = "res5a"} : (tensor<1x14x14x1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %res5a_branch1 = tile.cion add, mul, %cst, %150, %arg3 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map7, #map2]} : !f32, tensor<1x14x14x1024x!eltwise.f32>, tensor<1x1x1024x2048x!eltwise.f32> -> tensor<1x7x7x2048x!eltwise.f32>
    %res5a_branch2a = tile.cion add, mul, %cst, %150, %arg96 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map7, #map2]} : !f32, tensor<1x14x14x1024x!eltwise.f32>, tensor<1x1x1024x512x!eltwise.f32> -> tensor<1x7x7x512x!eltwise.f32>
    %151 = "eltwise.add"(%res5a_branch2a, %arg95) : (tensor<1x7x7x512x!eltwise.f32>, tensor<512x!eltwise.f32>) -> tensor<1x7x7x512x!eltwise.f32>
    %152 = "eltwise.cmp_lt"(%151, %cst) : (tensor<1x7x7x512x!eltwise.f32>, !f32) -> tensor<1x7x7x512x!eltwise.u1>
    %153 = "eltwise.select"(%152, %cst, %151) : (tensor<1x7x7x512x!eltwise.u1>, !f32, tensor<1x7x7x512x!eltwise.f32>) -> tensor<1x7x7x512x!eltwise.f32>
    %res5a_branch2b = tile.cion add, mul, %cst, %153, %arg94 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : !f32, tensor<1x7x7x512x!eltwise.f32>, tensor<3x3x512x512x!eltwise.f32> -> tensor<1x7x7x512x!eltwise.f32>
    %154 = "eltwise.add"(%res5a_branch2b, %arg93) : (tensor<1x7x7x512x!eltwise.f32>, tensor<512x!eltwise.f32>) -> tensor<1x7x7x512x!eltwise.f32>
    %155 = "eltwise.cmp_lt"(%154, %cst) : (tensor<1x7x7x512x!eltwise.f32>, !f32) -> tensor<1x7x7x512x!eltwise.u1>
    %156 = "eltwise.select"(%155, %cst, %154) : (tensor<1x7x7x512x!eltwise.u1>, !f32, tensor<1x7x7x512x!eltwise.f32>) -> tensor<1x7x7x512x!eltwise.f32>
    %res5a_branch2c = tile.cion add, mul, %cst, %156, %arg92 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x7x7x512x!eltwise.f32>, tensor<1x1x512x2048x!eltwise.f32> -> tensor<1x7x7x2048x!eltwise.f32>
    %157 = "eltwise.add"(%res5a_branch2c, %arg91) : (tensor<1x7x7x2048x!eltwise.f32>, tensor<2048x!eltwise.f32>) -> tensor<1x7x7x2048x!eltwise.f32>
    %158 = "eltwise.add"(%157, %res5a_branch1) : (tensor<1x7x7x2048x!eltwise.f32>, tensor<1x7x7x2048x!eltwise.f32>) -> tensor<1x7x7x2048x!eltwise.f32>
    %159 = "eltwise.add"(%158, %arg2) : (tensor<1x7x7x2048x!eltwise.f32>, tensor<2048x!eltwise.f32>) -> tensor<1x7x7x2048x!eltwise.f32>
    %160 = "eltwise.cmp_lt"(%159, %cst) : (tensor<1x7x7x2048x!eltwise.f32>, !f32) -> tensor<1x7x7x2048x!eltwise.u1>
    %161 = "eltwise.select"(%160, %cst, %159) : (tensor<1x7x7x2048x!eltwise.u1>, !f32, tensor<1x7x7x2048x!eltwise.f32>) -> tensor<1x7x7x2048x!eltwise.f32>
    %162 = "tile.trace"(%161) {msg = "res5b"} : (tensor<1x7x7x2048x!eltwise.f32>) -> tensor<1x7x7x2048x!eltwise.f32>
    %res5b_branch2a = tile.cion add, mul, %cst, %162, %arg102 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x7x7x2048x!eltwise.f32>, tensor<1x1x2048x512x!eltwise.f32> -> tensor<1x7x7x512x!eltwise.f32>
    %163 = "eltwise.add"(%res5b_branch2a, %arg101) : (tensor<1x7x7x512x!eltwise.f32>, tensor<512x!eltwise.f32>) -> tensor<1x7x7x512x!eltwise.f32>
    %164 = "eltwise.cmp_lt"(%163, %cst) : (tensor<1x7x7x512x!eltwise.f32>, !f32) -> tensor<1x7x7x512x!eltwise.u1>
    %165 = "eltwise.select"(%164, %cst, %163) : (tensor<1x7x7x512x!eltwise.u1>, !f32, tensor<1x7x7x512x!eltwise.f32>) -> tensor<1x7x7x512x!eltwise.f32>
    %res5b_branch2b = tile.cion add, mul, %cst, %165, %arg100 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : !f32, tensor<1x7x7x512x!eltwise.f32>, tensor<3x3x512x512x!eltwise.f32> -> tensor<1x7x7x512x!eltwise.f32>
    %166 = "eltwise.add"(%res5b_branch2b, %arg99) : (tensor<1x7x7x512x!eltwise.f32>, tensor<512x!eltwise.f32>) -> tensor<1x7x7x512x!eltwise.f32>
    %167 = "eltwise.cmp_lt"(%166, %cst) : (tensor<1x7x7x512x!eltwise.f32>, !f32) -> tensor<1x7x7x512x!eltwise.u1>
    %168 = "eltwise.select"(%167, %cst, %166) : (tensor<1x7x7x512x!eltwise.u1>, !f32, tensor<1x7x7x512x!eltwise.f32>) -> tensor<1x7x7x512x!eltwise.f32>
    %res5b_branch2c = tile.cion add, mul, %cst, %168, %arg98 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x7x7x512x!eltwise.f32>, tensor<1x1x512x2048x!eltwise.f32> -> tensor<1x7x7x2048x!eltwise.f32>
    %169 = "eltwise.add"(%res5b_branch2c, %arg97) : (tensor<1x7x7x2048x!eltwise.f32>, tensor<2048x!eltwise.f32>) -> tensor<1x7x7x2048x!eltwise.f32>
    %170 = "eltwise.add"(%169, %162) : (tensor<1x7x7x2048x!eltwise.f32>, tensor<1x7x7x2048x!eltwise.f32>) -> tensor<1x7x7x2048x!eltwise.f32>
    %171 = "eltwise.cmp_lt"(%170, %cst) : (tensor<1x7x7x2048x!eltwise.f32>, !f32) -> tensor<1x7x7x2048x!eltwise.u1>
    %172 = "eltwise.select"(%171, %cst, %170) : (tensor<1x7x7x2048x!eltwise.u1>, !f32, tensor<1x7x7x2048x!eltwise.f32>) -> tensor<1x7x7x2048x!eltwise.f32>
    %173 = "tile.trace"(%172) {msg = "res5c"} : (tensor<1x7x7x2048x!eltwise.f32>) -> tensor<1x7x7x2048x!eltwise.f32>
    %res5c_branch2a = tile.cion add, mul, %cst, %173, %arg108 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x7x7x2048x!eltwise.f32>, tensor<1x1x2048x512x!eltwise.f32> -> tensor<1x7x7x512x!eltwise.f32>
    %174 = "eltwise.add"(%res5c_branch2a, %arg107) : (tensor<1x7x7x512x!eltwise.f32>, tensor<512x!eltwise.f32>) -> tensor<1x7x7x512x!eltwise.f32>
    %175 = "eltwise.cmp_lt"(%174, %cst) : (tensor<1x7x7x512x!eltwise.f32>, !f32) -> tensor<1x7x7x512x!eltwise.u1>
    %176 = "eltwise.select"(%175, %cst, %174) : (tensor<1x7x7x512x!eltwise.u1>, !f32, tensor<1x7x7x512x!eltwise.f32>) -> tensor<1x7x7x512x!eltwise.f32>
    %res5c_branch2b = tile.cion add, mul, %cst, %176, %arg106 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : !f32, tensor<1x7x7x512x!eltwise.f32>, tensor<3x3x512x512x!eltwise.f32> -> tensor<1x7x7x512x!eltwise.f32>
    %177 = "eltwise.add"(%res5c_branch2b, %arg105) : (tensor<1x7x7x512x!eltwise.f32>, tensor<512x!eltwise.f32>) -> tensor<1x7x7x512x!eltwise.f32>
    %178 = "eltwise.cmp_lt"(%177, %cst) : (tensor<1x7x7x512x!eltwise.f32>, !f32) -> tensor<1x7x7x512x!eltwise.u1>
    %179 = "eltwise.select"(%178, %cst, %177) : (tensor<1x7x7x512x!eltwise.u1>, !f32, tensor<1x7x7x512x!eltwise.f32>) -> tensor<1x7x7x512x!eltwise.f32>
    %res5c_branch2c = tile.cion add, mul, %cst, %179, %arg104 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x7x7x512x!eltwise.f32>, tensor<1x1x512x2048x!eltwise.f32> -> tensor<1x7x7x2048x!eltwise.f32>
    %180 = "eltwise.add"(%res5c_branch2c, %arg103) : (tensor<1x7x7x2048x!eltwise.f32>, tensor<2048x!eltwise.f32>) -> tensor<1x7x7x2048x!eltwise.f32>
    %181 = "eltwise.add"(%180, %173) : (tensor<1x7x7x2048x!eltwise.f32>, tensor<1x7x7x2048x!eltwise.f32>) -> tensor<1x7x7x2048x!eltwise.f32>
    %182 = "eltwise.cmp_lt"(%181, %cst) : (tensor<1x7x7x2048x!eltwise.f32>, !f32) -> tensor<1x7x7x2048x!eltwise.u1>
    %183 = "eltwise.select"(%182, %cst, %181) : (tensor<1x7x7x2048x!eltwise.u1>, !f32, tensor<1x7x7x2048x!eltwise.f32>) -> tensor<1x7x7x2048x!eltwise.f32>
    %184 = tile.cion add, none, %cst, %183 {sink = #map8, srcs = [#map9]} : !f32, tensor<1x7x7x2048x!eltwise.f32> -> tensor<1x2048x!eltwise.f32>
    %185 = "eltwise.div"(%184, %c49) : (tensor<1x2048x!eltwise.f32>, !i32) -> tensor<1x2048x!eltwise.f32>
    %186 = tile.cion add, mul, %cst, %185, %arg1 {sink = #map10, srcs = [#map11, #map12]} : !f32, tensor<1x2048x!eltwise.f32>, tensor<2048x1000x!eltwise.f32> -> tensor<1x1000x!eltwise.f32>
    %187 = "eltwise.add"(%186, %arg0) : (tensor<1x1000x!eltwise.f32>, tensor<1000x!eltwise.f32>) -> tensor<1x1000x!eltwise.f32>
    %188 = "eltwise.ident"(%187) : (tensor<1x1000x!eltwise.f32>) -> tensor<1x1000x!eltwise.f32>
    %189 = tile.cion max, none, %cst, %188 {sink = #map13, srcs = [#map14]} : !f32, tensor<1x1000x!eltwise.f32> -> tensor<1x1x!eltwise.f32>
    %190 = "eltwise.sub"(%188, %189) : (tensor<1x1000x!eltwise.f32>, tensor<1x1x!eltwise.f32>) -> tensor<1x1000x!eltwise.f32>
    %191 = "eltwise.exp"(%190) : (tensor<1x1000x!eltwise.f32>) -> tensor<1x1000x!eltwise.f32>
    %192 = tile.cion add, none, %cst, %191 {sink = #map13, srcs = [#map14]} : !f32, tensor<1x1000x!eltwise.f32> -> tensor<1x1x!eltwise.f32>
    %193 = "eltwise.div"(%191, %192) : (tensor<1x1000x!eltwise.f32>, tensor<1x1x!eltwise.f32>) -> tensor<1x1000x!eltwise.f32>
    return %193 : tensor<1x1000x!eltwise.f32>
  }
}
