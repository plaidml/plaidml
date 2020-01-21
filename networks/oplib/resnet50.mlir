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
    %c49 = "eltwise.sconst"() {value = 49 : index} : () -> !i32
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> !f32
    %conv1 = tile.cion add, mul, %cst, %arg12, %arg11 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map1, #map2]} : !f32, tensor<1x224x224x3x!eltwise.f32>, tensor<7x7x3x64x!eltwise.f32> -> tensor<1x112x112x64x!eltwise.f32>
    %0 = "eltwise.add"(%conv1, %arg10) : (tensor<1x112x112x64x!eltwise.f32>, tensor<64x!eltwise.f32>) -> tensor<1x112x112x64x!eltwise.f32>
    %1 = "eltwise.cmp_lt"(%0, %cst) : (tensor<1x112x112x64x!eltwise.f32>, !f32) -> tensor<1x112x112x64x!eltwise.u1>
    %2 = "eltwise.select"(%1, %cst, %0) : (tensor<1x112x112x64x!eltwise.u1>, !f32, tensor<1x112x112x64x!eltwise.f32>) -> tensor<1x112x112x64x!eltwise.f32>
    %3 = tile.cion max, none, %cst, %2 {cons = #set0, sink = #map3, srcs = [#map4]} : !f32, tensor<1x112x112x64x!eltwise.f32> -> tensor<1x56x56x64x!eltwise.f32>
    %res2a_branch1 = tile.cion add, mul, %cst, %3, %arg9 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x56x56x64x!eltwise.f32>, tensor<1x1x64x256x!eltwise.f32> -> tensor<1x56x56x256x!eltwise.f32>
    %res2a_branch2a = tile.cion add, mul, %cst, %3, %arg18 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x56x56x64x!eltwise.f32>, tensor<1x1x64x64x!eltwise.f32> -> tensor<1x56x56x64x!eltwise.f32>
    %4 = "eltwise.add"(%res2a_branch2a, %arg17) : (tensor<1x56x56x64x!eltwise.f32>, tensor<64x!eltwise.f32>) -> tensor<1x56x56x64x!eltwise.f32>
    %5 = "eltwise.cmp_lt"(%4, %cst) : (tensor<1x56x56x64x!eltwise.f32>, !f32) -> tensor<1x56x56x64x!eltwise.u1>
    %6 = "eltwise.select"(%5, %cst, %4) : (tensor<1x56x56x64x!eltwise.u1>, !f32, tensor<1x56x56x64x!eltwise.f32>) -> tensor<1x56x56x64x!eltwise.f32>
    %res2a_branch2b = tile.cion add, mul, %cst, %6, %arg16 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : !f32, tensor<1x56x56x64x!eltwise.f32>, tensor<3x3x64x64x!eltwise.f32> -> tensor<1x56x56x64x!eltwise.f32>
    %7 = "eltwise.add"(%res2a_branch2b, %arg15) : (tensor<1x56x56x64x!eltwise.f32>, tensor<64x!eltwise.f32>) -> tensor<1x56x56x64x!eltwise.f32>
    %8 = "eltwise.cmp_lt"(%7, %cst) : (tensor<1x56x56x64x!eltwise.f32>, !f32) -> tensor<1x56x56x64x!eltwise.u1>
    %9 = "eltwise.select"(%8, %cst, %7) : (tensor<1x56x56x64x!eltwise.u1>, !f32, tensor<1x56x56x64x!eltwise.f32>) -> tensor<1x56x56x64x!eltwise.f32>
    %res2a_branch2c = tile.cion add, mul, %cst, %9, %arg14 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x56x56x64x!eltwise.f32>, tensor<1x1x64x256x!eltwise.f32> -> tensor<1x56x56x256x!eltwise.f32>
    %10 = "eltwise.add"(%res2a_branch2c, %arg13) : (tensor<1x56x56x256x!eltwise.f32>, tensor<256x!eltwise.f32>) -> tensor<1x56x56x256x!eltwise.f32>
    %11 = "eltwise.add"(%10, %res2a_branch1) : (tensor<1x56x56x256x!eltwise.f32>, tensor<1x56x56x256x!eltwise.f32>) -> tensor<1x56x56x256x!eltwise.f32>
    %12 = "eltwise.add"(%11, %arg8) : (tensor<1x56x56x256x!eltwise.f32>, tensor<256x!eltwise.f32>) -> tensor<1x56x56x256x!eltwise.f32>
    %13 = "eltwise.cmp_lt"(%12, %cst) : (tensor<1x56x56x256x!eltwise.f32>, !f32) -> tensor<1x56x56x256x!eltwise.u1>
    %14 = "eltwise.select"(%13, %cst, %12) : (tensor<1x56x56x256x!eltwise.u1>, !f32, tensor<1x56x56x256x!eltwise.f32>) -> tensor<1x56x56x256x!eltwise.f32>
    %res2b_branch2a = tile.cion add, mul, %cst, %14, %arg24 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x56x56x256x!eltwise.f32>, tensor<1x1x256x64x!eltwise.f32> -> tensor<1x56x56x64x!eltwise.f32>
    %15 = "eltwise.add"(%res2b_branch2a, %arg23) : (tensor<1x56x56x64x!eltwise.f32>, tensor<64x!eltwise.f32>) -> tensor<1x56x56x64x!eltwise.f32>
    %16 = "eltwise.cmp_lt"(%15, %cst) : (tensor<1x56x56x64x!eltwise.f32>, !f32) -> tensor<1x56x56x64x!eltwise.u1>
    %17 = "eltwise.select"(%16, %cst, %15) : (tensor<1x56x56x64x!eltwise.u1>, !f32, tensor<1x56x56x64x!eltwise.f32>) -> tensor<1x56x56x64x!eltwise.f32>
    %res2b_branch2b = tile.cion add, mul, %cst, %17, %arg22 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : !f32, tensor<1x56x56x64x!eltwise.f32>, tensor<3x3x64x64x!eltwise.f32> -> tensor<1x56x56x64x!eltwise.f32>
    %18 = "eltwise.add"(%res2b_branch2b, %arg21) : (tensor<1x56x56x64x!eltwise.f32>, tensor<64x!eltwise.f32>) -> tensor<1x56x56x64x!eltwise.f32>
    %19 = "eltwise.cmp_lt"(%18, %cst) : (tensor<1x56x56x64x!eltwise.f32>, !f32) -> tensor<1x56x56x64x!eltwise.u1>
    %20 = "eltwise.select"(%19, %cst, %18) : (tensor<1x56x56x64x!eltwise.u1>, !f32, tensor<1x56x56x64x!eltwise.f32>) -> tensor<1x56x56x64x!eltwise.f32>
    %res2b_branch2c = tile.cion add, mul, %cst, %20, %arg20 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x56x56x64x!eltwise.f32>, tensor<1x1x64x256x!eltwise.f32> -> tensor<1x56x56x256x!eltwise.f32>
    %21 = "eltwise.add"(%res2b_branch2c, %arg19) : (tensor<1x56x56x256x!eltwise.f32>, tensor<256x!eltwise.f32>) -> tensor<1x56x56x256x!eltwise.f32>
    %22 = "eltwise.add"(%21, %14) : (tensor<1x56x56x256x!eltwise.f32>, tensor<1x56x56x256x!eltwise.f32>) -> tensor<1x56x56x256x!eltwise.f32>
    %23 = "eltwise.cmp_lt"(%22, %cst) : (tensor<1x56x56x256x!eltwise.f32>, !f32) -> tensor<1x56x56x256x!eltwise.u1>
    %24 = "eltwise.select"(%23, %cst, %22) : (tensor<1x56x56x256x!eltwise.u1>, !f32, tensor<1x56x56x256x!eltwise.f32>) -> tensor<1x56x56x256x!eltwise.f32>
    %res2c_branch2a = tile.cion add, mul, %cst, %24, %arg30 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x56x56x256x!eltwise.f32>, tensor<1x1x256x64x!eltwise.f32> -> tensor<1x56x56x64x!eltwise.f32>
    %25 = "eltwise.add"(%res2c_branch2a, %arg29) : (tensor<1x56x56x64x!eltwise.f32>, tensor<64x!eltwise.f32>) -> tensor<1x56x56x64x!eltwise.f32>
    %26 = "eltwise.cmp_lt"(%25, %cst) : (tensor<1x56x56x64x!eltwise.f32>, !f32) -> tensor<1x56x56x64x!eltwise.u1>
    %27 = "eltwise.select"(%26, %cst, %25) : (tensor<1x56x56x64x!eltwise.u1>, !f32, tensor<1x56x56x64x!eltwise.f32>) -> tensor<1x56x56x64x!eltwise.f32>
    %res2c_branch2b = tile.cion add, mul, %cst, %27, %arg28 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : !f32, tensor<1x56x56x64x!eltwise.f32>, tensor<3x3x64x64x!eltwise.f32> -> tensor<1x56x56x64x!eltwise.f32>
    %28 = "eltwise.add"(%res2c_branch2b, %arg27) : (tensor<1x56x56x64x!eltwise.f32>, tensor<64x!eltwise.f32>) -> tensor<1x56x56x64x!eltwise.f32>
    %29 = "eltwise.cmp_lt"(%28, %cst) : (tensor<1x56x56x64x!eltwise.f32>, !f32) -> tensor<1x56x56x64x!eltwise.u1>
    %30 = "eltwise.select"(%29, %cst, %28) : (tensor<1x56x56x64x!eltwise.u1>, !f32, tensor<1x56x56x64x!eltwise.f32>) -> tensor<1x56x56x64x!eltwise.f32>
    %res2c_branch2c = tile.cion add, mul, %cst, %30, %arg26 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x56x56x64x!eltwise.f32>, tensor<1x1x64x256x!eltwise.f32> -> tensor<1x56x56x256x!eltwise.f32>
    %31 = "eltwise.add"(%res2c_branch2c, %arg25) : (tensor<1x56x56x256x!eltwise.f32>, tensor<256x!eltwise.f32>) -> tensor<1x56x56x256x!eltwise.f32>
    %32 = "eltwise.add"(%31, %24) : (tensor<1x56x56x256x!eltwise.f32>, tensor<1x56x56x256x!eltwise.f32>) -> tensor<1x56x56x256x!eltwise.f32>
    %33 = "eltwise.cmp_lt"(%32, %cst) : (tensor<1x56x56x256x!eltwise.f32>, !f32) -> tensor<1x56x56x256x!eltwise.u1>
    %34 = "eltwise.select"(%33, %cst, %32) : (tensor<1x56x56x256x!eltwise.u1>, !f32, tensor<1x56x56x256x!eltwise.f32>) -> tensor<1x56x56x256x!eltwise.f32>
    %res3a_branch1 = tile.cion add, mul, %cst, %34, %arg7 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map7, #map2]} : !f32, tensor<1x56x56x256x!eltwise.f32>, tensor<1x1x256x512x!eltwise.f32> -> tensor<1x28x28x512x!eltwise.f32>
    %res3a_branch2a = tile.cion add, mul, %cst, %34, %arg36 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map7, #map2]} : !f32, tensor<1x56x56x256x!eltwise.f32>, tensor<1x1x256x128x!eltwise.f32> -> tensor<1x28x28x128x!eltwise.f32>
    %35 = "eltwise.add"(%res3a_branch2a, %arg35) : (tensor<1x28x28x128x!eltwise.f32>, tensor<128x!eltwise.f32>) -> tensor<1x28x28x128x!eltwise.f32>
    %36 = "eltwise.cmp_lt"(%35, %cst) : (tensor<1x28x28x128x!eltwise.f32>, !f32) -> tensor<1x28x28x128x!eltwise.u1>
    %37 = "eltwise.select"(%36, %cst, %35) : (tensor<1x28x28x128x!eltwise.u1>, !f32, tensor<1x28x28x128x!eltwise.f32>) -> tensor<1x28x28x128x!eltwise.f32>
    %res3a_branch2b = tile.cion add, mul, %cst, %37, %arg34 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : !f32, tensor<1x28x28x128x!eltwise.f32>, tensor<3x3x128x128x!eltwise.f32> -> tensor<1x28x28x128x!eltwise.f32>
    %38 = "eltwise.add"(%res3a_branch2b, %arg33) : (tensor<1x28x28x128x!eltwise.f32>, tensor<128x!eltwise.f32>) -> tensor<1x28x28x128x!eltwise.f32>
    %39 = "eltwise.cmp_lt"(%38, %cst) : (tensor<1x28x28x128x!eltwise.f32>, !f32) -> tensor<1x28x28x128x!eltwise.u1>
    %40 = "eltwise.select"(%39, %cst, %38) : (tensor<1x28x28x128x!eltwise.u1>, !f32, tensor<1x28x28x128x!eltwise.f32>) -> tensor<1x28x28x128x!eltwise.f32>
    %res3a_branch2c = tile.cion add, mul, %cst, %40, %arg32 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x28x28x128x!eltwise.f32>, tensor<1x1x128x512x!eltwise.f32> -> tensor<1x28x28x512x!eltwise.f32>
    %41 = "eltwise.add"(%res3a_branch2c, %arg31) : (tensor<1x28x28x512x!eltwise.f32>, tensor<512x!eltwise.f32>) -> tensor<1x28x28x512x!eltwise.f32>
    %42 = "eltwise.add"(%41, %res3a_branch1) : (tensor<1x28x28x512x!eltwise.f32>, tensor<1x28x28x512x!eltwise.f32>) -> tensor<1x28x28x512x!eltwise.f32>
    %43 = "eltwise.add"(%42, %arg6) : (tensor<1x28x28x512x!eltwise.f32>, tensor<512x!eltwise.f32>) -> tensor<1x28x28x512x!eltwise.f32>
    %44 = "eltwise.cmp_lt"(%43, %cst) : (tensor<1x28x28x512x!eltwise.f32>, !f32) -> tensor<1x28x28x512x!eltwise.u1>
    %45 = "eltwise.select"(%44, %cst, %43) : (tensor<1x28x28x512x!eltwise.u1>, !f32, tensor<1x28x28x512x!eltwise.f32>) -> tensor<1x28x28x512x!eltwise.f32>
    %res3b_branch2a = tile.cion add, mul, %cst, %45, %arg42 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x28x28x512x!eltwise.f32>, tensor<1x1x512x128x!eltwise.f32> -> tensor<1x28x28x128x!eltwise.f32>
    %46 = "eltwise.add"(%res3b_branch2a, %arg41) : (tensor<1x28x28x128x!eltwise.f32>, tensor<128x!eltwise.f32>) -> tensor<1x28x28x128x!eltwise.f32>
    %47 = "eltwise.cmp_lt"(%46, %cst) : (tensor<1x28x28x128x!eltwise.f32>, !f32) -> tensor<1x28x28x128x!eltwise.u1>
    %48 = "eltwise.select"(%47, %cst, %46) : (tensor<1x28x28x128x!eltwise.u1>, !f32, tensor<1x28x28x128x!eltwise.f32>) -> tensor<1x28x28x128x!eltwise.f32>
    %res3b_branch2b = tile.cion add, mul, %cst, %48, %arg40 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : !f32, tensor<1x28x28x128x!eltwise.f32>, tensor<3x3x128x128x!eltwise.f32> -> tensor<1x28x28x128x!eltwise.f32>
    %49 = "eltwise.add"(%res3b_branch2b, %arg39) : (tensor<1x28x28x128x!eltwise.f32>, tensor<128x!eltwise.f32>) -> tensor<1x28x28x128x!eltwise.f32>
    %50 = "eltwise.cmp_lt"(%49, %cst) : (tensor<1x28x28x128x!eltwise.f32>, !f32) -> tensor<1x28x28x128x!eltwise.u1>
    %51 = "eltwise.select"(%50, %cst, %49) : (tensor<1x28x28x128x!eltwise.u1>, !f32, tensor<1x28x28x128x!eltwise.f32>) -> tensor<1x28x28x128x!eltwise.f32>
    %res3b_branch2c = tile.cion add, mul, %cst, %51, %arg38 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x28x28x128x!eltwise.f32>, tensor<1x1x128x512x!eltwise.f32> -> tensor<1x28x28x512x!eltwise.f32>
    %52 = "eltwise.add"(%res3b_branch2c, %arg37) : (tensor<1x28x28x512x!eltwise.f32>, tensor<512x!eltwise.f32>) -> tensor<1x28x28x512x!eltwise.f32>
    %53 = "eltwise.add"(%52, %45) : (tensor<1x28x28x512x!eltwise.f32>, tensor<1x28x28x512x!eltwise.f32>) -> tensor<1x28x28x512x!eltwise.f32>
    %54 = "eltwise.cmp_lt"(%53, %cst) : (tensor<1x28x28x512x!eltwise.f32>, !f32) -> tensor<1x28x28x512x!eltwise.u1>
    %55 = "eltwise.select"(%54, %cst, %53) : (tensor<1x28x28x512x!eltwise.u1>, !f32, tensor<1x28x28x512x!eltwise.f32>) -> tensor<1x28x28x512x!eltwise.f32>
    %res3c_branch2a = tile.cion add, mul, %cst, %55, %arg48 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x28x28x512x!eltwise.f32>, tensor<1x1x512x128x!eltwise.f32> -> tensor<1x28x28x128x!eltwise.f32>
    %56 = "eltwise.add"(%res3c_branch2a, %arg47) : (tensor<1x28x28x128x!eltwise.f32>, tensor<128x!eltwise.f32>) -> tensor<1x28x28x128x!eltwise.f32>
    %57 = "eltwise.cmp_lt"(%56, %cst) : (tensor<1x28x28x128x!eltwise.f32>, !f32) -> tensor<1x28x28x128x!eltwise.u1>
    %58 = "eltwise.select"(%57, %cst, %56) : (tensor<1x28x28x128x!eltwise.u1>, !f32, tensor<1x28x28x128x!eltwise.f32>) -> tensor<1x28x28x128x!eltwise.f32>
    %res3c_branch2b = tile.cion add, mul, %cst, %58, %arg46 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : !f32, tensor<1x28x28x128x!eltwise.f32>, tensor<3x3x128x128x!eltwise.f32> -> tensor<1x28x28x128x!eltwise.f32>
    %59 = "eltwise.add"(%res3c_branch2b, %arg45) : (tensor<1x28x28x128x!eltwise.f32>, tensor<128x!eltwise.f32>) -> tensor<1x28x28x128x!eltwise.f32>
    %60 = "eltwise.cmp_lt"(%59, %cst) : (tensor<1x28x28x128x!eltwise.f32>, !f32) -> tensor<1x28x28x128x!eltwise.u1>
    %61 = "eltwise.select"(%60, %cst, %59) : (tensor<1x28x28x128x!eltwise.u1>, !f32, tensor<1x28x28x128x!eltwise.f32>) -> tensor<1x28x28x128x!eltwise.f32>
    %res3c_branch2c = tile.cion add, mul, %cst, %61, %arg44 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x28x28x128x!eltwise.f32>, tensor<1x1x128x512x!eltwise.f32> -> tensor<1x28x28x512x!eltwise.f32>
    %62 = "eltwise.add"(%res3c_branch2c, %arg43) : (tensor<1x28x28x512x!eltwise.f32>, tensor<512x!eltwise.f32>) -> tensor<1x28x28x512x!eltwise.f32>
    %63 = "eltwise.add"(%62, %55) : (tensor<1x28x28x512x!eltwise.f32>, tensor<1x28x28x512x!eltwise.f32>) -> tensor<1x28x28x512x!eltwise.f32>
    %64 = "eltwise.cmp_lt"(%63, %cst) : (tensor<1x28x28x512x!eltwise.f32>, !f32) -> tensor<1x28x28x512x!eltwise.u1>
    %65 = "eltwise.select"(%64, %cst, %63) : (tensor<1x28x28x512x!eltwise.u1>, !f32, tensor<1x28x28x512x!eltwise.f32>) -> tensor<1x28x28x512x!eltwise.f32>
    %res3d_branch2a = tile.cion add, mul, %cst, %65, %arg54 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x28x28x512x!eltwise.f32>, tensor<1x1x512x128x!eltwise.f32> -> tensor<1x28x28x128x!eltwise.f32>
    %66 = "eltwise.add"(%res3d_branch2a, %arg53) : (tensor<1x28x28x128x!eltwise.f32>, tensor<128x!eltwise.f32>) -> tensor<1x28x28x128x!eltwise.f32>
    %67 = "eltwise.cmp_lt"(%66, %cst) : (tensor<1x28x28x128x!eltwise.f32>, !f32) -> tensor<1x28x28x128x!eltwise.u1>
    %68 = "eltwise.select"(%67, %cst, %66) : (tensor<1x28x28x128x!eltwise.u1>, !f32, tensor<1x28x28x128x!eltwise.f32>) -> tensor<1x28x28x128x!eltwise.f32>
    %res3d_branch2b = tile.cion add, mul, %cst, %68, %arg52 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : !f32, tensor<1x28x28x128x!eltwise.f32>, tensor<3x3x128x128x!eltwise.f32> -> tensor<1x28x28x128x!eltwise.f32>
    %69 = "eltwise.add"(%res3d_branch2b, %arg51) : (tensor<1x28x28x128x!eltwise.f32>, tensor<128x!eltwise.f32>) -> tensor<1x28x28x128x!eltwise.f32>
    %70 = "eltwise.cmp_lt"(%69, %cst) : (tensor<1x28x28x128x!eltwise.f32>, !f32) -> tensor<1x28x28x128x!eltwise.u1>
    %71 = "eltwise.select"(%70, %cst, %69) : (tensor<1x28x28x128x!eltwise.u1>, !f32, tensor<1x28x28x128x!eltwise.f32>) -> tensor<1x28x28x128x!eltwise.f32>
    %res3d_branch2c = tile.cion add, mul, %cst, %71, %arg50 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x28x28x128x!eltwise.f32>, tensor<1x1x128x512x!eltwise.f32> -> tensor<1x28x28x512x!eltwise.f32>
    %72 = "eltwise.add"(%res3d_branch2c, %arg49) : (tensor<1x28x28x512x!eltwise.f32>, tensor<512x!eltwise.f32>) -> tensor<1x28x28x512x!eltwise.f32>
    %73 = "eltwise.add"(%72, %65) : (tensor<1x28x28x512x!eltwise.f32>, tensor<1x28x28x512x!eltwise.f32>) -> tensor<1x28x28x512x!eltwise.f32>
    %74 = "eltwise.cmp_lt"(%73, %cst) : (tensor<1x28x28x512x!eltwise.f32>, !f32) -> tensor<1x28x28x512x!eltwise.u1>
    %75 = "eltwise.select"(%74, %cst, %73) : (tensor<1x28x28x512x!eltwise.u1>, !f32, tensor<1x28x28x512x!eltwise.f32>) -> tensor<1x28x28x512x!eltwise.f32>
    %res4a_branch1 = tile.cion add, mul, %cst, %75, %arg5 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map7, #map2]} : !f32, tensor<1x28x28x512x!eltwise.f32>, tensor<1x1x512x1024x!eltwise.f32> -> tensor<1x14x14x1024x!eltwise.f32>
    %res4a_branch2a = tile.cion add, mul, %cst, %75, %arg60 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map7, #map2]} : !f32, tensor<1x28x28x512x!eltwise.f32>, tensor<1x1x512x256x!eltwise.f32> -> tensor<1x14x14x256x!eltwise.f32>
    %76 = "eltwise.add"(%res4a_branch2a, %arg59) : (tensor<1x14x14x256x!eltwise.f32>, tensor<256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %77 = "eltwise.cmp_lt"(%76, %cst) : (tensor<1x14x14x256x!eltwise.f32>, !f32) -> tensor<1x14x14x256x!eltwise.u1>
    %78 = "eltwise.select"(%77, %cst, %76) : (tensor<1x14x14x256x!eltwise.u1>, !f32, tensor<1x14x14x256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %res4a_branch2b = tile.cion add, mul, %cst, %78, %arg58 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : !f32, tensor<1x14x14x256x!eltwise.f32>, tensor<3x3x256x256x!eltwise.f32> -> tensor<1x14x14x256x!eltwise.f32>
    %79 = "eltwise.add"(%res4a_branch2b, %arg57) : (tensor<1x14x14x256x!eltwise.f32>, tensor<256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %80 = "eltwise.cmp_lt"(%79, %cst) : (tensor<1x14x14x256x!eltwise.f32>, !f32) -> tensor<1x14x14x256x!eltwise.u1>
    %81 = "eltwise.select"(%80, %cst, %79) : (tensor<1x14x14x256x!eltwise.u1>, !f32, tensor<1x14x14x256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %res4a_branch2c = tile.cion add, mul, %cst, %81, %arg56 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x14x14x256x!eltwise.f32>, tensor<1x1x256x1024x!eltwise.f32> -> tensor<1x14x14x1024x!eltwise.f32>
    %82 = "eltwise.add"(%res4a_branch2c, %arg55) : (tensor<1x14x14x1024x!eltwise.f32>, tensor<1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %83 = "eltwise.add"(%82, %res4a_branch1) : (tensor<1x14x14x1024x!eltwise.f32>, tensor<1x14x14x1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %84 = "eltwise.add"(%83, %arg4) : (tensor<1x14x14x1024x!eltwise.f32>, tensor<1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %85 = "eltwise.cmp_lt"(%84, %cst) : (tensor<1x14x14x1024x!eltwise.f32>, !f32) -> tensor<1x14x14x1024x!eltwise.u1>
    %86 = "eltwise.select"(%85, %cst, %84) : (tensor<1x14x14x1024x!eltwise.u1>, !f32, tensor<1x14x14x1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %res4b_branch2a = tile.cion add, mul, %cst, %86, %arg66 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x14x14x1024x!eltwise.f32>, tensor<1x1x1024x256x!eltwise.f32> -> tensor<1x14x14x256x!eltwise.f32>
    %87 = "eltwise.add"(%res4b_branch2a, %arg65) : (tensor<1x14x14x256x!eltwise.f32>, tensor<256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %88 = "eltwise.cmp_lt"(%87, %cst) : (tensor<1x14x14x256x!eltwise.f32>, !f32) -> tensor<1x14x14x256x!eltwise.u1>
    %89 = "eltwise.select"(%88, %cst, %87) : (tensor<1x14x14x256x!eltwise.u1>, !f32, tensor<1x14x14x256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %res4b_branch2b = tile.cion add, mul, %cst, %89, %arg64 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : !f32, tensor<1x14x14x256x!eltwise.f32>, tensor<3x3x256x256x!eltwise.f32> -> tensor<1x14x14x256x!eltwise.f32>
    %90 = "eltwise.add"(%res4b_branch2b, %arg63) : (tensor<1x14x14x256x!eltwise.f32>, tensor<256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %91 = "eltwise.cmp_lt"(%90, %cst) : (tensor<1x14x14x256x!eltwise.f32>, !f32) -> tensor<1x14x14x256x!eltwise.u1>
    %92 = "eltwise.select"(%91, %cst, %90) : (tensor<1x14x14x256x!eltwise.u1>, !f32, tensor<1x14x14x256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %res4b_branch2c = tile.cion add, mul, %cst, %92, %arg62 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x14x14x256x!eltwise.f32>, tensor<1x1x256x1024x!eltwise.f32> -> tensor<1x14x14x1024x!eltwise.f32>
    %93 = "eltwise.add"(%res4b_branch2c, %arg61) : (tensor<1x14x14x1024x!eltwise.f32>, tensor<1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %94 = "eltwise.add"(%93, %86) : (tensor<1x14x14x1024x!eltwise.f32>, tensor<1x14x14x1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %95 = "eltwise.cmp_lt"(%94, %cst) : (tensor<1x14x14x1024x!eltwise.f32>, !f32) -> tensor<1x14x14x1024x!eltwise.u1>
    %96 = "eltwise.select"(%95, %cst, %94) : (tensor<1x14x14x1024x!eltwise.u1>, !f32, tensor<1x14x14x1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %res4c_branch2a = tile.cion add, mul, %cst, %96, %arg72 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x14x14x1024x!eltwise.f32>, tensor<1x1x1024x256x!eltwise.f32> -> tensor<1x14x14x256x!eltwise.f32>
    %97 = "eltwise.add"(%res4c_branch2a, %arg71) : (tensor<1x14x14x256x!eltwise.f32>, tensor<256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %98 = "eltwise.cmp_lt"(%97, %cst) : (tensor<1x14x14x256x!eltwise.f32>, !f32) -> tensor<1x14x14x256x!eltwise.u1>
    %99 = "eltwise.select"(%98, %cst, %97) : (tensor<1x14x14x256x!eltwise.u1>, !f32, tensor<1x14x14x256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %res4c_branch2b = tile.cion add, mul, %cst, %99, %arg70 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : !f32, tensor<1x14x14x256x!eltwise.f32>, tensor<3x3x256x256x!eltwise.f32> -> tensor<1x14x14x256x!eltwise.f32>
    %100 = "eltwise.add"(%res4c_branch2b, %arg69) : (tensor<1x14x14x256x!eltwise.f32>, tensor<256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %101 = "eltwise.cmp_lt"(%100, %cst) : (tensor<1x14x14x256x!eltwise.f32>, !f32) -> tensor<1x14x14x256x!eltwise.u1>
    %102 = "eltwise.select"(%101, %cst, %100) : (tensor<1x14x14x256x!eltwise.u1>, !f32, tensor<1x14x14x256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %res4c_branch2c = tile.cion add, mul, %cst, %102, %arg68 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x14x14x256x!eltwise.f32>, tensor<1x1x256x1024x!eltwise.f32> -> tensor<1x14x14x1024x!eltwise.f32>
    %103 = "eltwise.add"(%res4c_branch2c, %arg67) : (tensor<1x14x14x1024x!eltwise.f32>, tensor<1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %104 = "eltwise.add"(%103, %96) : (tensor<1x14x14x1024x!eltwise.f32>, tensor<1x14x14x1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %105 = "eltwise.cmp_lt"(%104, %cst) : (tensor<1x14x14x1024x!eltwise.f32>, !f32) -> tensor<1x14x14x1024x!eltwise.u1>
    %106 = "eltwise.select"(%105, %cst, %104) : (tensor<1x14x14x1024x!eltwise.u1>, !f32, tensor<1x14x14x1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %res4d_branch2a = tile.cion add, mul, %cst, %106, %arg78 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x14x14x1024x!eltwise.f32>, tensor<1x1x1024x256x!eltwise.f32> -> tensor<1x14x14x256x!eltwise.f32>
    %107 = "eltwise.add"(%res4d_branch2a, %arg77) : (tensor<1x14x14x256x!eltwise.f32>, tensor<256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %108 = "eltwise.cmp_lt"(%107, %cst) : (tensor<1x14x14x256x!eltwise.f32>, !f32) -> tensor<1x14x14x256x!eltwise.u1>
    %109 = "eltwise.select"(%108, %cst, %107) : (tensor<1x14x14x256x!eltwise.u1>, !f32, tensor<1x14x14x256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %res4d_branch2b = tile.cion add, mul, %cst, %109, %arg76 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : !f32, tensor<1x14x14x256x!eltwise.f32>, tensor<3x3x256x256x!eltwise.f32> -> tensor<1x14x14x256x!eltwise.f32>
    %110 = "eltwise.add"(%res4d_branch2b, %arg75) : (tensor<1x14x14x256x!eltwise.f32>, tensor<256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %111 = "eltwise.cmp_lt"(%110, %cst) : (tensor<1x14x14x256x!eltwise.f32>, !f32) -> tensor<1x14x14x256x!eltwise.u1>
    %112 = "eltwise.select"(%111, %cst, %110) : (tensor<1x14x14x256x!eltwise.u1>, !f32, tensor<1x14x14x256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %res4d_branch2c = tile.cion add, mul, %cst, %112, %arg74 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x14x14x256x!eltwise.f32>, tensor<1x1x256x1024x!eltwise.f32> -> tensor<1x14x14x1024x!eltwise.f32>
    %113 = "eltwise.add"(%res4d_branch2c, %arg73) : (tensor<1x14x14x1024x!eltwise.f32>, tensor<1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %114 = "eltwise.add"(%113, %106) : (tensor<1x14x14x1024x!eltwise.f32>, tensor<1x14x14x1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %115 = "eltwise.cmp_lt"(%114, %cst) : (tensor<1x14x14x1024x!eltwise.f32>, !f32) -> tensor<1x14x14x1024x!eltwise.u1>
    %116 = "eltwise.select"(%115, %cst, %114) : (tensor<1x14x14x1024x!eltwise.u1>, !f32, tensor<1x14x14x1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %res4e_branch2a = tile.cion add, mul, %cst, %116, %arg84 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x14x14x1024x!eltwise.f32>, tensor<1x1x1024x256x!eltwise.f32> -> tensor<1x14x14x256x!eltwise.f32>
    %117 = "eltwise.add"(%res4e_branch2a, %arg83) : (tensor<1x14x14x256x!eltwise.f32>, tensor<256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %118 = "eltwise.cmp_lt"(%117, %cst) : (tensor<1x14x14x256x!eltwise.f32>, !f32) -> tensor<1x14x14x256x!eltwise.u1>
    %119 = "eltwise.select"(%118, %cst, %117) : (tensor<1x14x14x256x!eltwise.u1>, !f32, tensor<1x14x14x256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %res4e_branch2b = tile.cion add, mul, %cst, %119, %arg82 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : !f32, tensor<1x14x14x256x!eltwise.f32>, tensor<3x3x256x256x!eltwise.f32> -> tensor<1x14x14x256x!eltwise.f32>
    %120 = "eltwise.add"(%res4e_branch2b, %arg81) : (tensor<1x14x14x256x!eltwise.f32>, tensor<256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %121 = "eltwise.cmp_lt"(%120, %cst) : (tensor<1x14x14x256x!eltwise.f32>, !f32) -> tensor<1x14x14x256x!eltwise.u1>
    %122 = "eltwise.select"(%121, %cst, %120) : (tensor<1x14x14x256x!eltwise.u1>, !f32, tensor<1x14x14x256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %res4e_branch2c = tile.cion add, mul, %cst, %122, %arg80 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x14x14x256x!eltwise.f32>, tensor<1x1x256x1024x!eltwise.f32> -> tensor<1x14x14x1024x!eltwise.f32>
    %123 = "eltwise.add"(%res4e_branch2c, %arg79) : (tensor<1x14x14x1024x!eltwise.f32>, tensor<1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %124 = "eltwise.add"(%123, %116) : (tensor<1x14x14x1024x!eltwise.f32>, tensor<1x14x14x1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %125 = "eltwise.cmp_lt"(%124, %cst) : (tensor<1x14x14x1024x!eltwise.f32>, !f32) -> tensor<1x14x14x1024x!eltwise.u1>
    %126 = "eltwise.select"(%125, %cst, %124) : (tensor<1x14x14x1024x!eltwise.u1>, !f32, tensor<1x14x14x1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %res4f_branch2a = tile.cion add, mul, %cst, %126, %arg90 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x14x14x1024x!eltwise.f32>, tensor<1x1x1024x256x!eltwise.f32> -> tensor<1x14x14x256x!eltwise.f32>
    %127 = "eltwise.add"(%res4f_branch2a, %arg89) : (tensor<1x14x14x256x!eltwise.f32>, tensor<256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %128 = "eltwise.cmp_lt"(%127, %cst) : (tensor<1x14x14x256x!eltwise.f32>, !f32) -> tensor<1x14x14x256x!eltwise.u1>
    %129 = "eltwise.select"(%128, %cst, %127) : (tensor<1x14x14x256x!eltwise.u1>, !f32, tensor<1x14x14x256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %res4f_branch2b = tile.cion add, mul, %cst, %129, %arg88 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : !f32, tensor<1x14x14x256x!eltwise.f32>, tensor<3x3x256x256x!eltwise.f32> -> tensor<1x14x14x256x!eltwise.f32>
    %130 = "eltwise.add"(%res4f_branch2b, %arg87) : (tensor<1x14x14x256x!eltwise.f32>, tensor<256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %131 = "eltwise.cmp_lt"(%130, %cst) : (tensor<1x14x14x256x!eltwise.f32>, !f32) -> tensor<1x14x14x256x!eltwise.u1>
    %132 = "eltwise.select"(%131, %cst, %130) : (tensor<1x14x14x256x!eltwise.u1>, !f32, tensor<1x14x14x256x!eltwise.f32>) -> tensor<1x14x14x256x!eltwise.f32>
    %res4f_branch2c = tile.cion add, mul, %cst, %132, %arg86 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x14x14x256x!eltwise.f32>, tensor<1x1x256x1024x!eltwise.f32> -> tensor<1x14x14x1024x!eltwise.f32>
    %133 = "eltwise.add"(%res4f_branch2c, %arg85) : (tensor<1x14x14x1024x!eltwise.f32>, tensor<1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %134 = "eltwise.add"(%133, %126) : (tensor<1x14x14x1024x!eltwise.f32>, tensor<1x14x14x1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %135 = "eltwise.cmp_lt"(%134, %cst) : (tensor<1x14x14x1024x!eltwise.f32>, !f32) -> tensor<1x14x14x1024x!eltwise.u1>
    %136 = "eltwise.select"(%135, %cst, %134) : (tensor<1x14x14x1024x!eltwise.u1>, !f32, tensor<1x14x14x1024x!eltwise.f32>) -> tensor<1x14x14x1024x!eltwise.f32>
    %res5a_branch1 = tile.cion add, mul, %cst, %136, %arg3 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map7, #map2]} : !f32, tensor<1x14x14x1024x!eltwise.f32>, tensor<1x1x1024x2048x!eltwise.f32> -> tensor<1x7x7x2048x!eltwise.f32>
    %res5a_branch2a = tile.cion add, mul, %cst, %136, %arg96 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map7, #map2]} : !f32, tensor<1x14x14x1024x!eltwise.f32>, tensor<1x1x1024x512x!eltwise.f32> -> tensor<1x7x7x512x!eltwise.f32>
    %137 = "eltwise.add"(%res5a_branch2a, %arg95) : (tensor<1x7x7x512x!eltwise.f32>, tensor<512x!eltwise.f32>) -> tensor<1x7x7x512x!eltwise.f32>
    %138 = "eltwise.cmp_lt"(%137, %cst) : (tensor<1x7x7x512x!eltwise.f32>, !f32) -> tensor<1x7x7x512x!eltwise.u1>
    %139 = "eltwise.select"(%138, %cst, %137) : (tensor<1x7x7x512x!eltwise.u1>, !f32, tensor<1x7x7x512x!eltwise.f32>) -> tensor<1x7x7x512x!eltwise.f32>
    %res5a_branch2b = tile.cion add, mul, %cst, %139, %arg94 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : !f32, tensor<1x7x7x512x!eltwise.f32>, tensor<3x3x512x512x!eltwise.f32> -> tensor<1x7x7x512x!eltwise.f32>
    %140 = "eltwise.add"(%res5a_branch2b, %arg93) : (tensor<1x7x7x512x!eltwise.f32>, tensor<512x!eltwise.f32>) -> tensor<1x7x7x512x!eltwise.f32>
    %141 = "eltwise.cmp_lt"(%140, %cst) : (tensor<1x7x7x512x!eltwise.f32>, !f32) -> tensor<1x7x7x512x!eltwise.u1>
    %142 = "eltwise.select"(%141, %cst, %140) : (tensor<1x7x7x512x!eltwise.u1>, !f32, tensor<1x7x7x512x!eltwise.f32>) -> tensor<1x7x7x512x!eltwise.f32>
    %res5a_branch2c = tile.cion add, mul, %cst, %142, %arg92 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x7x7x512x!eltwise.f32>, tensor<1x1x512x2048x!eltwise.f32> -> tensor<1x7x7x2048x!eltwise.f32>
    %143 = "eltwise.add"(%res5a_branch2c, %arg91) : (tensor<1x7x7x2048x!eltwise.f32>, tensor<2048x!eltwise.f32>) -> tensor<1x7x7x2048x!eltwise.f32>
    %144 = "eltwise.add"(%143, %res5a_branch1) : (tensor<1x7x7x2048x!eltwise.f32>, tensor<1x7x7x2048x!eltwise.f32>) -> tensor<1x7x7x2048x!eltwise.f32>
    %145 = "eltwise.add"(%144, %arg2) : (tensor<1x7x7x2048x!eltwise.f32>, tensor<2048x!eltwise.f32>) -> tensor<1x7x7x2048x!eltwise.f32>
    %146 = "eltwise.cmp_lt"(%145, %cst) : (tensor<1x7x7x2048x!eltwise.f32>, !f32) -> tensor<1x7x7x2048x!eltwise.u1>
    %147 = "eltwise.select"(%146, %cst, %145) : (tensor<1x7x7x2048x!eltwise.u1>, !f32, tensor<1x7x7x2048x!eltwise.f32>) -> tensor<1x7x7x2048x!eltwise.f32>
    %res5b_branch2a = tile.cion add, mul, %cst, %147, %arg102 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x7x7x2048x!eltwise.f32>, tensor<1x1x2048x512x!eltwise.f32> -> tensor<1x7x7x512x!eltwise.f32>
    %148 = "eltwise.add"(%res5b_branch2a, %arg101) : (tensor<1x7x7x512x!eltwise.f32>, tensor<512x!eltwise.f32>) -> tensor<1x7x7x512x!eltwise.f32>
    %149 = "eltwise.cmp_lt"(%148, %cst) : (tensor<1x7x7x512x!eltwise.f32>, !f32) -> tensor<1x7x7x512x!eltwise.u1>
    %150 = "eltwise.select"(%149, %cst, %148) : (tensor<1x7x7x512x!eltwise.u1>, !f32, tensor<1x7x7x512x!eltwise.f32>) -> tensor<1x7x7x512x!eltwise.f32>
    %res5b_branch2b = tile.cion add, mul, %cst, %150, %arg100 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : !f32, tensor<1x7x7x512x!eltwise.f32>, tensor<3x3x512x512x!eltwise.f32> -> tensor<1x7x7x512x!eltwise.f32>
    %151 = "eltwise.add"(%res5b_branch2b, %arg99) : (tensor<1x7x7x512x!eltwise.f32>, tensor<512x!eltwise.f32>) -> tensor<1x7x7x512x!eltwise.f32>
    %152 = "eltwise.cmp_lt"(%151, %cst) : (tensor<1x7x7x512x!eltwise.f32>, !f32) -> tensor<1x7x7x512x!eltwise.u1>
    %153 = "eltwise.select"(%152, %cst, %151) : (tensor<1x7x7x512x!eltwise.u1>, !f32, tensor<1x7x7x512x!eltwise.f32>) -> tensor<1x7x7x512x!eltwise.f32>
    %res5b_branch2c = tile.cion add, mul, %cst, %153, %arg98 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x7x7x512x!eltwise.f32>, tensor<1x1x512x2048x!eltwise.f32> -> tensor<1x7x7x2048x!eltwise.f32>
    %154 = "eltwise.add"(%res5b_branch2c, %arg97) : (tensor<1x7x7x2048x!eltwise.f32>, tensor<2048x!eltwise.f32>) -> tensor<1x7x7x2048x!eltwise.f32>
    %155 = "eltwise.add"(%154, %147) : (tensor<1x7x7x2048x!eltwise.f32>, tensor<1x7x7x2048x!eltwise.f32>) -> tensor<1x7x7x2048x!eltwise.f32>
    %156 = "eltwise.cmp_lt"(%155, %cst) : (tensor<1x7x7x2048x!eltwise.f32>, !f32) -> tensor<1x7x7x2048x!eltwise.u1>
    %157 = "eltwise.select"(%156, %cst, %155) : (tensor<1x7x7x2048x!eltwise.u1>, !f32, tensor<1x7x7x2048x!eltwise.f32>) -> tensor<1x7x7x2048x!eltwise.f32>
    %res5c_branch2a = tile.cion add, mul, %cst, %157, %arg108 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x7x7x2048x!eltwise.f32>, tensor<1x1x2048x512x!eltwise.f32> -> tensor<1x7x7x512x!eltwise.f32>
    %158 = "eltwise.add"(%res5c_branch2a, %arg107) : (tensor<1x7x7x512x!eltwise.f32>, tensor<512x!eltwise.f32>) -> tensor<1x7x7x512x!eltwise.f32>
    %159 = "eltwise.cmp_lt"(%158, %cst) : (tensor<1x7x7x512x!eltwise.f32>, !f32) -> tensor<1x7x7x512x!eltwise.u1>
    %160 = "eltwise.select"(%159, %cst, %158) : (tensor<1x7x7x512x!eltwise.u1>, !f32, tensor<1x7x7x512x!eltwise.f32>) -> tensor<1x7x7x512x!eltwise.f32>
    %res5c_branch2b = tile.cion add, mul, %cst, %160, %arg106 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map6, #map2]} : !f32, tensor<1x7x7x512x!eltwise.f32>, tensor<3x3x512x512x!eltwise.f32> -> tensor<1x7x7x512x!eltwise.f32>
    %161 = "eltwise.add"(%res5c_branch2b, %arg105) : (tensor<1x7x7x512x!eltwise.f32>, tensor<512x!eltwise.f32>) -> tensor<1x7x7x512x!eltwise.f32>
    %162 = "eltwise.cmp_lt"(%161, %cst) : (tensor<1x7x7x512x!eltwise.f32>, !f32) -> tensor<1x7x7x512x!eltwise.u1>
    %163 = "eltwise.select"(%162, %cst, %161) : (tensor<1x7x7x512x!eltwise.u1>, !f32, tensor<1x7x7x512x!eltwise.f32>) -> tensor<1x7x7x512x!eltwise.f32>
    %res5c_branch2c = tile.cion add, mul, %cst, %163, %arg104 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map5, #map2]} : !f32, tensor<1x7x7x512x!eltwise.f32>, tensor<1x1x512x2048x!eltwise.f32> -> tensor<1x7x7x2048x!eltwise.f32>
    %164 = "eltwise.add"(%res5c_branch2c, %arg103) : (tensor<1x7x7x2048x!eltwise.f32>, tensor<2048x!eltwise.f32>) -> tensor<1x7x7x2048x!eltwise.f32>
    %165 = "eltwise.add"(%164, %157) : (tensor<1x7x7x2048x!eltwise.f32>, tensor<1x7x7x2048x!eltwise.f32>) -> tensor<1x7x7x2048x!eltwise.f32>
    %166 = "eltwise.cmp_lt"(%165, %cst) : (tensor<1x7x7x2048x!eltwise.f32>, !f32) -> tensor<1x7x7x2048x!eltwise.u1>
    %167 = "eltwise.select"(%166, %cst, %165) : (tensor<1x7x7x2048x!eltwise.u1>, !f32, tensor<1x7x7x2048x!eltwise.f32>) -> tensor<1x7x7x2048x!eltwise.f32>
    %168 = tile.cion add, none, %cst, %167 {sink = #map8, srcs = [#map9]} : !f32, tensor<1x7x7x2048x!eltwise.f32> -> tensor<1x2048x!eltwise.f32>
    %169 = "eltwise.div"(%168, %c49) : (tensor<1x2048x!eltwise.f32>, !i32) -> tensor<1x2048x!eltwise.f32>
    %170 = tile.cion add, mul, %cst, %169, %arg1 {sink = #map10, srcs = [#map11, #map12]} : !f32, tensor<1x2048x!eltwise.f32>, tensor<2048x1000x!eltwise.f32> -> tensor<1x1000x!eltwise.f32>
    %171 = "eltwise.add"(%170, %arg0) : (tensor<1x1000x!eltwise.f32>, tensor<1000x!eltwise.f32>) -> tensor<1x1000x!eltwise.f32>
    %172 = "eltwise.ident"(%171) : (tensor<1x1000x!eltwise.f32>) -> tensor<1x1000x!eltwise.f32>
    %173 = tile.cion max, none, %cst, %172 {sink = #map13, srcs = [#map14]} : !f32, tensor<1x1000x!eltwise.f32> -> tensor<1x1x!eltwise.f32>
    %174 = "eltwise.sub"(%172, %173) : (tensor<1x1000x!eltwise.f32>, tensor<1x1x!eltwise.f32>) -> tensor<1x1000x!eltwise.f32>
    %175 = "eltwise.exp"(%174) : (tensor<1x1000x!eltwise.f32>) -> tensor<1x1000x!eltwise.f32>
    %176 = tile.cion add, none, %cst, %175 {sink = #map13, srcs = [#map14]} : !f32, tensor<1x1000x!eltwise.f32> -> tensor<1x1x!eltwise.f32>
    %177 = "eltwise.div"(%175, %176) : (tensor<1x1000x!eltwise.f32>, tensor<1x1x!eltwise.f32>) -> tensor<1x1000x!eltwise.f32>
    return %177 : tensor<1x1000x!eltwise.f32>
  }
}
