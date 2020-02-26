// RUN: pmlc-opt -convert-linalg-to-loops -convert-pxa-to-affine -lower-affine \
// RUN:     -convert-loop-to-std -convert-std-to-llvm %s | \
// RUN:   pmlc-jit -e baseline -entry-point-result=void | FileCheck %s
// RUN: pmlc-opt -convert-linalg-to-loops -convert-pxa-to-affine -lower-affine \
// RUN:     -convert-loop-to-std -convert-std-to-llvm %s | \
// RUN:   pmlc-jit -e tiled -entry-point-result=void | FileCheck %s
// RUN: pmlc-opt -convert-linalg-to-loops -convert-pxa-to-affine -lower-affine \
// RUN:     -convert-loop-to-std -convert-std-to-llvm %s | \
// RUN:   pmlc-jit -e xsmm -entry-point-result=void | FileCheck %s

func @print_memref_f32(memref<*xf32>)
func @plaidml_rt_xsmm_gemm_f32(memref<*xf32>, memref<*xf32>, memref<*xf32>, i32, i32, i32, i32, i32, i32)

func @baseline() {
  %dot = constant @dot : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  call @test_dot(%dot) : ((memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()) -> ()

  %conv2 = constant @conv2 : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>) -> ()
  call @test_conv2(%conv2) : ((memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>) -> ()) -> ()

  return
}

func @tiled() {
  %dot = constant @dot_tiled : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  call @test_dot(%dot) : ((memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()) -> ()

  %conv2 = constant @conv2_tiled : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>) -> ()
  call @test_conv2(%conv2) : ((memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>) -> ()) -> ()

  return
}

func @xsmm() {
  %dot = constant @dot_xsmm : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  call @test_dot(%dot) : ((memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()) -> ()

  %conv2 = constant @conv2_xsmm : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>) -> ()
  call @test_conv2(%conv2) : ((memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>) -> ()) -> ()

  return
}

func @fill_2d(%buf : memref<?x?xf32>, %alt : i1) {
  %c0 = constant 0 : index
  %c5 = constant 5 : index
  %X = dim %buf, 0 : memref<?x?xf32>
  %Y = dim %buf, 1 : memref<?x?xf32>
  affine.parallel (%x, %y) = (0, 0) to (%X, %Y) {
    // i = linear offset
    %i = affine.apply affine_map<(x, y)[Y] -> (x * Y + y)>(%x, %y)[%Y]
    // t = alt ? i : 0
    %t = select %alt, %i, %c0 : index
    // v = x + y + t - 5
    %1 = addi %x, %y : index
    %2 = addi %1, %t : index
    %v = subi %2, %c5 : index
    %v_i64 = index_cast %v : index to i64
    %v_f32 = sitofp %v_i64 : i64 to f32
    store %v_f32, %buf[%x, %y] : memref<?x?xf32>
  }
  return
}

func @fill_4d(%buf : memref<?x?x?x?xf32>, %alt : i1) {
  %c0 = constant 0 : index
  %c5 = constant 5 : index
  %X = dim %buf, 0 : memref<?x?x?x?xf32>
  %Y = dim %buf, 1 : memref<?x?x?x?xf32>
  %Z = dim %buf, 2 : memref<?x?x?x?xf32>
  %W = dim %buf, 3 : memref<?x?x?x?xf32>
  affine.parallel (%x, %y, %z, %w) = (0, 0, 0, 0) to (%X, %Y, %Z, %W) {
    // i = linear offset
    %i = affine.apply affine_map<(x, y, z, w)[Y, Z, W] -> (x * Y + y * W + z * W + w)>(%x, %y, %z, %w)[%Y, %Z, %W]
    // t = alt ? i : 0
    %t = select %alt, %i, %c0 : index
    %j = affine.apply affine_map<(x, y, z, w) -> (x + y + z + w)>(%x, %y, %z, %w)
    // v = j + t - 5
    %2 = addi %j, %t : index
    %v = subi %2, %c5 : index
    %v_i64 = index_cast %v : index to i64
    %v_f32 = sitofp %v_i64 : i64 to f32
    store %v_f32, %buf[%x, %y, %z, %w] : memref<?x?x?x?xf32>
  }
  return
}

func @test_dot(%impl : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()) {
  %false = constant 0 : i1
  %true = constant 1 : i1
  %f0 = constant 0.0 : f32
  %A = alloc() : memref<8x8xf32>
  %A_2d = memref_cast %A : memref<8x8xf32> to memref<?x?xf32>
  %A_ud = memref_cast %A : memref<8x8xf32> to memref<*xf32>
  call @fill_2d(%A_2d, %false) : (memref<?x?xf32>, i1) -> ()
  // call @print_memref_f32(%A_ud) : (memref<*xf32>) -> ()
  %B = alloc() : memref<8x8xf32>
  %B_2d = memref_cast %B : memref<8x8xf32> to memref<?x?xf32>
  %B_ud = memref_cast %B : memref<8x8xf32> to memref<*xf32>
  call @fill_2d(%B_2d, %true) : (memref<?x?xf32>, i1) -> ()
  // call @print_memref_f32(%B_ud) : (memref<*xf32>) -> ()
  %C = alloc() : memref<8x8xf32>
  %C_2d = memref_cast %C : memref<8x8xf32> to memref<?x?xf32>
  %C_ud = memref_cast %C : memref<8x8xf32> to memref<*xf32>

  linalg.fill(%C, %f0) : memref<8x8xf32>, f32
  call_indirect %impl(%A_2d, %B_2d, %C_2d) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  call @print_memref_f32(%C_ud) : (memref<*xf32>) -> ()
  // CHECK:  [60,   36,   12,   -12,   -36,   -60,   -84,   -108],
  // CHECK:  [272,   264,   256,   248,   240,   232,   224,   216],
  // CHECK:  [484,   492,   500,   508,   516,   524,   532,   540],
  // CHECK:  [696,   720,   744,   768,   792,   816,   840,   864],
  // CHECK:  [908,   948,   988,   1028,   1068,   1108,   1148,   1188],
  // CHECK:  [1120,   1176,   1232,   1288,   1344,   1400,   1456,   1512],
  // CHECK:  [1332,   1404,   1476,   1548,   1620,   1692,   1764,   1836],
  // CHECK:  [1544,   1632,   1720,   1808,   1896,   1984,   2072,   2160]

  dealloc %C : memref<8x8xf32>
  dealloc %B : memref<8x8xf32>
  dealloc %A : memref<8x8xf32>
  return
}

func @dot(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) {
  %M = dim %C, 0 : memref<?x?xf32>
  %N = dim %C, 1 : memref<?x?xf32>
  %K = dim %A, 1 : memref<?x?xf32>
  affine.parallel (%i, %j, %k) = (0, 0, 0) to (%M, %N, %K) {
    %0 = affine.load %A[%i, %k] : memref<?x?xf32>
    %1 = affine.load %B[%k, %j] : memref<?x?xf32>
    %2 = mulf %0, %1 : f32
    pxa.reduce add %2, %C[%i, %j] : memref<?x?xf32>
  }
  return
}

func @dot_tiled(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) {
  %M = dim %C, 0 : memref<?x?xf32>
  %N = dim %C, 1 : memref<?x?xf32>
  %K = dim %A, 1 : memref<?x?xf32>
  affine.parallel (%i0, %j0, %k0) = (0, 0, 0) to (%M, %N, %K) step (2, 2, 2) {
    affine.parallel (%i1, %j1, %k1) = (%i0, %j0, %k0) to (%i0 + 2, %j0 + 2, %k0 + 2) {
      %0 = affine.load %A[%i1, %k1] : memref<?x?xf32>
      %1 = affine.load %B[%k1, %j1] : memref<?x?xf32>
      %2 = mulf %0, %1 : f32
      pxa.reduce add %2, %C[%i1, %j1] : memref<?x?xf32>
    }
  }
  return
}

func @dot_xsmm(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) {
  %c0 = constant 0 : index
  %c2 = constant 2 : index
  %tile_m = constant 2 : i32
  %tile_n = constant 2 : i32
  %tile_k = constant 2 : i32
  %lda = constant 8 : i32
  %ldb = constant 8 : i32
  %ldc = constant 8 : i32
  %M = dim %C, 0 : memref<?x?xf32>
  %N = dim %C, 1 : memref<?x?xf32>
  %K = dim %A, 1 : memref<?x?xf32>
  affine.parallel (%i, %j, %k) = (0, 0, 0) to (%M, %N, %K) step (2, 2, 2) {
    %a_view = subview %A[%i, %k][][] : memref<?x?xf32> to memref<2x2xf32, offset: ?, strides: [?, ?]>
    %b_view = subview %B[%k, %j][][] : memref<?x?xf32> to memref<2x2xf32, offset: ?, strides: [?, ?]>
    %c_view = subview %C[%i, %j][][] : memref<?x?xf32> to memref<2x2xf32, offset: ?, strides: [?, ?]>
    %a_ref = memref_cast %a_view : memref<2x2xf32, offset: ?, strides: [?, ?]> to memref<*xf32>
    %b_ref = memref_cast %b_view : memref<2x2xf32, offset: ?, strides: [?, ?]> to memref<*xf32>
    %c_ref = memref_cast %c_view : memref<2x2xf32, offset: ?, strides: [?, ?]> to memref<*xf32>
    // NOTE: we need to swap the A, B and lda, ldb operands due to column-major vs row-major differences
    call @plaidml_rt_xsmm_gemm_f32(%b_ref, %a_ref, %c_ref, %ldb, %lda, %ldc, %tile_m, %tile_n, %tile_k)
      : (memref<*xf32>, memref<*xf32>, memref<*xf32>, i32, i32, i32, i32, i32, i32) -> ()
  }
  return
}

func @test_conv2(%impl : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>) -> ()) {
  %false = constant 0 : i1
  %true = constant 1 : i1
  %f0 = constant 0.0 : f32
  %I = alloc() : memref<1x6x5x7xf32>
  %I_2d = memref_cast %I : memref<1x6x5x7xf32> to memref<?x?x?x?xf32>
  %I_ud = memref_cast %I : memref<1x6x5x7xf32> to memref<*xf32>
  call @fill_4d(%I_2d, %false) : (memref<?x?x?x?xf32>, i1) -> ()
  // call @print_memref_f32(%I_ud) : (memref<*xf32>) -> ()
  %K = alloc() : memref<1x1x7x11xf32>
  %K_2d = memref_cast %K : memref<1x1x7x11xf32> to memref<?x?x?x?xf32>
  %K_ud = memref_cast %K : memref<1x1x7x11xf32> to memref<*xf32>
  call @fill_4d(%K_2d, %true) : (memref<?x?x?x?xf32>, i1) -> ()
  // call @print_memref_f32(%K_ud) : (memref<*xf32>) -> ()
  %O = alloc() : memref<1x6x5x11xf32>
  %O_2d = memref_cast %O : memref<1x6x5x11xf32> to memref<?x?x?x?xf32>
  %O_ud = memref_cast %O : memref<1x6x5x11xf32> to memref<*xf32>

  linalg.fill(%O, %f0) : memref<1x6x5x11xf32>, f32
  call_indirect %impl(%I_2d, %K_2d, %O_2d) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>) -> ()
  call @print_memref_f32(%O_ud) : (memref<*xf32>) -> ()
  // CHECK: [-98,     -126,     -154,     -182,     -210,     -238,     -266,     -294,     -322,     -350,     -378],
  // CHECK: [119,     105,     91,     77,     63,     49,     35,     21,     7,     -7,     -21],
  // CHECK: [336,     336,     336,     336,     336,     336,     336,     336,     336,     336,     336],
  // CHECK: [553,     567,     581,     595,     609,     623,     637,     651,     665,     679,     693],
  // CHECK: [770,     798,     826,     854,     882,     910,     938,     966,     994,     1022,     1050]],
  // CHECK: [119,     105,     91,     77,     63,     49,     35,     21,     7,     -7,     -21],
  // CHECK: [336,     336,     336,     336,     336,     336,     336,     336,     336,     336,     336],
  // CHECK: [553,     567,     581,     595,     609,     623,     637,     651,     665,     679,     693],
  // CHECK: [770,     798,     826,     854,     882,     910,     938,     966,     994,     1022,     1050],
  // CHECK: [987,     1029,     1071,     1113,     1155,     1197,     1239,     1281,     1323,     1365,     1407]],
  // CHECK: [336,     336,     336,     336,     336,     336,     336,     336,     336,     336,     336],
  // CHECK: [553,     567,     581,     595,     609,     623,     637,     651,     665,     679,     693],
  // CHECK: [770,     798,     826,     854,     882,     910,     938,     966,     994,     1022,     1050],
  // CHECK: [987,     1029,     1071,     1113,     1155,     1197,     1239,     1281,     1323,     1365,     1407],
  // CHECK: [1204,     1260,     1316,     1372,     1428,     1484,     1540,     1596,     1652,     1708,     1764]],
  // CHECK: [553,     567,     581,     595,     609,     623,     637,     651,     665,     679,     693],
  // CHECK: [770,     798,     826,     854,     882,     910,     938,     966,     994,     1022,     1050],
  // CHECK: [987,     1029,     1071,     1113,     1155,     1197,     1239,     1281,     1323,     1365,     1407],
  // CHECK: [1204,     1260,     1316,     1372,     1428,     1484,     1540,     1596,     1652,     1708,     1764],
  // CHECK: [1421,     1491,     1561,     1631,     1701,     1771,     1841,     1911,     1981,     2051,     2121]],
  // CHECK: [770,     798,     826,     854,     882,     910,     938,     966,     994,     1022,     1050],
  // CHECK: [987,     1029,     1071,     1113,     1155,     1197,     1239,     1281,     1323,     1365,     1407],
  // CHECK: [1204,     1260,     1316,     1372,     1428,     1484,     1540,     1596,     1652,     1708,     1764],
  // CHECK: [1421,     1491,     1561,     1631,     1701,     1771,     1841,     1911,     1981,     2051,     2121],
  // CHECK: [1638,     1722,     1806,     1890,     1974,     2058,     2142,     2226,     2310,     2394,     2478]],
  // CHECK: [987,     1029,     1071,     1113,     1155,     1197,     1239,     1281,     1323,     1365,     1407],
  // CHECK: [1204,     1260,     1316,     1372,     1428,     1484,     1540,     1596,     1652,     1708,     1764],
  // CHECK: [1421,     1491,     1561,     1631,     1701,     1771,     1841,     1911,     1981,     2051,     2121],
  // CHECK: [1638,     1722,     1806,     1890,     1974,     2058,     2142,     2226,     2310,     2394,     2478],
  // CHECK: [1855,     1953,     2051,     2149,     2247,     2345,     2443,     2541,     2639,     2737,     2835]]]]

  dealloc %O : memref<1x6x5x11xf32>
  dealloc %K : memref<1x1x7x11xf32>
  dealloc %I : memref<1x6x5x7xf32>
  return
}

func @conv2(%I: memref<?x?x?x?xf32>, %K: memref<?x?x?x?xf32>, %O: memref<?x?x?x?xf32>) {
  %X = dim %I, 1 : memref<?x?x?x?xf32>
  %Y = dim %I, 2 : memref<?x?x?x?xf32>
  %CI = dim %I, 3 : memref<?x?x?x?xf32>
  %CO = dim %O, 3 : memref<?x?x?x?xf32>
  affine.parallel (%x, %y, %ci, %co) = (0, 0, 0, 0) to (%X, %Y, %CI, %CO) {
    %0 = affine.load %I[0, %x, %y, %ci] : memref<?x?x?x?xf32>
    %1 = affine.load %K[0, 0, %ci, %co] : memref<?x?x?x?xf32>
    %2 = mulf %0, %1 : f32
    pxa.reduce add %2, %O[0, %x, %y, %co] : memref<?x?x?x?xf32>
  }
  return
}

func @conv2_tiled(%I: memref<?x?x?x?xf32>, %K: memref<?x?x?x?xf32>, %O: memref<?x?x?x?xf32>) {
  %X = dim %I, 1 : memref<?x?x?x?xf32>
  %Y = dim %I, 2 : memref<?x?x?x?xf32>
  %CI = dim %I, 3 : memref<?x?x?x?xf32>
  %CO = dim %O, 3 : memref<?x?x?x?xf32>
  affine.parallel (%x0, %y) = (0, 0) to (%X, %Y) step (2, 1) {
    affine.parallel (%x1, %ci, %co) = (%x0, 0, 0) to (%x0 + 2, %CI, %CO) {
      %0 = affine.load %I[0, %x1, %y, %ci] : memref<?x?x?x?xf32>
      %1 = affine.load %K[0, 0, %ci, %co] : memref<?x?x?x?xf32>
      %2 = mulf %0, %1 : f32
      pxa.reduce add %2, %O[0, %x1, %y, %co] : memref<?x?x?x?xf32>
    }
  }
  return
}

func @conv2_xsmm(%I: memref<?x?x?x?xf32>, %K: memref<?x?x?x?xf32>, %O: memref<?x?x?x?xf32>) {
  %c0 = constant 0 : index
  %X = dim %I, 1 : memref<?x?x?x?xf32>
  %Y = dim %I, 2 : memref<?x?x?x?xf32>
  %m = constant 11 : i32
  %n = constant 2 : i32
  %k = constant 7 : i32
  %lda = constant 35 : i32
  %ldb = constant 11 : i32
  %ldc = constant 55 : i32
  affine.parallel (%x, %y) = (0, 0) to (%X, %Y) step (2, 1) {
    // xsmm.gemm %O[%c0, %x, %y, %c0] = %K[%c0, %c0, %c0, %c0], %I[%c0, %x, %y, %c0], [co:11, x:2, ci:7]
    %I_view = subview %I[%c0,  %x,  %y, %c0][][] : memref<?x?x?x?xf32> to memref<1x3x5x7xf32, offset: ?, strides: [?, ?, ?, ?]>
    %K_view = subview %K[%c0, %c0, %c0, %c0][][] : memref<?x?x?x?xf32> to memref<1x1x7x11xf32, offset: ?, strides: [?, ?, ?, ?]>
    %O_view = subview %O[%c0,  %x,  %y, %c0][][] : memref<?x?x?x?xf32> to memref<1x3x5x11xf32, offset: ?, strides: [?, ?, ?, ?]>
    %I_ref = memref_cast %I_view : memref<1x3x5x7xf32, offset: ?, strides: [?, ?, ?, ?]> to memref<*xf32>
    %K_ref = memref_cast %K_view : memref<1x1x7x11xf32, offset: ?, strides: [?, ?, ?, ?]> to memref<*xf32>
    %O_ref = memref_cast %O_view : memref<1x3x5x11xf32, offset: ?, strides: [?, ?, ?, ?]> to memref<*xf32>
    // NOTE: we need to swap the A, B and lda, ldb operands due to column-major vs row-major differences
    call @plaidml_rt_xsmm_gemm_f32(%K_ref, %I_ref, %O_ref, %ldb, %lda, %ldc, %m, %n, %k)
      : (memref<*xf32>, memref<*xf32>, memref<*xf32>, i32, i32, i32, i32, i32, i32) -> ()
  }
  return
}
