// RUN: pmlc-opt -affinex-memref-dataflow-opt %s | FileCheck %s

// CHECK: simple
func @simple(%arg: memref<1xf32>) {
  %cst = constant 0.000000e+00 : f32
  // CHECK: affine.store
  affine.store %cst, %arg[0] : memref<1xf32>
  // CHECK-NOT: affine.load
  %0 = affine.load %arg[0] : memref<1xf32>
  // CHECK: addf %cst, %cst
  %1 = addf %0, %0 : f32
  return
}

// CHECK: no_store
func @no_store(%arg: memref<1xf32>) {
  %cst = constant 0.000000e+00 : f32
  // CHECK: affine.load
  %0 = affine.load %arg[1] : memref<1xf32>
  // CHECK: addf %0, %0
  %1 = addf %0, %0 : f32
  return
}

// CHECK: re_store
func @re_store(%arg: memref<1xf32>) {
  %cst = constant 0.000000e+00 : f32
  // CHECK-NOT: affine.store
  affine.store %cst, %arg[0] : memref<1xf32>
  %cst_0 = constant 1.000000e+00 : f32
  // CHECK: affine.store
  affine.store %cst_0, %arg[0] : memref<1xf32>
  // CHECK-NOT: affine.load
  %0 = affine.load %arg[0] : memref<1xf32>
  // CHECK: addf %cst_0, %cst_0
  %1 = addf %0, %0 : f32
  return
}

// CHECK: location_tracking
func @location_tracking(%arg: memref<2xf32>) {
  %cst = constant 0.000000e+00 : f32
  // CHECK: affine.store
  affine.store %cst, %arg[0] : memref<2xf32>
  // CHECK: affine.load
  %0 = affine.load %arg[1] : memref<2xf32>
  // CHECK: addf %0, %0
  %1 = addf %0, %0 : f32
  return
}

// CHECK: memref_tracking
func @memref_tracking(%arg0: memref<1xf32>, %arg1: memref<1xf32>) {
  %cst = constant 0.000000e+00 : f32
  // CHECK: affine.store
  affine.store %cst, %arg0[0] : memref<1xf32>
  // CHECK: affine.load
  %0 = affine.load %arg1[0] : memref<1xf32>
  // CHECK: addf %0, %0
  %1 = addf %0, %0 : f32
  return
}

// CHECK: multi_location
func @multi_location(%arg: memref<2xf32>) {
  %cst = constant 0.000000e+00 : f32
  // CHECK: affine.store
  affine.store %cst, %arg[0] : memref<2xf32>
  %cst_0 = constant 1.000000e+00 : f32
  // CHECK: affine.store
  affine.store %cst_0, %arg[1] : memref<2xf32>
  // CHECK-NOT: affine.load
  %0 = affine.load %arg[0] : memref<2xf32>
  // CHECK: addf %cst, %cst
  %1 = addf %0, %0 : f32
  // CHECK-NOT: affine.load
  %2 = affine.load %arg[1] : memref<2xf32>
  // CHECK: addf %cst_0, %cst_0
  %3 = addf %2, %2 : f32
  return
}

// CHECK: multi_memref
func @multi_memref(%arg0: memref<1xf32>, %arg1: memref<1xf32>) {
  %cst = constant 0.000000e+00 : f32
  // CHECK: affine.store
  affine.store %cst, %arg0[0] : memref<1xf32>
  %cst_0 = constant 1.000000e+00 : f32
  // CHECK: affine.store
  affine.store %cst_0, %arg1[0] : memref<1xf32>
  // CHECK-NOT: affine.load
  %0 = affine.load %arg0[0] : memref<1xf32>
  // CHECK: addf %cst, %cst
  %1 = addf %0, %0 : f32
  // CHECK-NOT: affine.load
  %2 = affine.load %arg1[0] : memref<1xf32>
  // CHECK: addf %cst_0, %cst_0
  %3 = addf %2, %2 : f32
  return
}

// CHECK: remove_alloc
func @remove_alloc () {
  %cst = constant 0.000000e+00 : f32
  // CHECK-NOT: alloc
  %0 = alloc() : memref<1xf32>
  // CHECK-NOT: affine.store
  affine.store %cst, %0[0] : memref<1xf32>
  // CHECK-NOT: affine.load
  %1 = affine.load %0[0] : memref<1xf32>
  // CHECK: addf %cst, %cst
  %2 = addf %1, %1 : f32
  return
}

// CHECK: multi_block
func @multi_block(%arg: memref<1xf32>) {
  %cst = constant 0.000000e+00 : f32
  affine.for %i = 0 to 1 {
    // CHECK: affine.store
    affine.store %cst, %arg[0] : memref<1xf32>
    // CHECK-NOT: affine.load
    %0 = affine.load %arg[0] : memref<1xf32>
    // CHECK: addf %cst, %cst
    %1 = addf %0, %0 : f32
  }
  affine.for %i = 0 to 1 {
    // CHECK: affine.store
    affine.store %cst, %arg[0] : memref<1xf32>
    // CHECK-NOT: affine.load
    %0 = affine.load %arg[0] : memref<1xf32>
    // CHECK: addf %cst, %cst
    %1 = addf %0, %0 : f32
  }
  // CHECK: affine.store
  affine.store %cst, %arg[0] : memref<1xf32>
  // CHECK-NOT: affine.load
  %0 = affine.load %arg[0] : memref<1xf32>
  // CHECK: addf %cst, %cst
  %1 = addf %0, %0 : f32
  return
}

// CHECK: multi_block_neg
func @multi_block_neg(%arg: memref<1xf32>) {
  %cst = constant 0.000000e+00 : f32
  affine.for %i = 0 to 1 {
    // CHECK: affine.store
    affine.store %cst, %arg[0] : memref<1xf32>
  }
  affine.for %i = 0 to 1 {
    // CHECK: affine.load
    %0 = affine.load %arg[0] : memref<1xf32>
    // CHECK: addf %0, %0
    %1 = addf %0, %0 : f32
  }
  return
}

// CHECK: res2a_accum
func @res2a_accum(%arg0: memref<1x56x56x64xf32>, %arg1: memref<1x1x64x64xf32>, %arg2: memref<1x56x56x64xf32>) {
  %c0 = constant 0 : index
  %cst = constant dense<0.000000e+00> : vector<16xf32>
  %0 = alloc() : memref<1x1x8x1xvector<16xf32>>
  affine.store %cst, %0[0, 0, 0, 0] : memref<1x1x8x1xvector<16xf32>>
  affine.parallel (%arg3, %arg4) = (0, 0) to (56, 7) {
    affine.parallel (%arg5) = (0) to (4) {
      // CHECK-NOT: affine.vector_store
      affine.vector_store %cst, %arg2[0, %arg3, %arg4 * 8 + %c0, %arg5 * 16] : memref<1x56x56x64xf32>, vector<16xf32>
      %1 = affine.load %0[0, 0, 0, 0] : memref<1x1x8x1xvector<16xf32>>
      // CHECK-NOT: affine.vector_load
      %2 = affine.vector_load %arg2[0, %arg3, %arg4 * 8 + %c0, %arg5 * 16] : memref<1x56x56x64xf32>, vector<16xf32>
      // CHECK: addf %cst
      %3 = addf %2, %1 : vector<16xf32>
      affine.vector_store %3, %arg2[0, %arg3, %arg4 * 8 + %c0, %arg5 * 16] : memref<1x56x56x64xf32>, vector<16xf32>
    } {tags = {gpuThread, subgroupSize = 16 : i64}}
  } {tags = {gpuBlock, subgroupSize = 16 : i64}}
  return
}

// CHECK: re2a2a
module @get_value {
func @re2a2a(%arg0: memref<1x56x56x64xf32>, %arg1: memref<1x1x64x64xf32>, %arg2: memref<1x56x56x64xf32>) {
  %c0 = constant 0 : index
  %cst = constant dense<0.000000e+00> : vector<16xf32>
  %c8 = constant 8 : index
  %c9 = constant 9 : index
  %c10 = constant 10 : index
  %c11 = constant 11 : index
  %c12 = constant 12 : index
  %c13 = constant 13 : index
  %c14 = constant 14 : index
  %c15 = constant 15 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %c4 = constant 4 : index
  %c5 = constant 5 : index
  %c6 = constant 6 : index
  %c7 = constant 7 : index
  affine.parallel (%arg3, %arg4) = (0, 0) to (56, 7) {
    affine.parallel (%arg5) = (0) to (4) {
      %0 = alloc() : memref<1x1x8x1xvector<16xf32>>
      affine.store %cst, %0[0, 0, 0, 0] : memref<1x1x8x1xvector<16xf32>>
      affine.store %cst, %0[0, 0, 1, 0] : memref<1x1x8x1xvector<16xf32>>
      affine.store %cst, %0[0, 0, 2, 0] : memref<1x1x8x1xvector<16xf32>>
      affine.store %cst, %0[0, 0, 3, 0] : memref<1x1x8x1xvector<16xf32>>
      affine.store %cst, %0[0, 0, 4, 0] : memref<1x1x8x1xvector<16xf32>>
      affine.store %cst, %0[0, 0, 5, 0] : memref<1x1x8x1xvector<16xf32>>
      affine.store %cst, %0[0, 0, 6, 0] : memref<1x1x8x1xvector<16xf32>>
      affine.store %cst, %0[0, 0, 7, 0] : memref<1x1x8x1xvector<16xf32>>
      affine.for %arg6 = 0 to 4 {
        %25 = alloc() : memref<1x1x16x1xvector<16xf32>>
        %26 = affine.vector_load %arg1[0, 0, %arg6 * 16 + %c0, %arg5 * 16] : memref<1x1x64x64xf32>, vector<16xf32>
        affine.store %26, %25[0, 0, 0, 0] : memref<1x1x16x1xvector<16xf32>>
        %27 = affine.vector_load %arg1[0, 0, %arg6 * 16 + %c1, %arg5 * 16] : memref<1x1x64x64xf32>, vector<16xf32>
        affine.store %27, %25[0, 0, 1, 0] : memref<1x1x16x1xvector<16xf32>>
        %28 = affine.vector_load %arg1[0, 0, %arg6 * 16 + %c2, %arg5 * 16] : memref<1x1x64x64xf32>, vector<16xf32>
        affine.store %28, %25[0, 0, 2, 0] : memref<1x1x16x1xvector<16xf32>>
        %29 = affine.vector_load %arg1[0, 0, %arg6 * 16 + %c3, %arg5 * 16] : memref<1x1x64x64xf32>, vector<16xf32>
        affine.store %29, %25[0, 0, 3, 0] : memref<1x1x16x1xvector<16xf32>>
        %30 = affine.vector_load %arg1[0, 0, %arg6 * 16 + %c4, %arg5 * 16] : memref<1x1x64x64xf32>, vector<16xf32>
        affine.store %30, %25[0, 0, 4, 0] : memref<1x1x16x1xvector<16xf32>>
        %31 = affine.vector_load %arg1[0, 0, %arg6 * 16 + %c5, %arg5 * 16] : memref<1x1x64x64xf32>, vector<16xf32>
        affine.store %31, %25[0, 0, 5, 0] : memref<1x1x16x1xvector<16xf32>>
        %32 = affine.vector_load %arg1[0, 0, %arg6 * 16 + %c6, %arg5 * 16] : memref<1x1x64x64xf32>, vector<16xf32>
        affine.store %32, %25[0, 0, 6, 0] : memref<1x1x16x1xvector<16xf32>>
        %33 = affine.vector_load %arg1[0, 0, %arg6 * 16 + %c7, %arg5 * 16] : memref<1x1x64x64xf32>, vector<16xf32>
        affine.store %33, %25[0, 0, 7, 0] : memref<1x1x16x1xvector<16xf32>>
        %34 = affine.vector_load %arg1[0, 0, %arg6 * 16 + %c8, %arg5 * 16] : memref<1x1x64x64xf32>, vector<16xf32>
        affine.store %34, %25[0, 0, 8, 0] : memref<1x1x16x1xvector<16xf32>>
        %35 = affine.vector_load %arg1[0, 0, %arg6 * 16 + %c9, %arg5 * 16] : memref<1x1x64x64xf32>, vector<16xf32>
        affine.store %35, %25[0, 0, 9, 0] : memref<1x1x16x1xvector<16xf32>>
        %36 = affine.vector_load %arg1[0, 0, %arg6 * 16 + %c10, %arg5 * 16] : memref<1x1x64x64xf32>, vector<16xf32>
        affine.store %36, %25[0, 0, 10, 0] : memref<1x1x16x1xvector<16xf32>>
        %37 = affine.vector_load %arg1[0, 0, %arg6 * 16 + %c11, %arg5 * 16] : memref<1x1x64x64xf32>, vector<16xf32>
        affine.store %37, %25[0, 0, 11, 0] : memref<1x1x16x1xvector<16xf32>>
        %38 = affine.vector_load %arg1[0, 0, %arg6 * 16 + %c12, %arg5 * 16] : memref<1x1x64x64xf32>, vector<16xf32>
        affine.store %38, %25[0, 0, 12, 0] : memref<1x1x16x1xvector<16xf32>>
        %39 = affine.vector_load %arg1[0, 0, %arg6 * 16 + %c13, %arg5 * 16] : memref<1x1x64x64xf32>, vector<16xf32>
        affine.store %39, %25[0, 0, 13, 0] : memref<1x1x16x1xvector<16xf32>>
        %40 = affine.vector_load %arg1[0, 0, %arg6 * 16 + %c14, %arg5 * 16] : memref<1x1x64x64xf32>, vector<16xf32>
        affine.store %40, %25[0, 0, 14, 0] : memref<1x1x16x1xvector<16xf32>>
        %41 = affine.vector_load %arg1[0, 0, %arg6 * 16 + %c15, %arg5 * 16] : memref<1x1x64x64xf32>, vector<16xf32>
        affine.store %41, %25[0, 0, 15, 0] : memref<1x1x16x1xvector<16xf32>>
        affine.for %arg7 = 0 to 8 {
          %42 = affine.vector_load %arg0[0, %arg3, %arg4 * 8 + %arg7, %arg6 * 16] : memref<1x56x56x64xf32>, vector<16xf32>
          %43 = extract_element %42[%c0] : vector<16xf32>
          %44 = affine.load %25[0, 0, 0, 0] : memref<1x1x16x1xvector<16xf32>>
          %45 = vector.broadcast %43 : f32 to vector<16xf32>
          %46 = mulf %45, %44 : vector<16xf32>
          %47 = affine.load %0[0, 0, %arg7, 0] : memref<1x1x8x1xvector<16xf32>>
          %48 = addf %47, %46 : vector<16xf32>
          %49 = extract_element %42[%c1] : vector<16xf32>
          %50 = affine.load %25[0, 0, 1, 0] : memref<1x1x16x1xvector<16xf32>>
          %51 = vector.broadcast %49 : f32 to vector<16xf32>
          %52 = mulf %51, %50 : vector<16xf32>
          %53 = addf %48, %52 : vector<16xf32>
          %54 = extract_element %42[%c2] : vector<16xf32>
          %55 = affine.load %25[0, 0, 2, 0] : memref<1x1x16x1xvector<16xf32>>
          %56 = vector.broadcast %54 : f32 to vector<16xf32>
          %57 = mulf %56, %55 : vector<16xf32>
          %58 = addf %53, %57 : vector<16xf32>
          %59 = extract_element %42[%c3] : vector<16xf32>
          %60 = affine.load %25[0, 0, 3, 0] : memref<1x1x16x1xvector<16xf32>>
          %61 = vector.broadcast %59 : f32 to vector<16xf32>
          %62 = mulf %61, %60 : vector<16xf32>
          %63 = addf %58, %62 : vector<16xf32>
          %64 = extract_element %42[%c4] : vector<16xf32>
          %65 = affine.load %25[0, 0, 4, 0] : memref<1x1x16x1xvector<16xf32>>
          %66 = vector.broadcast %64 : f32 to vector<16xf32>
          %67 = mulf %66, %65 : vector<16xf32>
          %68 = addf %63, %67 : vector<16xf32>
          %69 = extract_element %42[%c5] : vector<16xf32>
          %70 = affine.load %25[0, 0, 5, 0] : memref<1x1x16x1xvector<16xf32>>
          %71 = vector.broadcast %69 : f32 to vector<16xf32>
          %72 = mulf %71, %70 : vector<16xf32>
          %73 = addf %68, %72 : vector<16xf32>
          %74 = extract_element %42[%c6] : vector<16xf32>
          %75 = affine.load %25[0, 0, 6, 0] : memref<1x1x16x1xvector<16xf32>>
          %76 = vector.broadcast %74 : f32 to vector<16xf32>
          %77 = mulf %76, %75 : vector<16xf32>
          %78 = addf %73, %77 : vector<16xf32>
          %79 = extract_element %42[%c7] : vector<16xf32>
          %80 = affine.load %25[0, 0, 7, 0] : memref<1x1x16x1xvector<16xf32>>
          %81 = vector.broadcast %79 : f32 to vector<16xf32>
          %82 = mulf %81, %80 : vector<16xf32>
          %83 = addf %78, %82 : vector<16xf32>
          %84 = extract_element %42[%c8] : vector<16xf32>
          %85 = affine.load %25[0, 0, 8, 0] : memref<1x1x16x1xvector<16xf32>>
          %86 = vector.broadcast %84 : f32 to vector<16xf32>
          %87 = mulf %86, %85 : vector<16xf32>
          %88 = addf %83, %87 : vector<16xf32>
          %89 = extract_element %42[%c9] : vector<16xf32>
          %90 = affine.load %25[0, 0, 9, 0] : memref<1x1x16x1xvector<16xf32>>
          %91 = vector.broadcast %89 : f32 to vector<16xf32>
          %92 = mulf %91, %90 : vector<16xf32>
          %93 = addf %88, %92 : vector<16xf32>
          %94 = extract_element %42[%c10] : vector<16xf32>
          %95 = affine.load %25[0, 0, 10, 0] : memref<1x1x16x1xvector<16xf32>>
          %96 = vector.broadcast %94 : f32 to vector<16xf32>
          %97 = mulf %96, %95 : vector<16xf32>
          %98 = addf %93, %97 : vector<16xf32>
          %99 = extract_element %42[%c11] : vector<16xf32>
          %100 = affine.load %25[0, 0, 11, 0] : memref<1x1x16x1xvector<16xf32>>
          %101 = vector.broadcast %99 : f32 to vector<16xf32>
          %102 = mulf %101, %100 : vector<16xf32>
          %103 = addf %98, %102 : vector<16xf32>
          %104 = extract_element %42[%c12] : vector<16xf32>
          %105 = affine.load %25[0, 0, 12, 0] : memref<1x1x16x1xvector<16xf32>>
          %106 = vector.broadcast %104 : f32 to vector<16xf32>
          %107 = mulf %106, %105 : vector<16xf32>
          %108 = addf %103, %107 : vector<16xf32>
          %109 = extract_element %42[%c13] : vector<16xf32>
          %110 = affine.load %25[0, 0, 13, 0] : memref<1x1x16x1xvector<16xf32>>
          %111 = vector.broadcast %109 : f32 to vector<16xf32>
          %112 = mulf %111, %110 : vector<16xf32>
          %113 = addf %108, %112 : vector<16xf32>
          %114 = extract_element %42[%c14] : vector<16xf32>
          %115 = affine.load %25[0, 0, 14, 0] : memref<1x1x16x1xvector<16xf32>>
          %116 = vector.broadcast %114 : f32 to vector<16xf32>
          %117 = mulf %116, %115 : vector<16xf32>
          %118 = addf %113, %117 : vector<16xf32>
          %119 = extract_element %42[%c15] : vector<16xf32>
          %120 = affine.load %25[0, 0, 15, 0] : memref<1x1x16x1xvector<16xf32>>
          %121 = vector.broadcast %119 : f32 to vector<16xf32>
          %122 = mulf %121, %120 : vector<16xf32>
          %123 = addf %118, %122 : vector<16xf32>
          affine.store %123, %0[0, 0, %arg7, 0] : memref<1x1x8x1xvector<16xf32>>
        }
      }
      affine.vector_store %cst, %arg2[0, %arg3, %arg4 * 8 + %c0, %arg5 * 16] : memref<1x56x56x64xf32>, vector<16xf32>
      %1 = affine.load %0[0, 0, 0, 0] : memref<1x1x8x1xvector<16xf32>>
      %2 = affine.vector_load %arg2[0, %arg3, %arg4 * 8 + %c0, %arg5 * 16] : memref<1x56x56x64xf32>, vector<16xf32>
      %3 = addf %2, %1 : vector<16xf32>
      // CHECK: affine.vector_store
      affine.vector_store %3, %arg2[0, %arg3, %arg4 * 8 + %c0, %arg5 * 16] : memref<1x56x56x64xf32>, vector<16xf32>
      // CHECK: blarg
      affine.vector_store %cst, %arg2[0, %arg3, %arg4 * 8 + %c1, %arg5 * 16] : memref<1x56x56x64xf32>, vector<16xf32>
      %4 = affine.load %0[0, 0, 1, 0] : memref<1x1x8x1xvector<16xf32>>
      %5 = affine.vector_load %arg2[0, %arg3, %arg4 * 8 + %c1, %arg5 * 16] : memref<1x56x56x64xf32>, vector<16xf32>
      %6 = addf %5, %4 : vector<16xf32>
      affine.vector_store %6, %arg2[0, %arg3, %arg4 * 8 + %c1, %arg5 * 16] : memref<1x56x56x64xf32>, vector<16xf32>
      affine.vector_store %cst, %arg2[0, %arg3, %arg4 * 8 + %c2, %arg5 * 16] : memref<1x56x56x64xf32>, vector<16xf32>
      %7 = affine.load %0[0, 0, 2, 0] : memref<1x1x8x1xvector<16xf32>>
      %8 = affine.vector_load %arg2[0, %arg3, %arg4 * 8 + %c2, %arg5 * 16] : memref<1x56x56x64xf32>, vector<16xf32>
      %9 = addf %8, %7 : vector<16xf32>
      affine.vector_store %9, %arg2[0, %arg3, %arg4 * 8 + %c2, %arg5 * 16] : memref<1x56x56x64xf32>, vector<16xf32>
      affine.vector_store %cst, %arg2[0, %arg3, %arg4 * 8 + %c3, %arg5 * 16] : memref<1x56x56x64xf32>, vector<16xf32>
      %10 = affine.load %0[0, 0, 3, 0] : memref<1x1x8x1xvector<16xf32>>
      %11 = affine.vector_load %arg2[0, %arg3, %arg4 * 8 + %c3, %arg5 * 16] : memref<1x56x56x64xf32>, vector<16xf32>
      %12 = addf %11, %10 : vector<16xf32>
      affine.vector_store %12, %arg2[0, %arg3, %arg4 * 8 + %c3, %arg5 * 16] : memref<1x56x56x64xf32>, vector<16xf32>
      affine.vector_store %cst, %arg2[0, %arg3, %arg4 * 8 + %c4, %arg5 * 16] : memref<1x56x56x64xf32>, vector<16xf32>
      %13 = affine.load %0[0, 0, 4, 0] : memref<1x1x8x1xvector<16xf32>>
      %14 = affine.vector_load %arg2[0, %arg3, %arg4 * 8 + %c4, %arg5 * 16] : memref<1x56x56x64xf32>, vector<16xf32>
      %15 = addf %14, %13 : vector<16xf32>
      affine.vector_store %15, %arg2[0, %arg3, %arg4 * 8 + %c4, %arg5 * 16] : memref<1x56x56x64xf32>, vector<16xf32>
      affine.vector_store %cst, %arg2[0, %arg3, %arg4 * 8 + %c5, %arg5 * 16] : memref<1x56x56x64xf32>, vector<16xf32>
      %16 = affine.load %0[0, 0, 5, 0] : memref<1x1x8x1xvector<16xf32>>
      %17 = affine.vector_load %arg2[0, %arg3, %arg4 * 8 + %c5, %arg5 * 16] : memref<1x56x56x64xf32>, vector<16xf32>
      %18 = addf %17, %16 : vector<16xf32>
      affine.vector_store %18, %arg2[0, %arg3, %arg4 * 8 + %c5, %arg5 * 16] : memref<1x56x56x64xf32>, vector<16xf32>
      affine.vector_store %cst, %arg2[0, %arg3, %arg4 * 8 + %c6, %arg5 * 16] : memref<1x56x56x64xf32>, vector<16xf32>
      %19 = affine.load %0[0, 0, 6, 0] : memref<1x1x8x1xvector<16xf32>>
      %20 = affine.vector_load %arg2[0, %arg3, %arg4 * 8 + %c6, %arg5 * 16] : memref<1x56x56x64xf32>, vector<16xf32>
      %21 = addf %20, %19 : vector<16xf32>
      affine.vector_store %21, %arg2[0, %arg3, %arg4 * 8 + %c6, %arg5 * 16] : memref<1x56x56x64xf32>, vector<16xf32>
      affine.vector_store %cst, %arg2[0, %arg3, %arg4 * 8 + %c7, %arg5 * 16] : memref<1x56x56x64xf32>, vector<16xf32>
      %22 = affine.load %0[0, 0, 7, 0] : memref<1x1x8x1xvector<16xf32>>
      %23 = affine.vector_load %arg2[0, %arg3, %arg4 * 8 + %c7, %arg5 * 16] : memref<1x56x56x64xf32>, vector<16xf32>
      %24 = addf %23, %22 : vector<16xf32>
      affine.vector_store %24, %arg2[0, %arg3, %arg4 * 8 + %c7, %arg5 * 16] : memref<1x56x56x64xf32>, vector<16xf32>
    } {tags = {gpuThread, subgroupSize = 16 : i64}}
  } {tags = {gpuBlock, subgroupSize = 16 : i64}}
  return
}
}