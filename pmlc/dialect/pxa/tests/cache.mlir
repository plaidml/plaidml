// RUN: pmlc-opt -pxa-cache %s | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0 * 4 + d1)>

func @argmax_0(%arg0: memref<1x256x256x16xf16>, %arg1: memref<1x256x16xf16>) {
  %cst = constant 0xFC00 : f16
  %0 = affine.parallel (%arg2) = (0) to (64) reduce ("assign") -> (memref<1x256x16xf16>) {
    %1 = affine.parallel () = () to () reduce ("assign") -> (memref<1x256x16xf16>) {
      %2 = affine.parallel (%arg3, %arg4) = (0, 0) to (4, 16) reduce ("assign") -> (memref<1x256x16xf16>) {
        %3 = pxa.reduce assign %cst, %arg1[0, %arg2 * 4 + %arg3, %arg4] : memref<1x256x16xf16>
        %4 = affine.parallel (%arg5) = (0) to (256) reduce ("assign") -> (memref<1x256x16xf16>) {
          %5 = pxa.load %arg0[0, %arg5, %arg2 * 4 + %arg3, %arg4] : memref<1x256x256x16xf16>
          %6 = pxa.reduce maxf %5, %3[0, %arg2 * 4 + %arg3, %arg4] : memref<1x256x16xf16>
          affine.yield %6 : memref<1x256x16xf16>
        }
        affine.yield %4 : memref<1x256x16xf16>
      } {tags = {inner}}
      affine.yield %2 : memref<1x256x16xf16>
    } {tags = {middle}}
    affine.yield %1 : memref<1x256x16xf16>
  } {tags = {outer, outermost}}
  return
}

// CHECK-LABEL: func @argmax_0
// CHECK:       affine.parallel (%{{.*}}) = (0) to (64)
// CHECK:         %[[cache2:.*]] = memref.alloc() {cache} : memref<1x4x16xf16>
// CHECK:         %[[copy2:.*]] = affine.parallel (%{{.*}}, %{{.*}}, %{{.*}}) = (0, 0, 0) to (1, 4, 16)
// CHECK:           pxa.load %{{.*}}[{{.*}}] : memref<1x256x16xf16>
// CHECK:           pxa.reduce assign %{{.*}}, %[[cache2]][{{.*}}] : memref<1x4x16xf16>
// CHECK:         {cache_in}
// CHECK:         %[[cache1:.*]] = memref.alloc() {cache} : memref<1x256x4x16xf16>
// CHECK:         %[[copy1:.*]] = affine.parallel (%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) = (0, 0, 0, 0) to (1, 256, 4, 16)
// CHECK:           pxa.load %{{.*}}[{{.*}}] : memref<1x256x256x16xf16>
// CHECK:           pxa.reduce assign %{{.*}}, %[[cache1]][{{.*}}] : memref<1x256x4x16xf16>
// CHECK:         {cache_in}
// CHECK:         %[[middle:.*]] = affine.parallel () = () to ()
// CHECK:           affine.parallel (%{{.*}}, %{{.*}}) = (0, 0) to (4, 16)
// CHECK:             pxa.load %[[copy1]]
// CHECK:             pxa.reduce maxf %{{.*}}, %[[copy2]]
// CHECK:           {inner}
// CHECK:         {middle}
// CHECK:         affine.parallel (%{{.*}}, %{{.*}}, %{{.*}}) = (0, 0, 0) to (1, 4, 16)
// CHECK:           pxa.load %[[middle]][{{.*}}] : memref<1x4x16xf16>
// CHECK:           pxa.reduce assign %{{.*}}, %{{.*}}[{{.*}}] : memref<1x256x16xf16>
// CHECK:         {cache_out}
// CHECK:       {outer, outermost}

func @argmax_1(%arg0: memref<1x256x256x16xf16>, %arg1: memref<1x256x16xf16>, %arg2: memref<1x256x16xi32>) {
  %c0_i32 = constant 0 : i32
  %0 = affine.parallel () = () to () reduce ("assign") -> (memref<1x256x16xi32>) {
    %1 = affine.parallel (%arg3) = (0) to (64) reduce ("assign") -> (memref<1x256x16xi32>) {
      %2 = affine.parallel (%arg4) = (0) to (4) reduce ("assign") -> (memref<1x256x16xi32>) {
        %3 = affine.apply #map0(%arg3, %arg4)
        %4 = index_cast %3 : index to i32
        %5 = affine.parallel (%arg5, %arg6) = (0, 0) to (256, 16) reduce ("assign") -> (memref<1x256x16xi32>) {
          %6 = pxa.load %arg0[0, %arg3 * 4 + %arg4, %arg5, %arg6] : memref<1x256x256x16xf16>
          %7 = pxa.load %arg1[0, %arg5, %arg6] : memref<1x256x16xf16>
          %8 = cmpf "oeq", %6, %7 : f16
          %9 = select %8, %4, %c0_i32 : i32
          %10 = pxa.reduce maxu %9, %arg2[0, %arg5, %arg6] : memref<1x256x16xi32>
          affine.yield %10 : memref<1x256x16xi32>
        }
        affine.yield %5 : memref<1x256x16xi32>
      } {tags = {inner}}
      affine.yield %2 : memref<1x256x16xi32>
    } {tags = {middle}}
    affine.yield %1 : memref<1x256x16xi32>
  } {tags = {outer, outermost}}
  return
}

// CHECK-LABEL: func @argmax_1
// CHECK:       affine.parallel () = () to ()
// CHECK:         %[[cache2:.*]] = memref.alloc() {cache} : memref<1x256x16xi32>
// CHECK:         %[[copy2:.*]] = affine.parallel (%{{.*}}, %{{.*}}, %{{.*}}) = (0, 0, 0) to (1, 256, 16)
// CHECK:           pxa.load %{{.*}}[{{.*}}] : memref<1x256x16xi32>
// CHECK:           pxa.reduce assign %{{.*}}, %[[cache2]][{{.*}}] : memref<1x256x16xi32>
// CHECK:         {cache_in}
// CHECK:         %[[cache1:.*]] = memref.alloc() {cache} : memref<1x256x16xf16>
// CHECK:         %[[copy1:.*]] = affine.parallel (%{{.*}}, %{{.*}}, %{{.*}}) = (0, 0, 0) to (1, 256, 16)
// CHECK:           pxa.load %{{.*}}[{{.*}}] : memref<1x256x16xf16>
// CHECK:           pxa.reduce assign %{{.*}}, %[[cache1]][{{.*}}] : memref<1x256x16xf16>
// CHECK:         {cache_in}
// CHECK:         %[[middle:.*]] = affine.parallel (%{{.*}}) = (0) to (64)
// CHECK:           %[[cache3:.*]] = memref.alloc() {cache} : memref<1x4x256x16xf16>
// CHECK:           %[[copy3:.*]] = affine.parallel (%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) = (0, 0, 0, 0) to (1, 4, 256, 16)
// CHECK:             pxa.load %{{.*}}[{{.*}}] : memref<1x256x256x16xf16>
// CHECK:             pxa.reduce assign %{{.*}}, %[[cache3]][{{.*}}] : memref<1x4x256x16xf16>
// CHECK:           {cache_in}
// CHECK:           affine.parallel (%{{.*}}) = (0) to (4)
// CHECK:             affine.parallel (%{{.*}}, %{{.*}}) = (0, 0) to (256, 16)
// CHECK:               pxa.load %[[copy3]][{{.*}}] : memref<1x4x256x16xf16>
// CHECK:               pxa.load %[[copy1]][{{.*}}] : memref<1x256x16xf16>
// CHECK:               pxa.reduce maxu %{{.*}}, %[[copy2]][{{.*}}] : memref<1x256x16xi32>
// CHECK:           {inner}
// CHECK:         {middle}
// CHECK:         affine.parallel (%{{.*}}, %{{.*}}, %{{.*}}) = (0, 0, 0) to (1, 256, 16)
// CHECK:           pxa.load %[[middle]][{{.*}}] : memref<1x256x16xi32>
// CHECK:           pxa.reduce assign %{{.*}}, %{{.*}}[{{.*}}] : memref<1x256x16xi32>
// CHECK:         {cache_out}
// CHECK:       {outer, outermost}
