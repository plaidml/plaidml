// RUN: pmlc-opt -pxa-test-indirect-uses-iterator -split-input-file %s | FileCheck %s --check-prefix=USES
// RUN: pmlc-opt -pxa-test-indirect-values-iterator -split-input-file %s | FileCheck %s --check-prefix=VALUES

func @simple(%arg0: memref<3xf32>) -> memref<3xf32> {
  %zero = constant 0.0 : f32
  // USES: alloc: %0
  // VALUES: alloc: %0
  %0 = alloc() {tag="%0"} : memref<3xf32>
  %1 = affine.parallel (%i) = (0) to (3) reduce ("assign") -> (memref<3xf32>) {
    // USES: use: %1
    // VALUES: def: %1
    %2 = pxa.reduce assign %zero, %0[%i] {tag="%1"} : memref<3xf32>
    // USES: use: %2
    affine.yield {tag="%2"} %2 : memref<3xf32>
  } {tag="%3"}
  // VALUES: def: %3
  // USES: use: %4
  return {tag="%4"} %1 : memref<3xf32>
  // USES-NEXT: alloc end: %0
  // VALUES-NEXT: alloc end: %0
}

// -----

#set0 = affine_set<(d0, d1, d2, d3, d4, d5) : (d3 - d4 * 64 - d5 * 128 >= 0, -d3 + d4 * 64 + d5 * 128 + 63 >= 0)>

func @complex(%arg0: memref<1x26x26x64xi8>) -> memref<1x13x13x256xi8> {
  %c0_i8 = constant 0 : i8
  // USES: alloc: %0
  // VALUES: alloc: %0
  %0 = alloc() {tag="%0"} : memref<1x13x13x256xi8>
  %1 = affine.parallel (%arg1, %arg2, %arg3, %arg4) = (0, 0, 0, 0) to (1, 13, 13, 256) reduce ("assign") -> (memref<1x13x13x256xi8>) {
    // USES-DAG: use: %1
    // VALUES-DAG: def: %1
    %3 = pxa.reduce assign %c0_i8, %0[%arg1, %arg2, %arg3, %arg4] {tag="%1"} : memref<1x13x13x256xi8>
    // USES-DAG: use: %2
    affine.yield {tag="%2"} %3 : memref<1x13x13x256xi8>
  } {tag="%3"}
  // VALUES-DAG: def: %3
  %2 = affine.parallel (%arg1, %arg2, %arg3, %arg4, %arg5, %arg6) = (0, 0, 0, 0, 0, 0) to (1, 13, 13, 256, 2, 2) reduce ("assign") -> (memref<1x13x13x256xi8>) {
    %3 = affine.if #set0(%arg1, %arg2, %arg3, %arg4, %arg5, %arg6) -> memref<1x13x13x256xi8> {
      %4 = affine.load %arg0[%arg1, %arg6 + %arg2 * 2, %arg5 + %arg3 * 2, %arg6 * -128 + %arg5 * -64 + %arg4] : memref<1x26x26x64xi8>
      // USES-DAG: use: %4
      // VALUES-DAG: def: %4
      %5 = pxa.reduce assign %4, %1[%arg1, %arg2, %arg3, %arg4] {tag="%4"} : memref<1x13x13x256xi8>
      // USES-DAG: use: %5
      affine.yield {tag="%5"} %5 : memref<1x13x13x256xi8>
    } else {
      // USES-DAG: use: %6
      affine.yield {tag="%6"} %1 : memref<1x13x13x256xi8>
    } {tag="%7"}
    // VALUES-DAG: def: %7
    // USES-DAG: use: %8
    affine.yield {tag="%8"} %3 : memref<1x13x13x256xi8>
  } {tag="%9"}
  // VALUES-DAG: def: %9
  // USES: use: %10
  return {tag="%10"} %2 : memref<1x13x13x256xi8>
  // USES-NEXT: alloc end: %0
  // VALUES-NEXT: alloc end: %0
}
