// RUN: pmlc-opt -convert-tile-to-pxa -canonicalize -cse -split-input-file %s | FileCheck %s

func @gather1(%arg0: tensor<4xsi32>, %arg1: tensor<3x2xf32>) -> tensor<4x2xf32> {
  %0 = "tile.gather"(%arg1, %arg0) : (tensor<3x2xf32>, tensor<4xsi32>) -> tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}

// CHECK-LABEL: func @gather1
// CHECK: affine.parallel (%[[I:.*]], %[[J:.*]]) = (0, 0) to (4, 2)
// CHECK: %[[IDX_RAW:.*]] = pxa.load {{%.*}}[%[[I]]] : memref<4xi32>
// CHECK: %[[IDX:.*]] = index_cast %[[IDX_RAW]] : i32 to index
// CHECK: %[[SRC:.*]] = load %{{.*}}[%[[IDX]], %[[J]]] : memref<3x2xf32>
// CHECK: %[[OUT:.*]] = pxa.reduce assign %[[SRC]], %{{.*}}[%[[I]], %[[J]]] : memref<4x2xf32>
// CHECK: affine.yield %[[OUT]] : memref<4x2xf32>


func @gather2(%arg0: tensor<3x2xf32>, %arg1: tensor<4xf32>) -> tensor<3x4xf32> {
  %0 = tile.gather %arg0 %arg1 {axis = 1 : index, mode = 1 : i64} : (tensor<3x2xf32>, tensor<4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// CHECK-LABEL: func @gather2
// CHECK: affine.parallel (%[[I:.*]], %[[J:.*]]) = (0, 0) to (3, 4)
// CHECK: %[[IDX:.*]] = pxa.load {{%.*}}[%[[J]]] : memref<4xf32>
// CHECK: %[[FF:.*]] = floorf %[[IDX]] : f32
// CHECK: %[[FI:.*]] = fptosi %[[FF]] : f32 to i32
// CHECK: %[[CF:.*]] = ceilf %[[IDX]] : f32
// CHECK: %[[CI:.*]] = fptosi %[[CF]] : f32 to i32
// CHECK: %[[CMP0:.*]] = cmpi slt, %[[FI]], %{{.*}}: i32
// CHECK: %[[CMP1:.*]] = cmpi slt, %[[FI]], %{{.*}}: i32
// CHECK: %{{.*}} = select %[[CMP0]],  %{{.*}}, %[[FI]] : i32
// CHECK: %[[FLOOR:.*]] = select %[[CMP1]], %{{.*}},  %{{.*}} : i32
// CHECK: %[[CMP2:.*]] = cmpi slt, %[[CI]], %{{.*}}: i32
// CHECK: %[[CMP3:.*]] = cmpi slt, %[[CI]], %{{.*}}: i32
// CHECK: %{{.*}} = select %[[CMP2]],  %{{.*}}, %[[CI]] : i32
// CHECK: %[[CEIL:.*]] = select %[[CMP3]], %{{.*}},  %{{.*}} : i32
// CHECK: %[[FIDX:.*]] = index_cast %[[FLOOR]] : i32 to index
// CHECK: %[[CIDX:.*]] = index_cast %[[CEIL]] : i32 to index
// CHECK: %[[G1:.*]] = load %{{.*}}[%[[I]], %[[CIDX]]] : memref<3x2xf32>
// CHECK: %[[G0:.*]] = load %{{.*}}[%[[I]], %[[FIDX]]] : memref<3x2xf32>
// CHECK: %[[S1:.*]] = subf %[[IDX]], %[[FF]] : f32
// CHECK: %[[S0:.*]] = subf %{{.*}}, %[[S1]] : f32
// CHECK: %[[P1:.*]] = mulf %[[S1]], %[[G1]] : f32
// CHECK: %[[P0:.*]] = mulf %[[S0]], %[[G0]]  : f32
// CHECK: %[[SRC:.*]] = addf %[[P1]], %[[P0]] : f32
// CHECK: %[[OUT:.*]] = pxa.reduce assign %[[SRC]], %{{.*}}[%[[I]], %[[J]]] : memref<3x4xf32>
// CHECK: affine.yield %[[OUT]] : memref<3x4xf32>
