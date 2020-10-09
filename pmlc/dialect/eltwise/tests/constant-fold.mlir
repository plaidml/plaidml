// RUN: pmlc-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: @basic
func @basic(%arg0: f32) -> f32 {
  %cst = "eltwise.sconst"() {value = 1.0 : f32} : () -> f32
  %0 = "eltwise.add"(%arg0, %arg0) : (f32, f32) -> f32
  %1 = "eltwise.mul"(%0, %cst) : (f32, f32) -> f32
  return %1 : f32
  // CHECK-NEXT: "eltwise.add"(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
  // CHECK-NEXT: return %{{.*}}
}

// CHECK-LABEL: @fold_mul_1_f32
func @fold_mul_1_f32(%arg0: f32) -> f32 {
  %cst = "eltwise.sconst"() {value = 1.0 : f32} : () -> f32
  %0 = "eltwise.mul"(%arg0, %cst) : (f32, f32) -> f32
  return %0 : f32
  // CHECK-NEXT: return %{{.*}}
}

// CHECK-LABEL: @fold_mul_1_i32
func @fold_mul_1_i32(%arg0: si32) -> si32 {
  %cst = "eltwise.sconst"() {value = 1 : i32} : () -> si32
  %0 = "eltwise.mul"(%arg0, %cst) : (si32, si32) -> si32
  return %0 : si32
  // CHECK-NEXT: return %{{.*}}
}

// CHECK-LABEL: @fold_add_0_f32
func @fold_add_0_f32(%arg0: f32) -> f32 {
  %cst = "eltwise.sconst"() {value = 0.0 : f32} : () -> f32
  %0 = "eltwise.add"(%arg0, %cst) : (f32, f32) -> f32
  return %0 : f32
  // CHECK-NEXT: return %{{.*}}
}

// CHECK-LABEL: @fold_add_0_i32
func @fold_add_0_i32(%arg0: si32) -> si32 {
  %cst = "eltwise.sconst"() {value = 0 : i32} : () -> si32
  %0 = "eltwise.add"(%arg0, %cst) : (si32, si32) -> si32
  return %0 : si32
  // CHECK-NEXT: return %{{.*}}
}

// CHECK-LABEL: @fold_add_f32_f32
func @fold_add_f32_f32() -> f32 {
  %cst_0 = "eltwise.sconst"() {value = 1.0 : f32} : () -> f32
  %cst_1 = "eltwise.sconst"() {value = 3.0 : f32} : () -> f32
  %0 = "eltwise.add"(%cst_0, %cst_1) : (f32, f32) -> f32
  return %0 : f32
  // CHECK-NEXT: "eltwise.sconst"() {value = 4.000000e+00 : f32} : () -> f32
  // CHECK-NEXT: return %{{.*}}
}

// CHECK-LABEL: @fold_add_f32_i32
func @fold_add_f32_i32() -> f32 {
  %cst_0 = "eltwise.sconst"() {value = 1.0 : f32} : () -> f32
  %cst_1 = "eltwise.sconst"() {value = 3 : i32} : () -> si32
  %0 = "eltwise.add"(%cst_0, %cst_1) : (f32, si32) -> f32
  return %0 : f32
  // CHECK-NEXT: "eltwise.sconst"() {value = 4.000000e+00 : f32} : () -> f32
  // CHECK-NEXT: return %{{.*}} : f32
}

// CHECK-LABEL: @fold_add_i32_i32
func @fold_add_i32_i32() -> si32 {
  %cst_0 = "eltwise.sconst"() {value = 1 : i32} : () -> si32
  %cst_1 = "eltwise.sconst"() {value = 3 : i32} : () -> si32
  %0 = "eltwise.add"(%cst_0, %cst_1) : (si32, si32) -> si32
  return %0 : si32
  // CHECK-NEXT: "eltwise.sconst"() {value = 4 : i32} : () -> si32
  // CHECK-NEXT: return %{{.*}} : si32
}

// CHECK-LABEL: @fold_sub_f32_f32
func @fold_sub_f32_f32() -> f32 {
  %cst_0 = "eltwise.sconst"() {value = 1.0 : f32} : () -> f32
  %cst_1 = "eltwise.sconst"() {value = 3.0 : f32} : () -> f32
  %0 = "eltwise.sub"(%cst_0, %cst_1) : (f32, f32) -> f32
  return %0 : f32
  // CHECK-NEXT: "eltwise.sconst"() {value = -2.000000e+00 : f32} : () -> f32
  // CHECK-NEXT: return %{{.*}}
}

// CHECK-LABEL: @fold_sub_f32_i32
func @fold_sub_f32_i32() -> f32 {
  %cst_0 = "eltwise.sconst"() {value = 1.0 : f32} : () -> f32
  %cst_1 = "eltwise.sconst"() {value = 3 : i32} : () -> si32
  %0 = "eltwise.sub"(%cst_0, %cst_1) : (f32, si32) -> f32
  return %0 : f32
  // CHECK-NEXT: "eltwise.sconst"() {value = -2.000000e+00 : f32} : () -> f32
  // CHECK-NEXT: return %{{.*}} : f32
}

// CHECK-LABEL: @fold_sub_i32_f32
func @fold_sub_i32_f32() -> f32 {
  %cst_0 = "eltwise.sconst"() {value = 1 : i32} : () -> si32
  %cst_1 = "eltwise.sconst"() {value = 3.0 : f32} : () -> f32
  %0 = "eltwise.sub"(%cst_0, %cst_1) : (si32, f32) -> f32
  return %0 : f32
  // CHECK-NEXT: "eltwise.sconst"() {value = -2.000000e+00 : f32} : () -> f32
  // CHECK-NEXT: return %{{.*}} : f32
}

// CHECK-LABEL: @fold_sub_i32_i32
func @fold_sub_i32_i32() -> si32 {
  %cst_0 = "eltwise.sconst"() {value = 1 : i32} : () -> si32
  %cst_1 = "eltwise.sconst"() {value = 3 : i32} : () -> si32
  %0 = "eltwise.sub"(%cst_0, %cst_1) : (si32, si32) -> si32
  return %0 : si32
  // CHECK-NEXT: "eltwise.sconst"() {value = -2 : i32} : () -> si32
  // CHECK-NEXT: return %{{.*}} : si32
}

// CHECK-LABEL: @fold_sub_i32_0
func @fold_sub_i32_0(%arg0: si32) -> si32 {
  %cst = "eltwise.sconst"() {value = 0 : i32} : () -> si32
  %0 = "eltwise.sub"(%arg0, %cst) : (si32, si32) -> si32
  return %0 : si32
  // CHECK-NEXT: return %{{.*}}
}

// CHECK-LABEL: @fold_sub_f32_0
func @fold_sub_f32_0(%arg0: f32) -> f32 {
  %cst = "eltwise.sconst"() {value = 0.0 : f32} : () -> f32
  %0 = "eltwise.sub"(%arg0, %cst) : (f32, f32) -> f32
  return %0 : f32
  // CHECK-NEXT: return %{{.*}}
}

// CHECK-LABEL: @fold_div_f32_f32
func @fold_div_f32_f32() -> f32 {
  %cst_0 = "eltwise.sconst"() {value = 3.0 : f32} : () -> f32
  %cst_1 = "eltwise.sconst"() {value = 2.0 : f32} : () -> f32
  %0 = "eltwise.div"(%cst_0, %cst_1) : (f32, f32) -> f32
  return %0 : f32
  // CHECK-NEXT: "eltwise.sconst"() {value = 1.500000e+00 : f32} : () -> f32
  // CHECK-NEXT: return %{{.*}}
}

// CHECK-LABEL: @fold_div_f32_i32
func @fold_div_f32_i32() -> f32 {
  %cst_0 = "eltwise.sconst"() {value = 3.0 : f32} : () -> f32
  %cst_1 = "eltwise.sconst"() {value = 2 : i32} : () -> si32
  %0 = "eltwise.div"(%cst_0, %cst_1) : (f32, si32) -> f32
  return %0 : f32
  // CHECK-NEXT: "eltwise.sconst"() {value = 1.500000e+00 : f32} : () -> f32
  // CHECK-NEXT: return %{{.*}} : f32
}

// CHECK-LABEL: @fold_div_i32_f32
func @fold_div_i32_f32() -> f32 {
  %cst_0 = "eltwise.sconst"() {value = 3 : i32} : () -> si32
  %cst_1 = "eltwise.sconst"() {value = 2.0 : f32} : () -> f32
  %0 = "eltwise.div"(%cst_0, %cst_1) : (si32, f32) -> f32
  return %0 : f32
  // CHECK-NEXT: "eltwise.sconst"() {value = 1.500000e+00 : f32} : () -> f32
  // CHECK-NEXT: return %{{.*}} : f32
}

// CHECK-LABEL: @fold_div_i32_i32
func @fold_div_i32_i32() -> si32 {
  %cst_0 = "eltwise.sconst"() {value = 3 : i32} : () -> si32
  %cst_1 = "eltwise.sconst"() {value = 2 : i32} : () -> si32
  %0 = "eltwise.div"(%cst_0, %cst_1) : (si32, si32) -> si32
  return %0 : si32
  // CHECK-NEXT: "eltwise.sconst"() {value = 1 : i32} : () -> si32
  // CHECK-NEXT: return %{{.*}} : si32
}

// CHECK-LABEL: @fold_div_i32_1
func @fold_div_i32_1(%arg0: si32) -> si32 {
  %cst = "eltwise.sconst"() {value = 1 : i32} : () -> si32
  %0 = "eltwise.div"(%arg0, %cst) : (si32, si32) -> si32
  return %0 : si32
  // CHECK-NEXT: return %{{.*}}
}

// CHECK-LABEL: @fold_div_f32_1
func @fold_div_f32_1(%arg0: f32) -> f32 {
  %cst = "eltwise.sconst"() {value = 1.0 : f32} : () -> f32
  %0 = "eltwise.div"(%arg0, %cst) : (f32, f32) -> f32
  return %0 : f32
  // CHECK-NEXT: return %{{.*}}
}

// CHECK-LABEL: @fold_div_0_i32
func @fold_div_0_i32(%arg0: si32) -> si32 {
  %cst = "eltwise.sconst"() {value = 0 : i32} : () -> si32
  %0 = "eltwise.div"(%cst, %arg0) : (si32, si32) -> si32
  return %0 : si32
  // CHECK-NEXT: "eltwise.sconst"() {value = 0 : i32} : () -> si32
  // CHECK-NEXT: return %{{.*}} : si32
}

// CHECK-LABEL: @fold_div_0_f32
func @fold_div_0_f32(%arg0: f32) -> f32 {
  %cst = "eltwise.sconst"() {value = 0.0 : f32} : () -> f32
  %0 = "eltwise.div"(%cst, %arg0) : (f32, f32) -> f32
  return %0 : f32
  // CHECK-NEXT: "eltwise.sconst"() {value = 0.000000e+00 : f32} : () -> f32
  // CHECK-NEXT: return %{{.*}} : f32
}

// Expected behavior of div by 0 is to not fold
// CHECK-LABEL: @fold_div_f32_0
func @fold_div_f32_0(%arg0: f32) -> f32 {
  %cst = "eltwise.sconst"() {value = 0.0 : f32} : () -> f32
  %0 = "eltwise.div"(%arg0, %cst) : (f32, f32) -> f32
  return %0 : f32
  // CHECK-NEXT: "eltwise.sconst"() {value = 0.000000e+00 : f32} : () -> f32
  // CHECK-NEXT: "eltwise.div"(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
  // CHECK-NEXT: return %{{.*}} : f32
}

// Expected behavior in this case of cast is to fold
// CHECK-LABEL: @fold_cast_si32
func @fold_cast_si32(%arg0: tensor<3xsi32>) -> tensor<3xsi32> {
  %0 = "eltwise.cast"(%arg0) : (tensor<3xsi32>) -> tensor<3xsi32>
  return %0 : tensor<3xsi32>
  // CHECK-NEXT: return %{{.*}} : tensor<3xsi32>
}

// Expected behavior in this case of cast is to fold
// CHECK-LABEL: @fold_cast_f32
func @fold_cast_f32(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %0 = "eltwise.cast"(%arg0) : (tensor<3xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
  // CHECK-NEXT: return %{{.*}} : tensor<3xf32>
}

// Expected behavior in this case of cast is NOT to fold
// CHECK-LABEL: @cast_si32_to_f32
func @cast_si32_to_f32(%arg0: tensor<3xsi32>) -> tensor<3xf32> {
  %0 = "eltwise.cast"(%arg0) : (tensor<3xsi32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
  // CHECK-NEXT: %[[cast:.*]] = "eltwise.cast"(%{{.*}}) : (tensor<3xsi32>) -> tensor<3xf32>
  // CHECK-NEXT: return %[[cast]] : tensor<3xf32>
}
