// RUN: pmlc-opt %s -split-input-file -verify-diagnostics

func @atomic_rmw_idxs_rank_mismatch(%I: memref<16x10xf32>, %i : index) {
  %cst = constant 1.0 : f32
  // expected-error@+1 {{expects the number of subscripts to be equal to memref rank}}
  stdx.atomic_rmw %val = %I[%i] : memref<16x10xf32> {
    %0 = addf %val, %cst : f32
    stdx.atomic_rmw.yield %0 : f32
  }
  return
}

// -----

func @atomic_rmw_empty_body(%I: memref<16x10xf32>, %i : index, %j : index) {
  %cst = constant 1.0 : f32
  // expected-error@+1 {{expects a non-empty body}}
  stdx.atomic_rmw %val = %I[%i, %j] : memref<16x10xf32> {}
  return
}

// -----

func @atomic_rmw_region_arg_missing(%I: memref<16x10xf32>, %i : index, %j : index) {
  %cst = constant 1.0 : f32
  // expected-error@+1 {{expects a body with one argument of type 'f32'}}
  "stdx.atomic_rmw"(%I, %i, %j) ({
    stdx.atomic_rmw.yield %cst : f32
  }) : (memref<16x10xf32>, index, index) -> ()
  return
}

// -----

func @atomic_rmw_region_arg_type_mismatch(%I: memref<16x10xf32>, %i : index, %j : index) {
  %cst = constant 1.0 : f32
  // expected-error@+1 {{expects a body with one argument of type 'f32'}}
  "stdx.atomic_rmw"(%I, %i, %j) ({
  ^bb0(%val: i32):
    stdx.atomic_rmw.yield %cst : f32
  }) : (memref<16x10xf32>, index, index) -> ()
  return
}

// -----

func @atomic_rmw_missing_yield(%I: memref<16x10xf32>, %i : index, %j : index) {
  %cst = constant 1.0 : f32
  // expected-error@+1 {{expects the body to be terminated with a 'stdx.atomic_rmw.yield' op}}
  stdx.atomic_rmw %val = %I[%i, %j] : memref<16x10xf32> {
    %0 = addf %val, %cst : f32
    "loop.terminator"() : () -> ()
  }
  return
}

// -----

func @atomic_rmw_yield_type_mismatch(%I: memref<16x10xf32>, %i : index, %j : index) {
  %c0 = constant 1 : i32
  %cst = constant 1.0 : f32
  stdx.atomic_rmw %val = %I[%i, %j] : memref<16x10xf32> {
    // expected-error@+1 {{needs to have type 'f32'}}
    stdx.atomic_rmw.yield %c0 : i32
  }
  return
}
