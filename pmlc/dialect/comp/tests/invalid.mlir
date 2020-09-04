// RUN: pmlc-opt --allow-unregistered-dialect --verify-diagnostics --split-input-file %s

func @invalid_runtime() {
  // expected-error@+2 {{unrecognized runtime string name}}
  // expected-note@+1 {{available runtime names are}}
  %env = comp.create_execenv : !comp.execenv<invalid:0,(11)>
  return
}

// -----

func @invalid_schedule() {
  %env = comp.create_execenv : !comp.execenv<0:0,(11)>
  // expected-error@+1 {{'comp.schedule_func' op body must have one operation - 'gpu.launch_func'}}
  "comp.schedule_func"(%env) ({
    "comp.schedule_end"() : () -> ()
  }) : (!comp.execenv<0:0,(11)>) -> (!comp.event<0>)
  return
}

// -----

func @invalid_schedule() {
  %env = comp.create_execenv : !comp.execenv<0:0,(11)>
  // expected-error@+1 {{'comp.schedule_func' op body must have one operation - 'gpu.launch_func'}}
  "comp.schedule_func"(%env) ({
    %cst = constant 42 : i32
    "comp.schedule_end"() : () -> ()
  }) : (!comp.execenv<0:0,(11)>) -> (!comp.event<0>)
  return
}

// -----

module attributes { gpu.container_module } {
  func @invalid_schedule() {
    %env = comp.create_execenv : !comp.execenv<0:0,(11)>
    %ev = "op"() : () -> (!comp.event<1>)
    %idx = "op"() : () -> (index)
    // expected-error@+1 {{'comp.schedule_func' op mismatch between execenv runtime and dependant event runtime}}
    "comp.schedule_func"(%env, %ev) ({
      "gpu.launch_func"(%idx, %idx, %idx, %idx, %idx, %idx) {kernel = @kernel::@kernel} : (index, index, index, index, index, index) -> ()
      "comp.schedule_end"() : () -> ()
    }) : (!comp.execenv<0:0,(11)>, !comp.event<1>) -> (!comp.event<0>)
    return
  }

  gpu.module @kernel {
    gpu.func @kernel() kernel {
      gpu.return
    }
  }
}

// -----

func @invalid_alloc() {
  %env = comp.create_execenv : !comp.execenv<0:0,(11)>
  // expected-error@+1 {{'comp.alloc' op failed to verify that memory space is supported by execenv}}
  %mem = comp.alloc %env : (!comp.execenv<0:0,(11)>) -> (memref<2x3xf32, 1>)
  return
}

// -----

func @invalid_alloc() {
  %env = comp.create_execenv : !comp.execenv<0:0,(11)>
  %host = "op"() : () -> (memref<1x1xf32>)
  // expected-error@+1 {{'comp.alloc' op host and device memory shapes must match}}
  %mem = comp.alloc %env %host : (!comp.execenv<0:0,(11)>, memref<1x1xf32>) -> (memref<2x3xf32, 11>)
  return
}

// -----

func @invalid_alloc() {
  %env = comp.create_execenv : !comp.execenv<0:0,(11)>
  %host = "op"() : () -> (memref<2x3xf32>)
  // expected-error@+1 {{'comp.alloc' op host and device memory element types must match}}
  %mem = comp.alloc %env %host : (!comp.execenv<0:0,(11)>, memref<2x3xf32>) -> (memref<2x3xf16, 11>)
  return
}

// -----

func @invalid_dealloc() {
  %env = comp.create_execenv : !comp.execenv<0:0,(11)>
  %mem = "op"() : () -> (memref<2x3xf32, 1>)
  // expected-error@+1 {{'comp.dealloc' op failed to verify that memory space is supported by execenv}}
  comp.dealloc %env %mem : (!comp.execenv<0:0,(11)>, memref<2x3xf32, 1>) -> ()
  return
}

// -----

func @invalid_write() {
  %env = comp.create_execenv : !comp.execenv<0:0,(11)>
  %host = "op"() : () -> (memref<2x3xf32>)
  %device = "op"() : () -> (memref<2x3xf32, 1>)
  // expected-error@+1 {{'comp.schedule_write' op memory space is not supported by execenv}}
  %ev = comp.schedule_write %host to %device on %env
      : (memref<2x3xf32>, memref<2x3xf32, 1>, !comp.execenv<0:0,(11)>) -> (!comp.event<0>)
  return
}

// -----

func @invalid_read() {
  %env = comp.create_execenv : !comp.execenv<0:0,(11)>
  %host = "op"() : () -> (memref<2x3xf32>)
  %device = "op"() : () -> (memref<2x3xf32, 1>)
  // expected-error@+1 {{'comp.schedule_read' op memory space is not supported by execenv}}
  %ev = comp.schedule_read %host from %device on %env
      : (memref<2x3xf32>, memref<2x3xf32, 1>, !comp.execenv<0:0,(11)>) -> (!comp.event<0>)
  return
}

// -----

func @invalid_read() {
  %env = comp.create_execenv : !comp.execenv<0:0,(11)>
  %host = "op"() : () -> (memref<2x3xf32>)
  %device = "op"() : () -> (memref<2x3xf32, 11>)
  // expected-error@+1 {{'comp.schedule_read' op mismatch between execenv runtime and resulting event runtime}}
  %ev = comp.schedule_read %host from %device on %env
      : (memref<2x3xf32>, memref<2x3xf32, 11>, !comp.execenv<0:0,(11)>) -> (!comp.event<1>)
  return
}

// -----

func @invalid_barrier() {
  %env = comp.create_execenv : !comp.execenv<0:0,(11)>
  %ev0 = "op"() : () -> (!comp.event<0>)
  %ev1 = "op"() : () -> (!comp.event<1>)
  // expected-error@+1 {{'comp.schedule_barrier' op mismatch between execenv runtime and dependant event runtime}}
  %bar = comp.schedule_barrier %env wait for %ev0, %ev1
      : (!comp.execenv<0:0,(11)>, !comp.event<0>, !comp.event<1>) -> (!comp.event<0>)
  return
}
