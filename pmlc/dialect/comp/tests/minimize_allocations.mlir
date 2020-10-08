// RUN: pmlc-opt --comp-minimize-allocations --allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: func @two_wo_host
//  CHECK-SAME:     %[[ENV:[a-zA-Z0-9]*]]: !comp.execenv
//       CHECK:   %[[MEM:.*]] = comp.alloc %[[ENV]]
//  CHECK-NEXT:   "op1"(%[[MEM]])
//  CHECK-NEXT:   "op2"(%[[MEM]])
//  CHECK-NEXT:   comp.dealloc %[[ENV]] %[[MEM]]
func @two_wo_host(%env: !comp.execenv<ocl:0,(11)>) {
  %mem1 = comp.alloc %env : (!comp.execenv<ocl:0,(11)>) -> memref<2x3xf32, 11>
  "op1"(%mem1) : (memref<2x3xf32, 11>) -> ()
  comp.dealloc %env %mem1 : (!comp.execenv<ocl:0,(11)>, memref<2x3xf32, 11>) -> ()

  %mem2 = comp.alloc %env : (!comp.execenv<ocl:0,(11)>) -> memref<2x3xf32, 11>
  "op2"(%mem2) : (memref<2x3xf32, 11>) -> ()
  comp.dealloc %env %mem2 : (!comp.execenv<ocl:0,(11)>, memref<2x3xf32, 11>) -> ()
  return
}

// CHECK-LABEL: func @two_with_host
//  CHECK-SAME:     %[[ENV:[a-zA-Z0-9]*]]: !comp.execenv
//  CHECK-SAME:     %[[ARG:[a-zA-Z0-9]*]]: memref
//       CHECK:   %[[MEM:.*]] = comp.alloc %[[ENV]] %[[ARG]]
//  CHECK-NEXT:   "op1"(%[[MEM]])
//  CHECK-NEXT:   %[[WEV:.*]] = comp.schedule_write %[[ARG]] to %[[MEM]]
//  CHECK-NEXT:   comp.wait %[[WEV]]
//  CHECK-NEXT:   "op2"(%[[MEM]])
//  CHECK-NEXT:   comp.dealloc %[[ENV]] %[[MEM]]
func @two_with_host(%env: !comp.execenv<ocl:0,(11)>, %arg: memref<2x3xf32>) {
  %mem1 = comp.alloc %env %arg : (!comp.execenv<ocl:0,(11)>, memref<2x3xf32>) -> memref<2x3xf32, 11>
  "op1"(%mem1) : (memref<2x3xf32, 11>) -> ()
  comp.dealloc %env %mem1 : (!comp.execenv<ocl:0,(11)>, memref<2x3xf32, 11>) -> ()

  %mem2 = comp.alloc %env %arg : (!comp.execenv<ocl:0,(11)>, memref<2x3xf32>) -> memref<2x3xf32, 11>
  "op2"(%mem2) : (memref<2x3xf32, 11>) -> ()
  comp.dealloc %env %mem2 : (!comp.execenv<ocl:0,(11)>, memref<2x3xf32, 11>) -> ()
  return
}

// CHECK-LABEL: func @three_overlaping
//  CHECK-SAME:     %[[ENV:[a-zA-Z0-9]*]]: !comp.execenv
//       CHECK:   %[[MEM1:.*]] = comp.alloc %[[ENV]]
//       CHECK:   %[[MEM2:.*]] = comp.alloc %[[ENV]]
//       CHECK:   "op1"(%[[MEM1]], %[[MEM2]])
//       CHECK:   "op2"(%[[MEM2]], %[[MEM1]])
//   CHECK-DAG:   comp.dealloc %[[ENV]] %[[MEM1]]
//   CHECK-DAG:   comp.dealloc %[[ENV]] %[[MEM2]]
func @three_overlaping(%env: !comp.execenv<ocl:0,(11)>) {
  %mem1 = comp.alloc %env : (!comp.execenv<ocl:0,(11)>) -> memref<2x3xf32, 11>
  %mem2 = comp.alloc %env : (!comp.execenv<ocl:0,(11)>) -> memref<2x3xf32, 11>
  "op1"(%mem1, %mem2) :  (memref<2x3xf32, 11>, memref<2x3xf32, 11>) -> ()
  comp.dealloc %env %mem1 : (!comp.execenv<ocl:0,(11)>, memref<2x3xf32, 11>) -> ()
  %mem3 = comp.alloc %env : (!comp.execenv<ocl:0,(11)>) -> memref<2x3xf32, 11>
  "op2"(%mem2, %mem3) :  (memref<2x3xf32, 11>, memref<2x3xf32, 11>) -> ()
  comp.dealloc %env %mem2 : (!comp.execenv<ocl:0,(11)>, memref<2x3xf32, 11>) -> ()
  comp.dealloc %env %mem3 : (!comp.execenv<ocl:0,(11)>, memref<2x3xf32, 11>) -> ()
  return
}
