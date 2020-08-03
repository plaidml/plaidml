// Copyright 2020, Intel Corporation

#ifndef __PMLC_COMP_OPS__
#define __PMLC_COMP_OPS__

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
// include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

include "interfaces.td"

// TODO Type constraints
// TODO Verification
// TODO Builders
// TODO Descriptions

// =============================================================================
// Dialect
// =============================================================================
def COMP_Dialect : Dialect {
  let name = "comp";
  let cppNamespace = "pmlc::dialect::comp";
  let extraClassDeclaration = [{
    Type parseType(DialectAsmParser& parser) const override;
    void printType(Type, DialectAsmPrinter& printer) const override;
  }];
}

// =============================================================================
// Types
// =============================================================================
def COMP_IsExecEnvTypePred : CPred<"$_self.isa<::pmlc::dialect::comp::ExecEnvType>()">;
def COMP_AnyExecEnv : Type<COMP_IsExecEnvTypePred>;

def COMP_IsEventTypePred : CPred<"$_self.isa<::pmlc::dialect::comp::EventType>()">;
def COMP_AnyEvent : Type<COMP_IsEventTypePred>;

// =============================================================================
// Operations
// =============================================================================
class COMP_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<COMP_Dialect, mnemonic, traits>;

def COMP_CreateExecEnv : COMP_Op<"create_execenv",
                                 [MemoryEffects<[MemAlloc<DefaultResource>]>]>
                       , Results<(outs COMP_AnyExecEnv:$execEnv)> {
  let summary = "Creates execution environment.";

  let description = [{
    Execution environment provides functionality for scheduling operations and
    synchronizing between them.
  }];

  let assemblyFormat = [{
    attr-dict `:` type($execEnv)
  }];
}

def COMP_DestroyExecEnv : COMP_Op<"destroy_execenv",
                                  [MemoryEffects<[MemFree<DefaultResource>]>]> {
  let summary = "Destroys execution environment.";

  let description = [{
    Any usage of environment or connected with it memory and events is invalid
    after this operation.
    Does not guarantee that submitted tasks finished executing.
  }];

  let arguments = (ins COMP_AnyExecEnv:$execEnv);

  let assemblyFormat = [{
     $execEnv attr-dict `:` type($execEnv)
  }];
}

def COMP_ScheduleEnd : COMP_Op<"schedule_end", [Terminator]> {
  let summary = "End of scheduling block.";
}

def COMP_ScheduleSingleBlock : SingleBlockImplicitTerminator<"ScheduleEnd">;

def COMP_ScheduleFunc : COMP_Op<"schedule_func",
                                [COMP_ScheduleSingleBlock,
                                 DeclareOpInterfaceMethods<COMP_ExecEnvOpInterface>,
                                 DeclareOpInterfaceMethods<COMP_ScheduleOpInterface>]>
                  , Arguments<(ins COMP_AnyExecEnv:$execEnv,
                                   Optional<COMP_AnyEvent>:$depEvent)>
                  , Results<(outs COMP_AnyEvent:$outEvent)> {
  let summary = "Schedules function in region for execution on device.";
  let description = [{
    This is a wrapper around gpu.launch_func that schedules function
    to execute on environment and provides synchronization with
    event dependencies and output event.
    The region is expected to have only this single gpu.launch_func operation
    and implicit terminator - comp.schedule_end.
    Example:
    %ev = "comp.launch_func"(%execEnv, %depEvent) ( {
      "gpu.launch_func"(...)
      "comp.schedule_end"
    })
    comp.wait %ev
  }];

  let regions = (region AnyRegion:$body);
}


def COMP_Submit : COMP_Op<"submit",
                          [DeclareOpInterfaceMethods<COMP_ExecEnvOpInterface>]>
                , Arguments<(ins COMP_AnyExecEnv:$execEnv)> {
  let summary = "Submits all scheduled tasks.";

  let description = [{
    Forces all previously scheduled tasks to start executing.
  }];

  let assemblyFormat = [{
    $execEnv attr-dict `:` type($execEnv)
  }];
}

def COMP_Alloc : COMP_Op<"alloc", [MemoryEffects<[MemAlloc<DefaultResource>]>,
                                   DeclareOpInterfaceMethods<COMP_ExecEnvOpInterface>]>
               , Arguments<(ins COMP_AnyExecEnv:$execEnv,
                                Optional<AnyMemRef>:$hostMem)>
               , Results<(outs AnyMemRef:$out)> {
  let summary = "Allocates memory for use in execution environment.";

  let description = [{
    Optionally accepts host memory, in which case there is guarantee that
    after allocation new device memory will contain the same data.
  }];

  let assemblyFormat = [{
     $execEnv ($hostMem^)? attr-dict `:` functional-type(operands, results)
  }];
}

def COMP_Dealloc : COMP_Op<"dealloc", [MemoryEffects<[MemFree<DefaultResource>]>]>
                 , Arguments<(ins AnyMemRef:$deviceMem)> {
  let summary = "Deallocates memory previously allocated for use in executon environment.";

  let assemblyFormat = [{
      $deviceMem attr-dict `:` functional-type(operands, results)
  }];
}

def COMP_ScheduleWrite : COMP_Op<"schedule_write",
                                 [DeclareOpInterfaceMethods<COMP_ExecEnvOpInterface>,
                                  DeclareOpInterfaceMethods<COMP_ScheduleOpInterface>,
                                  DeclareOpInterfaceMethods<COMP_MemoryTransferOpInterface, ["getSource", "getDestination", "getSourceExecEnv", "getDestinationExecEnv"]>]>
                       , Arguments<(ins AnyMemRef:$hostMem,
                                        AnyMemRef:$deviceMem,
                                        COMP_AnyExecEnv:$execEnv,
                                        Optional<COMP_AnyEvent>:$depEvent)>
                       , Results<(outs COMP_AnyEvent:$outEvent)> {
  let summary = "Schedules data copy from host for use in execution environment.";

  let assemblyFormat = [{
    $hostMem `to` $deviceMem `on` $execEnv (`wait` `for` $depEvent^)? attr-dict `:` functional-type(operands, results)
  }];
}

def COMP_ScheduleRead : COMP_Op<"schedule_read",
                                [DeclareOpInterfaceMethods<COMP_ExecEnvOpInterface>,
                                 DeclareOpInterfaceMethods<COMP_ScheduleOpInterface>,
                                 DeclareOpInterfaceMethods<COMP_MemoryTransferOpInterface, ["getSource", "getDestination", "getSourceExecEnv", "getDestinationExecEnv"]>]>
                      , Arguments<(ins AnyMemRef:$hostMem,
                                       AnyMemRef:$deviceMem,
                                       COMP_AnyExecEnv:$execEnv,
                                       Optional<COMP_AnyEvent>:$depEvent)>
                      , Results<(outs COMP_AnyEvent:$outEvent)> {
  let summary = "Schedules data copy from execution environment for use on host.";

  let assemblyFormat = [{
    $hostMem `from` $deviceMem `on` $execEnv (`wait` `for` $depEvent^)? attr-dict `:` functional-type(operands, results)
  }];
}

def COMP_ScheduleCopy : COMP_Op<"schedule_copy",
                                [DeclareOpInterfaceMethods<COMP_ExecEnvOpInterface>,
                                 DeclareOpInterfaceMethods<COMP_ScheduleOpInterface>,
                                 DeclareOpInterfaceMethods<COMP_MemoryTransferOpInterface,
                                                           ["getSource", "getDestination", "getSourceExecEnv", "getDestinationExecEnv"]>]>
                      , Arguments<(ins AnyMemRef:$srcMem,
                                       AnyMemRef:$dstMem,
                                       COMP_AnyExecEnv:$execEnv,
                                       Optional<COMP_AnyEvent>:$depEvent)>
                      , Results<(outs COMP_AnyEvent:$outEvent)> {
  let summary = "Schedules data copy between two memrefs on same execeution device.";

  let assemblyFormat = [{
    $srcMem `to` $dstMem `on` $execEnv (`wait` `for` $depEvent^)? attr-dict `:` functional-type(operands, results)
  }];
}

def COMP_Wait : COMP_Op<"wait">
               , Arguments<(ins COMP_AnyEvent:$event)> {
  let summary = "Waits for event being finished.";

  let description = [{
    After this operation it is guaranteed that operations marked by event are finished,
    and it is save to use their results from host.
  }];

  let assemblyFormat = [{
    $event attr-dict `:` type($event)
  }];
}

def COMP_ScheduleBarrier : COMP_Op<"schedule_barrier",
                                   [DeclareOpInterfaceMethods<COMP_ExecEnvOpInterface>,
                                    DeclareOpInterfaceMethods<COMP_ScheduleOpInterface>]>
                          , Arguments<(ins COMP_AnyExecEnv:$execEnv,
                                           Optional<COMP_AnyEvent>:$depEvent)>
                          , Results<(outs COMP_AnyEvent:$outEvent)> {
  let summary = "Schedules barrier between previous and next sheduling operations.";

  let description = [{
    Barrier serves as one of synchronization primitives.
    After scheduling it provides guarantee that all operations scheduled after
    barrier will start executing only after operations before barrier are finished.
    Optionally if additional event dependency is specified the barrier loosens
    the guarantee that only this event needs to finish before processing later tasks.
  }];

  let assemblyFormat = [{
    $execEnv (`wait` `for` $depEvent^)? attr-dict `:` functional-type(operands, results)
  }];
}

def COMP_ScheduleMarker : COMP_Op<"schedule_marker",
                                  [DeclareOpInterfaceMethods<COMP_ExecEnvOpInterface>,
                                   DeclareOpInterfaceMethods<COMP_ScheduleOpInterface>]>
                        , Arguments<(ins COMP_AnyExecEnv:$execEnv,
                                         Optional<COMP_AnyEvent>:$depEvent)>
                        , Results<(outs COMP_AnyEvent:$outEvent)> {
  let summary = "Schedules marker between previous and next sheduling operations.";

  let description = [{
    Marker works similarly to barrier, but does not create a synchronization point.
  }];

  let assemblyFormat = [{
    $execEnv (`wait` `for` $depEvent^)? attr-dict `:` functional-type(operands, results)
  }];
}

def COMP_GroupEvents : COMP_Op<"group_events", [NoSideEffect]>
               , Arguments<(ins Variadic<COMP_AnyEvent>:$events)>
               , Results<(outs COMP_AnyEvent:$grouped)> {
  let summary = "Groups multiple events into one.";

  let description = [{
    This is syntactic only operation that groups several events into one for
    easier operation.
  }];

  let assemblyFormat = [{
    $events attr-dict `:` functional-type(operands, results)
  }];

  let hasFolder = 1;
}

#endif // __PMLC_COMP_OPS__
