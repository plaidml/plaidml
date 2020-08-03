// Copyright 2020, Intel Corporation

#ifndef __PMLC_COMP_INTERFACES__
#define __PMLC_COMP_INTERFACES__

include "mlir/IR/OpBase.td"

def COMP_ExecEnvOpInterface : OpInterface<"ExecEnvOpInterface"> {
  let description = [{

  }];

  let methods = [
    InterfaceMethod<"", "::mlir::Value", "getExecEnv", (ins), [{}], [{
      ConcreteOp op = ::mlir::cast<ConcreteOp>(this->getOperation());
      return op.execEnv();
    }]>
  ];
}

def COMP_ScheduleOpInterface : OpInterface<"ScheduleOpInterface"> {
  let description = [{
    Interface for operations that are scheduled on execution environment.
    Provides access for dependant event and produced event.
  }];

  let methods = [
    InterfaceMethod<"", "::mlir::Value", "getDependency", (ins), [{}], [{
      ConcreteOp op = ::mlir::cast<ConcreteOp>(this->getOperation());
      return op.depEvent();
    }]>,
    InterfaceMethod<"", "::mlir::MutableOperandRange", "getDependencyMutable", (ins), [{}], [{
      ConcreteOp op = ::mlir::cast<ConcreteOp>(this->getOperation());
      return op.depEventMutable();
    }]>,
    InterfaceMethod<"", "bool", "hasDependency", (ins), [{}], [{
      ConcreteOp op = ::mlir::cast<ConcreteOp>(this->getOperation());
      return static_cast<bool>(op.depEvent());
    }]>,
    InterfaceMethod<"", "::mlir::Value", "getResultingEvent", (ins), [{}], [{
      ConcreteOp op = ::mlir::cast<ConcreteOp>(this->getOperation());
      return op.getResult();
    }]>,
  ];
}

def COMP_MemoryTransferOpInterface : OpInterface<"MemoryTransferOpInterface"> {
  let description = [{
  }];

  let methods = [
    InterfaceMethod<"", "::mlir::Value", "getSource">,
    InterfaceMethod<"", "::mlir::Value", "getDestination">,
    InterfaceMethod<"", "::mlir::Value", "getSourceExecEnv">,
    InterfaceMethod<"", "::mlir::Value", "getDestinationExecEnv">,
    InterfaceMethod<"", "bool", "sourceHasExecEnv", (ins), [{}], [{
      ConcreteOp op = ::mlir::cast<ConcreteOp>(this->getOperation());
      return static_cast<bool>(op.getSourceExecEnv());
    }]>,
    InterfaceMethod<"", "bool", "destinationHasExecEnv", (ins), [{}], [{
      ConcreteOp op = ::mlir::cast<ConcreteOp>(this->getOperation());
      return static_cast<bool>(op.getDestinationExecEnv());
    }]>,
  ];
}

#endif  // __PMLC_COMP_INTERFACES__
