// Copyright 2020 Intel Corporation
#ifndef __PMLC_COMP_INTERFACES__
#define __PMLC_COMP_INTERFACES__

include "mlir/IR/OpBase.td"

class COMP_OpInterface<string name> : OpInterface<name>;

def COMP_ExecEnvOpInterface : COMP_OpInterface<"ExecEnvOpInterface"> {
  let description = [{
    Ties operation to specific execution environment.
  }];

  let methods = [
    InterfaceMethod<"", "::mlir::Value", "getExecEnv", (ins), [{}], [{
      ConcreteOp op = ::mlir::cast<ConcreteOp>(this->getOperation());
      return op.execEnv();
    }]>
  ];
}

def COMP_ScheduleOpInterface : COMP_OpInterface<"ScheduleOpInterface"> {
  let description = [{
    Interface for operations that are scheduled on execution environment.
    Provides access for dependant events and produced event.
    Operations implementing this interface are expected to also implement
    ExecEnvOp interface.
  }];

  let methods = [
    InterfaceMethod<"", "::mlir::OperandRange", "getDependencies", (ins), [{}], [{
      ConcreteOp op = ::mlir::cast<ConcreteOp>(this->getOperation());
      return op.depEvents();
    }]>,
    InterfaceMethod<"", "::mlir::MutableOperandRange", "getDependencyMutable", (ins), [{}], [{
      ConcreteOp op = ::mlir::cast<ConcreteOp>(this->getOperation());
      return op.depEventsMutable();
    }]>,
    InterfaceMethod<"", "bool", "hasDependencies", (ins), [{}], [{
      ConcreteOp op = ::mlir::cast<ConcreteOp>(this->getOperation());
      return !op.depEvents().empty();
    }]>,
    InterfaceMethod<"", "::mlir::Value", "getResultingEvent", (ins), [{}], [{
      ConcreteOp op = ::mlir::cast<ConcreteOp>(this->getOperation());
      return op.getResult();
    }]>,
  ];

  let verify = [{
    static_assert(ConcreteOp::template hasTrait<ExecEnvOpInterface::Trait>(),
                  "expected schedule operation to implement ExecEnvOpInterface");
    return verifyScheduleOp($_op);
  }];
}

def COMP_MemoryTransferOpInterface : COMP_OpInterface<"MemoryTransferOpInterface"> {
  let description = [{
    Interface for operations performing memory transfer between execution environments
    or between host and execution environment.
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

  let verify = [{ return verifyMemoryTransferOp($_op); }];
}

#endif  // __PMLC_COMP_INTERFACES__
