#pragma once

#include <tuple>

#include "mlir/IR/Types.h"
#include "llvm/ADT/Hashing.h"

namespace pmlc::dialect::comp {

namespace CompTypes {
enum Kinds {
  ExecEnv = mlir::Type::Kind::FIRST_PRIVATE_EXPERIMENTAL_9_TYPE,
  Event
};
} // namespace CompTypes

enum ExecEnvRuntime : unsigned { Vulkan, OpenCL, Custom_0 };

using ExecEnvTag = unsigned;

struct ExecEnvStorage : public mlir::TypeStorage {
  ExecEnvStorage(ExecEnvRuntime runtime, ExecEnvTag tag,
                 mlir::ArrayRef<unsigned> memorySpaces)
      : runtime(runtime), tag(tag),
        memorySpaces(memorySpaces.begin(), memorySpaces.end()) {}

  using KeyTy =
      std::tuple<ExecEnvRuntime, ExecEnvTag, mlir::SmallVector<unsigned, 1>>;

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(runtime, tag, memorySpaces);
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(std::get<0>(key), std::get<1>(key),
                              mlir::ArrayRef<unsigned>(std::get<2>(key)));
  }

  static KeyTy getKey(ExecEnvRuntime runtime, ExecEnvTag tag,
                      mlir::ArrayRef<unsigned> memorySpaces) {
    return KeyTy(runtime, tag,
                 mlir::SmallVector<unsigned, 1>(memorySpaces.begin(),
                                                memorySpaces.end()));
  }

  static ExecEnvStorage *construct(mlir::TypeStorageAllocator &allocator,
                                   const KeyTy &key) {
    return new (allocator.allocate<ExecEnvStorage>())
        ExecEnvStorage(std::get<0>(key), std::get<1>(key), std::get<2>(key));
  }

  ExecEnvRuntime runtime;
  ExecEnvTag tag;
  mlir::SmallVector<unsigned, 1> memorySpaces;
};

class ExecEnvType
    : public mlir::Type::TypeBase<ExecEnvType, mlir::Type, ExecEnvStorage> {
public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == CompTypes::ExecEnv; }
  static ExecEnvType get(mlir::MLIRContext *context, ExecEnvRuntime runtime,
                         ExecEnvTag tag,
                         mlir::ArrayRef<unsigned> memorySpaces) {
    return Base::get(context, CompTypes::ExecEnv, runtime, tag, memorySpaces);
  }

  static ExecEnvType getChecked(ExecEnvRuntime runtime, ExecEnvTag tag,
                                mlir::ArrayRef<unsigned> memorySpaces,
                                mlir::Location location) {
    return Base::getChecked(location, CompTypes::ExecEnv, runtime, tag,
                            memorySpaces);
  }

  ExecEnvRuntime getRuntime() const { return getImpl()->runtime; }

  ExecEnvTag getTag() const { return getImpl()->tag; }

  mlir::ArrayRef<unsigned> getMemorySpaces() const {
    return getImpl()->memorySpaces;
  }

  unsigned getDefaultMemorySpace() const { return getMemorySpaces()[0]; }
};

struct EventTypeStorage : public mlir::TypeStorage {
  explicit EventTypeStorage(ExecEnvRuntime runtime) : runtime(runtime) {}

  using KeyTy = ExecEnvRuntime;

  bool operator==(const KeyTy &key) const { return key == KeyTy(runtime); }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  static KeyTy getKey(ExecEnvRuntime runtime) { return KeyTy(runtime); }

  static EventTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                     const KeyTy &key) {
    return new (allocator.allocate<EventTypeStorage>()) EventTypeStorage(key);
  }

  ExecEnvRuntime runtime;
};

class EventType
    : public mlir::Type::TypeBase<EventType, mlir::Type, EventTypeStorage> {
public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == CompTypes::Event; }
  static EventType get(mlir::MLIRContext *context, ExecEnvRuntime runtime) {
    return Base::get(context, CompTypes::Event, runtime);
  }

  static EventType getChecked(ExecEnvRuntime runtime, mlir::Location location) {
    return Base::getChecked(location, CompTypes::Event, runtime);
  }

  static EventType get(mlir::MLIRContext *context, ExecEnvType envType) {
    return EventType::get(context, envType.getRuntime());
  }

  static EventType getChecked(ExecEnvType envType, mlir::Location location) {
    return EventType::getChecked(envType.getRuntime(), location);
  }

  ExecEnvRuntime getRuntime() { return getImpl()->runtime; }
};

} // namespace pmlc::dialect::comp
