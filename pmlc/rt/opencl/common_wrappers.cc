// Copyright 2020 Intel Corporation
#include "mlir/ExecutionEngine/RunnerUtils.h"

#include "pmlc/rt/symbol_registry.h"
#include "pmlc/util/logging.h"

namespace pmlc::rt::opencl {
namespace {
template <typename T>
void *castMemrefToPtr(::UnrankedMemRefType<T> *unrankedMemRef) {
  DynamicMemRefType<T> memRef(*unrankedMemRef);
  return memRef.data;
}
} // namespace
// clang-format off
#define ALL_CAST_MEMREF_TO_PTR(macro)                                          \
    macro(castMemrefToPtrI8, int8_t)                                           \
    macro(castMemrefToPtrI16, int16_t)                                         \
    macro(castMemrefToPtrI32, int32_t)                                         \
    macro(castMemrefToPtrI64, int64_t)                                         \
    macro(castMemrefToPtrBF16, int16_t)                                        \
    macro(castMemrefToPtrF16, int16_t)                                         \
    macro(castMemrefToPtrF32, float)                                           \
    macro(castMemrefToPtrF64, double)
// clang-format on

extern "C" {
#define DECLARE_CAST_MEMREF_TO_PTR(name, type)                                 \
  char *name(::UnrankedMemRefType<type> *unrankedMemRef) {                     \
    return static_cast<char *>(castMemrefToPtr<type>(unrankedMemRef));         \
  }

ALL_CAST_MEMREF_TO_PTR(DECLARE_CAST_MEMREF_TO_PTR)

#undef DECLARE_CAST_MEMREF_TO_PTR
} // extern "C"

namespace {
struct Registration {
  Registration() {
    using pmlc::rt::registerSymbol;
#define REGISTER_CAST_MEMREF_TO_PTR(name, type)                                \
  registerSymbol("_mlir_ciface_" #name, reinterpret_cast<void *>(name));

    ALL_CAST_MEMREF_TO_PTR(REGISTER_CAST_MEMREF_TO_PTR)

#undef REGISTER_CAST_MEMREF_TO_PTR
  }
};
static Registration reg;
} // namespace
} // namespace pmlc::rt::opencl
