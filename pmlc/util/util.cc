// Copyright 2019, Intel Corporation

#include "pmlc/util/util.h"

namespace pmlc {
namespace util {

llvm::StringRef getOpName(const mlir::OperationName& name) {
  return name.getStringRef().drop_front(name.getDialect().size() + 1);
}

}  // namespace util
}  // namespace pmlc
