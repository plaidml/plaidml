// Copyright 2020, Intel Corporation

#pragma once

#include "llvm/ADT/StringRef.h"

namespace pmlc::rt {

void registerSymbol(llvm::StringRef symbol, void *ptr);

void *resolveSymbol(llvm::StringRef symbol);

} // namespace pmlc::rt
