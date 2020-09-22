// Copyright 2020, Intel Corporation

#pragma once

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"

#include "pmlc/util/buffer.h"

namespace pmlc::rt {

template <typename T>
struct Registry {
  static Registry *instance() {
    static Registry registry;
    return &registry;
  }

  void registerItem(llvm::StringRef key, T value) {
    if (map.count(key)) {
      throw std::runtime_error(
          llvm::formatv("Item is already registered: {0}", key));
    }
    map[key] = value;
  }

  T resolve(llvm::StringRef key) {
    auto it = map.find(key);
    if (it == map.end()) {
      return nullptr;
    }
    return it->second;
  }

  llvm::StringMap<T> map;
};

using ConstantRegistry = Registry<pmlc::util::BufferPtr>;

} // namespace pmlc::rt
