// Copyright 2017-2018 Intel Corporation.

#include "tile/base/dbgsync.h"

DEFINE_bool(tile_enable_dbgsync, false, "Enable debug operation synchronization");

namespace vertexai {
namespace tile {

std::mutex dbgsync_mu;

std::unique_lock<std::mutex> LockDbgSync() {
  if (!FLAGS_tile_enable_dbgsync) {
    return std::unique_lock<std::mutex>();
  }
  return std::unique_lock<std::mutex>(dbgsync_mu);
}

}  // namespace tile
}  // namespace vertexai
