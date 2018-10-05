// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <gflags/gflags.h>

#include <mutex>

// Tile debug synchronization support.

// The following flag is specified to request debug synchronization.  Components may use this
// flag to synchronize operations where feasible, reducing interleaving.
DECLARE_bool(tile_enable_dbgsync);

namespace vertexai {
namespace tile {

// The global debug synchronization mutex.
extern std::mutex dbgsync_mu;

// Returns a std::unique_lock.  If debug synchronization has been requested on the command line, the lock will have
// locked a global mutex (and otherwise, the unique_lock will not have an associated mutex).  This may be used to reduce
// operation interleaving while debugging, although callers must be careful to avoid producer/consumer dependencies.
std::unique_lock<std::mutex> LockDbgSync();

}  // namespace tile
}  // namespace vertexai
