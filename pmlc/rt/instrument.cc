// Copyright 2021 Intel Corporation

#include "pmlc/rt/instrument.h"

#include <chrono>

#include "libxsmm.h" // NOLINT [build/include_subdir]
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include "pmlc/rt/symbol_registry.h"
#include "pmlc/util/env.h"

namespace pmlc::rt {

static uint64_t initTime;
static uint64_t globalTime;

void initInstrument() { //
  initTime = globalTime =
      libxsmm_timer_tick(); // globalTime = steady_clock::now();
  if (util::getEnvVar("PLAIDML_PROFILE") == "1") {
    llvm::outs() << "\"id\",\"loc\",\"tag\",\"elapsed\",\"accumulated\"\n";
    llvm::outs().flush();
  }
}
void instrumentPoint(int64_t id, int64_t tag, const char *loc) {
  auto now = libxsmm_timer_tick();
  auto lastDuration = libxsmm_timer_duration(globalTime, now);
  auto initDuration = libxsmm_timer_duration(initTime, now);
  llvm::outs() << llvm::format("%03d,%s,%d,%7.6f,%7.6f\n", id, loc, tag,
                               lastDuration, initDuration);
  llvm::outs().flush();
  globalTime = now;
}

double finishInstrument() {
  instrumentPoint(9999, 9999, "endTime");
  return libxsmm_timer_duration(initTime, libxsmm_timer_tick());
}

} // namespace pmlc::rt

extern "C" void _mlir_ciface_plaidml_rt_instrument(int64_t id, int64_t tag,
                                                   const char *loc) {
  pmlc::rt::instrumentPoint(id, tag, loc);
}

namespace pmlc::rt {

void registerInstrument() { //
  REGISTER_SYMBOL(_mlir_ciface_plaidml_rt_instrument);
}

} // namespace pmlc::rt
