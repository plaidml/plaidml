// Copyright 2020 Intel Corporation

#include "pmlc/rt/device_id.h"

#include "llvm/Support/FormatVariadic.h"

#include "pmlc/rt/internal.h"
#include "pmlc/rt/runtime_registry.h"
#include "pmlc/util/logging.h"

namespace pmlc::rt {

const std::regex &getDeviceIDRegex() {
  static const std::regex re(
      "([[:alpha:]][_[:alnum:]]*(-[[:alpha:]][_[:alnum:]]*)*)[.]([[:digit:]]+)",
      std::regex::extended);
  return re;
}

std::shared_ptr<Device> getDevice(llvm::StringRef deviceID) {
  std::cmatch match;
  if (!std::regex_match(deviceID.data(), match, getDeviceIDRegex())) {
    throw std::runtime_error{
        llvm::formatv("Invalid device name: {0}", deviceID)};
  }

  const auto &runtimeMap = getRuntimeMap();
  auto runtimeIt = runtimeMap.find(match[1]);
  if (runtimeIt == runtimeMap.end()) {
    throw std::runtime_error{
        llvm::formatv("Device not available (missing runtime): {0}", deviceID)};
  }

  return runtimeIt->second()->device(std::stoul(match[3].str()));
}

std::vector<std::string> getDeviceIDs() {
  std::vector<std::string> result;
  for (auto &[id, runtimeFunc] : getRuntimeMap()) {
    Runtime *runtime;
    try {
      runtime = runtimeFunc();
    } catch (const std::exception &e) {
      IVLOG(1, "Runtime " << id << " initialization failed: " << e.what());
      continue;
    }
    for (std::size_t idx = 0; idx < runtime->deviceCount(); ++idx) {
      result.emplace_back(llvm::formatv("{0}.{1}", id, idx));
    }
  }
  return result;
}

} // namespace pmlc::rt
