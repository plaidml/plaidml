// Copyright 2020 Intel Corporation

#ifdef _WIN32
#include <intrin.h>
#endif // _WIN32

#include <string>
#include <thread>

#include "pmlc/target/x86/utils.h"

namespace pmlc::target::x86 {

void cpuId(unsigned i, unsigned regs[4]) {
#ifdef _WIN32
  __cpuid(reinterpret_cast<int *>(regs), static_cast<int>(i));
#else
  asm volatile("cpuid"
               : "=a"(regs[0]), "=b"(regs[1]), "=c"(regs[2]), "=d"(regs[3])
               : "a"(i), "c"(0));
  // ECX is set to zero for CPUID function 4
#endif
}

/// Gets the number of physical cores on this machine.
/// Returns the number of cores or 0 if the core count could not be determined.
unsigned getPhysicalCoreNumber() {
  unsigned regs[4];

  // Get manufacturer
  char manufacturer[12];
  cpuId(0, regs);
  reinterpret_cast<unsigned *>(manufacturer)[0] = regs[1]; // EBX
  reinterpret_cast<unsigned *>(manufacturer)[1] = regs[3]; // EDX
  reinterpret_cast<unsigned *>(manufacturer)[2] = regs[2]; // ECX
  std::string cpuManufacturer = std::string(manufacturer, 12);

  unsigned cores = 0;
  if (cpuManufacturer == "GenuineIntel") {
    cpuId(1, regs);
    unsigned cpuFeatures = regs[3]; // EDX
    unsigned int logical = std::thread::hardware_concurrency();
    // Get DCP cache info
    cpuId(4, regs);
    unsigned dpcCores = ((regs[0] >> 26) & 0x3f) + 1; // EAX[31:26] + 1
    bool hyperthreading = cpuFeatures & (1 << 28) && dpcCores < logical;
    // If HT is enabled, divide the logical cores by 2.
    if (hyperthreading) {
      cores = logical / 2;
    }
  } else if (cpuManufacturer == "AuthenticAMD") {
    // Get NC: Number of CPU cores - 1
    cpuId(0x80000008, regs);
    cores = ((unsigned)(regs[2] & 0xff)) + 1; // ECX[7:0] + 1
  }

  return cores;
}
} // namespace pmlc::target::x86
