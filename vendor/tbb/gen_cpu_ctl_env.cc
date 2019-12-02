#include <float.h>

#include <cstddef>

#define private public
#include "tbb/tbb_machine.h"
#undef private

const int FE_TONEAREST = 0x0000, FE_DOWNWARD = 0x0400, FE_UPWARD = 0x0800, FE_TOWARDZERO = 0x0c00,
          FE_RND_MODE_MASK = FE_TOWARDZERO, SSE_RND_MODE_MASK = FE_RND_MODE_MASK << 3, SSE_DAZ = 0x0040,
          SSE_FTZ = 0x8000, SSE_MODE_MASK = SSE_DAZ | SSE_FTZ, SSE_STATUS_MASK = 0x3F;
const int NumSseModes = 4;
const int SseModes[NumSseModes] = {0, SSE_DAZ, SSE_FTZ, SSE_DAZ | SSE_FTZ};

void __TBB_get_cpu_ctl_env(tbb::internal::cpu_ctl_env* fe) {
  fe->x87cw = int16_t(_control87(0, 0) & _MCW_RC) << 2;
  fe->mxcsr = _mm_getcsr();
}

void __TBB_set_cpu_ctl_env(const tbb::internal::cpu_ctl_env* fe) {
  _control87((fe->x87cw & FE_RND_MODE_MASK) >> 6, _MCW_RC);
  _mm_setcsr(fe->mxcsr);
}
