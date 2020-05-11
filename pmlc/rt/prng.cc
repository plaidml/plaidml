// Copyright 2020 Intel Corporation
#include "mlir/ExecutionEngine/RunnerUtils.h"
#include "pmlc/compiler/registry.h"
#include "pmlc/util/logging.h"

extern "C" void
plaidml_rt_prng(size_t stateRank, UnrankedMemRefType<float> *state,
                size_t resultRank, UnrankedMemRefType<float> *result,
                size_t newStateRank, UnrankedMemRefType<float> *newState) {
  if (resultRank <= 0) {
    // Nothing to do or a bad parameter.
    return;
  }

  // From the StridedMemRef types below we use only the sizes[]. Their offset is
  // independent of the value of the second type parameter.
  StridedMemRefType<int, 1> *stateStrided = (StridedMemRefType<int, 1> *)state;
  StridedMemRefType<float, 1> *resultStrided =
      (StridedMemRefType<float, 1> *)result;
  StridedMemRefType<int, 1> *newStateStrided =
      (StridedMemRefType<int, 1> *)newState;

  int count = resultStrided->sizes[0];
  for (size_t i = 1; i < resultRank; i++) {
    count *= resultStrided->sizes[i];
  }

  int *in_state = stateStrided->data + stateStrided->offset;
  float *buf = resultStrided->data + resultStrided->offset;
  int *out_state = newStateStrided->data + newStateStrided->offset;

  // A reimplementation of the PRNG from tile/lang/gen_special.cc.
  // x_n = (s1_n ^ s2_n ^ s3_n)
  // s1_{n+1} = (((s1_n & 4294967294) <<12) ^ (((s1_n <<13) ^ s1_n) >>19))
  // s2_{n+1} = (((s2_n & 4294967288) << 4) ^ (((s2_n << 2) ^ s2_n) >>25))
  // s3_{n+1} = (((s3_n & 4294967280) <<17) ^ (((s3_n << 3) ^ s3_n) >>11))
  for (size_t i = 0; i < count; ++i) {
    buf[i] = (in_state[0] ^ in_state[1] ^ in_state[2]) / 4294967296.0;
    out_state[0] = (((in_state[0] & 4294967294) << 12) ^
                    (((in_state[0] << 13) ^ in_state[0]) >> 19));
    out_state[1] = (((in_state[1] & 4294967288) << 4) ^
                    (((in_state[1] << 2) ^ in_state[1]) >> 25));
    out_state[2] = (((in_state[2] & 4294967280) << 17) ^
                    (((in_state[2] << 3) ^ in_state[2]) >> 11));
    in_state = out_state;
  }
}

namespace {
struct Registration {
  Registration() {
    using pmlc::compiler::registerSymbol;
    registerSymbol("plaidml_rt_prng",
                   reinterpret_cast<void *>(plaidml_rt_prng));
  }
};
static Registration reg;
} // namespace
