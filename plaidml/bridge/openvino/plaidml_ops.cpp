// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"

namespace PlaidMLPlugin {

#define OP(_op_) extern void register##_op_();  // NOLINT[build/storage_class]
#include "plaidml_ops.def"                      // NOLINT[build/include]
#undef OP

void registerOps() {
#define OP(_op_) register##_op_();
#include "plaidml_ops.def"  // NOLINT[build/include]
#undef OP
}

}  // namespace PlaidMLPlugin
