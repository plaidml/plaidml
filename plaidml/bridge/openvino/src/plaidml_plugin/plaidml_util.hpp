// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "ie_layouts.h"  // NOLINT[build/include_subdir]
#include "ie_precision.hpp"

#include "plaidml/edsl/edsl.h"
#include "plaidml/exec/exec.h"

namespace PlaidMLPlugin {

plaidml::DType to_plaidml(InferenceEngine::Precision prec);
plaidml::edsl::LogicalShape to_plaidml(const InferenceEngine::TensorDesc& desc);
std::vector<int64_t> to_plaidml(const InferenceEngine::SizeVector& dims);

}  // namespace PlaidMLPlugin
