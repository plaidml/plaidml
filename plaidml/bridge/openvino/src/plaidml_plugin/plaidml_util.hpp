// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "ie_layouts.h"  // NOLINT[build/include_subdir]
#include "ie_precision.hpp"

#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/type/element_type.hpp"

#include "plaidml/edsl/edsl.h"
#include "plaidml/op/op.h"

namespace PlaidMLPlugin {

plaidml::DType to_plaidml(const ngraph::element::Type& ng_type);

plaidml::op::AutoPadMode to_plaidml(const ngraph::op::PadType& ng_type);

}  // namespace PlaidMLPlugin
