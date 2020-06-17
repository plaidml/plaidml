// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "ie_layouts.h"  // NOLINT[build/include_subdir]
#include "ie_precision.hpp"

#include "ngraph/axis_set.hpp"
#include "ngraph/axis_vector.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/type/element_type.hpp"

#include "plaidml/edsl/edsl.h"
#include "plaidml/op/op.h"

namespace PlaidMLPlugin {

ngraph::AxisSet get_axes_set_from_constant_operand(size_t operand_idx, ngraph::Node* layer);
ngraph::AxisVector get_axes_vector_from_constant_operand(size_t operand_idx, ngraph::Node* layer);
plaidml::DType to_plaidml(const ngraph::element::Type& ng_type);

plaidml::op::AutoPadMode to_plaidml(const ngraph::op::PadType& ng_type);

}  // namespace PlaidMLPlugin
