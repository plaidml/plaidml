// Copyright 2019 Intel Corporation.

#pragma once

#include "plaidml2/edsl/edsl.h"
#include "plaidml2/op/ffi.h"

namespace plaidml {
namespace op {

inline void init() {  //
  plaidml::init();
  plaidml::edsl::init();
  ffi::call_void(plaidml_op_init);
}

}  // namespace op
}  // namespace plaidml
