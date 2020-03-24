// Copyright 2019 Intel Corporation.

#pragma once

#include <string>
#include <vector>

#include "plaidml/edsl/edsl.h"

namespace plaidml {
namespace edsl {

// Given a forward pass tensor operation that takes inputs `wrt` and produces output `loss`,
// compute the gradients for each Tensor in `wrt`.
inline std::vector<Tensor> Gradient(const std::vector<Tensor>& wrt, const Tensor& loss) {
  std::vector<plaidml_expr*> wrt_exprs(wrt.size());
  std::vector<plaidml_expr*> deriv_exprs(wrt.size());
  for (size_t i = 0; i < wrt.size(); ++i) {
    wrt_exprs[i] = wrt[i].as_ptr();
  }
  ffi::call_void(             //
      plaidml_expr_gradient,  //
      wrt_exprs.size(),       //
      wrt_exprs.data(),       //
      loss.as_ptr(),          //
      deriv_exprs.data());
  std::vector<Tensor> ret(wrt.size());
  for (size_t i = 0; i < wrt.size(); ++i) {
    ret[i] = Tensor(deriv_exprs[i]);
  }
  return ret;
}

inline void RegisterTensorDeriv(const std::string& name, TensorDeriv fn) {
  auto thunk = TensorDerivThunk();
  ffi::call_void(plaidml_deriv_register, name.c_str(), thunk, reinterpret_cast<void*>(fn));
}

}  // namespace edsl
}  // namespace plaidml
