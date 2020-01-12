// Copyright 2019 Intel Corporation.

#include "plaidml/edsl/derivs.h"

#include <vector>

#include "plaidml/edsl/autodiff.h"

namespace plaidml::edsl {

#define DERIV_ARGS const Tensor &Y, const Tensor &DY, const std::vector<Tensor>&X

using Tensors = std::vector<Tensor>;

void RegisterDerivs() {
  // TODO: A number of operations for which I am uncertain if we want to add a derivative here have been marked "TODO"
  RegisterTensorDeriv("eltwise.abs", [](DERIV_ARGS) {  //
    return Tensors{select(X[0] < 0, -DY, DY)};
  });
  RegisterTensorDeriv("eltwise.acos", [](DERIV_ARGS) {  //
    return Tensors{-DY / sqrt(1 - X[0] * X[0])};
  });
  RegisterTensorDeriv("eltwise.add", [](DERIV_ARGS) {  //
    return Tensors{DY, DY};
  });
  // TODO: And
  RegisterTensorDeriv("eltwise.asin", [](DERIV_ARGS) {  //
    return Tensors{DY / sqrt(1 - X[0] * X[0])};
  });
  // TODO: Assign
  RegisterTensorDeriv("eltwise.atan", [](DERIV_ARGS) {  //
    return Tensors{DY / (1 + X[0] * X[0])};
  });
  // TODO: Ceil
  RegisterTensorDeriv("eltwise.cmp_eq", [](DERIV_ARGS) {  //
    return Tensors{zero(), zero()};
  });
  RegisterTensorDeriv("eltwise.cmp_ne", [](DERIV_ARGS) {  //
    return Tensors{zero(), zero()};
  });
  RegisterTensorDeriv("eltwise.cmp_lt", [](DERIV_ARGS) {  //
    return Tensors{zero(), zero()};
  });
  RegisterTensorDeriv("eltwise.cmp_gt", [](DERIV_ARGS) {  //
    return Tensors{zero(), zero()};
  });
  RegisterTensorDeriv("eltwise.cmp_le", [](DERIV_ARGS) {  //
    return Tensors{zero(), zero()};
  });
  RegisterTensorDeriv("eltwise.cmp_ge", [](DERIV_ARGS) {  //
    return Tensors{zero(), zero()};
  });
  RegisterTensorDeriv("eltwise.cosh", [](DERIV_ARGS) {  //
    return Tensors{sinh(X[0]) * DY};
  });
  RegisterTensorDeriv("eltwise.cos", [](DERIV_ARGS) {  //
    return Tensors{-sin(X[0]) * DY};
  });
  RegisterTensorDeriv("eltwise.div", [](DERIV_ARGS) {  //
    return Tensors{
        DY / X[1],
        -X[0] * DY / (X[1] * X[1]),
    };
  });
  RegisterTensorDeriv("eltwise.exp", [](DERIV_ARGS) {  //
    return Tensors{exp(X[0]) * DY};
  });
  // TODO: Floor
  RegisterTensorDeriv("eltwise.ident", [](DERIV_ARGS) {  //
    return Tensors{DY};
  });
  RegisterTensorDeriv("eltwise.log", [](DERIV_ARGS) {  //
    return Tensors{DY / X[0]};
  });
  RegisterTensorDeriv("eltwise.max", [](DERIV_ARGS) {  //
    return Tensors{
        select(X[0] < X[1], zero(), DY),
        select(X[0] < X[1], DY, zero()),
    };
  });
  RegisterTensorDeriv("eltwise.min", [](DERIV_ARGS) {  //
    return Tensors{
        select(X[0] < X[1], DY, zero()),
        select(X[0] < X[1], zero(), DY),
    };
  });
  // TODO: Mod
  RegisterTensorDeriv("eltwise.mul", [](DERIV_ARGS) {  //
    return Tensors{
        X[1] * DY,
        X[0] * DY,
    };
  });
  RegisterTensorDeriv("eltwise.neg", [](DERIV_ARGS) {  //
    return Tensors{-DY};
  });
  // TODO: Not
  // TODO: Or
  RegisterTensorDeriv("eltwise.pow", [](DERIV_ARGS) {  //
    return Tensors{
        DY * X[1] * pow(X[0], X[1] - 1),
        log(X[0]) * Y * DY,
    };
  });
  // TODO: Relu
  // TODO: Round
  RegisterTensorDeriv("eltwise.select", [](DERIV_ARGS) {  //
    // TODO: Can disable this to find weird cases that are using select when they shouldn't
    return Tensors{
        zero(),
        select(X[0], DY, zero()),
        select(X[0], zero(), DY),
    };
  });
  // TODO: Shl
  // TODO: Shr
  // TODO: Sign
  RegisterTensorDeriv("eltwise.sinh", [](DERIV_ARGS) {  //
    return Tensors{cosh(X[0]) * DY};
  });
  RegisterTensorDeriv("eltwise.sin", [](DERIV_ARGS) {  //
    return Tensors{cos(X[0]) * DY};
  });
  RegisterTensorDeriv("eltwise.sqrt", [](DERIV_ARGS) {  //
    return Tensors{DY / (2 * Y)};
  });
  RegisterTensorDeriv("eltwise.sub", [](DERIV_ARGS) {  //
    return Tensors{DY, -DY};
  });
  RegisterTensorDeriv("eltwise.tanh", [](DERIV_ARGS) {  //
    return Tensors{DY * (1 - Y * Y)};
  });
  RegisterTensorDeriv("eltwise.tan", [](DERIV_ARGS) {  //
    return Tensors{(1 + Y * Y) * DY};
  });
  // TODO: Xor
}

}  // namespace plaidml::edsl
