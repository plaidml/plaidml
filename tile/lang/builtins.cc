#include "tile/lang/builtins.h"

#include <vector>

namespace vertexai {
namespace tile {
namespace lang {

static std::shared_ptr<BoundFunction> ddef(const std::vector<std::string>& derivs) {
  size_t size = derivs.size();
  std::string r = "function (";
  for (size_t i = 0; i < size; i++) {
    r += "X" + std::to_string(1 + i) + ", ";
  }
  r += "Y, DY) -> (";
  for (size_t i = 0; i < size; i++) {
    r += "DX" + std::to_string(1 + i);
    if (i + 1 != size) {
      r += ", ";
    }
  }
  r += ") {";
  for (size_t i = 0; i < size; i++) {
    r += "DX" + std::to_string(1 + i) + " = " + derivs[i] + ";";
  }
  r += "}";
  return std::make_shared<BoundFunction>(r);
}

static Program idef(const std::string& str) {
  Parser p;
  return p.Parse(str);
}

std::map<std::string, Program> InlineDefines = {
    {"abs", idef("function (X1) -> (Y) { Y = (X1 < 0 ? -X1 : X1); }")},
    {"max", idef("function (X1, X2) -> (Y) { Y = (X1 < X2 ? X2 : X1); }")},
    {"min", idef("function (X1, X2) -> (Y) { Y = (X1 < X2 ? X1 : X2); }")},
    {"relu", idef("function (X1) -> (Y) { Y = (X1 < 0 ? 0 : X1); }")},
    {"sigmoid", idef("function (X1) -> (Y) { Y = (1.0 / (1.0 + exp(-X1))); }")},
    {"builtin_softmax", idef(R"***(
      function (X1, X2, X3) -> (Y) {
        M[i, 0 : X2, 1] = >(X1[i, j]);
        E = exp(X1 - M);
        N[i, 0 : X2, 1] = +(E[i, j]);
        Y = E / N;
      } )***")},
    {"builtin_logsoftmax", idef(R"***(
      function (X1, X2, X3) -> (Y) {
        M[i, 0 : X2, 1] = >(X1[i, j]);
        E = exp(X1 - M);
        N[i, 0 : X2, 1] = +(E[i, j]);
        Y = X1 - (M + log(N));
      } )***")},
    // In binary crossentropy, X3 is a scaling factor used only in the derivative
    {"builtin_binary_crossentropy", idef(R"***(
      function (X1, X2, X3) -> (Y) {
        Y = -X2*log(X1) - (1-X2)*log(1-X1);
      } )***")},
    {"reverse_grad", idef("function (X1, X2) -> (Y) { Y = X1; }")},
};

std::map<std::string, std::shared_ptr<BoundFunction>> DerivDefines = {
    {"abs", ddef({"(Y < 0 ? -DY : DY)"})},
    {"add", ddef({"DY", "DY"})},
    {"as_float", ddef({"0", "0"})},
    {"as_int", ddef({"0", "0"})},
    {"as_uint", ddef({"0", "0"})},
    {"bit_and", ddef({"0", "0"})},
    {"bit_or", ddef({"0", "0"})},
    {"bit_xor", ddef({"0", "0"})},
    {"bit_left", ddef({"0", "0"})},
    {"bit_right", ddef({"0", "0"})},
    {"bit_not", ddef({"0"})},
    {"sub", ddef({"DY", "-DY"})},
    {"mul", ddef({"X2*DY", "X1*DY"})},
    {"div", ddef({"DY/X2", "-X1*DY/(X2*X2)"})},
    {"cmp_eq", ddef({"0", "0"})},
    {"cmp_ne", ddef({"0", "0"})},
    {"cmp_gt", ddef({"0", "0"})},
    {"cmp_lt", ddef({"0", "0"})},
    {"cmp_ge", ddef({"0", "0"})},
    {"cmp_le", ddef({"0", "0"})},
    {"cond", ddef({"0", "cond(X1, DY, 0)", "cond(X1, 0, DY)"})},
    {"neg", ddef({"-DY"})},
    {"recip", ddef({"-Y*Y*DY"})},
    {"sqrt", ddef({"DY/(2*Y)"})},
    {"exp", ddef({"exp(X1)*DY"})},
    {"log", ddef({"DY/X1"})},
    {"pow", ddef({"DY * X2 * pow(X1, X2 - 1)", "log(X1)*Y*DY"})},
    {"tanh", ddef({"DY*(1 - Y*Y)"})},
    {"max", ddef({"X1 < X2 ? 0 : DY"})},
    {"min", ddef({"X1 < X2 ? DY : 0"})},
    {"relu", ddef({"(Y < 0 ? 0 : DY)"})},
    {"sigmoid", ddef({"Y*(1.0 - Y)*DY"})},
    {"shape", ddef({"0"})},
    {"gather", std::make_shared<BoundFunction>(R"***(
      function (X1, X2, Y, DY) -> (DX1, DX2) {
        DX1 = scatter(DY, X2, X1);
        DX2 = 0;
      } )***")},
    {"builtin_softmax", std::make_shared<BoundFunction>(R"***(
      function (X1, X2, X3, Y, DY) -> (DX1, DX2, DX3) {
        DYY = (DY * Y);
        T[i : X2] = +(DYY[i, j]);
        TB[i, j : X2, X3] = +(T[i]);
        DX1 = DYY - TB * Y; 
        DX2 = 0;
        DX3 = 0;
      } )***")},
    {"builtin_logsoftmax", std::make_shared<BoundFunction>(R"***(
      function (X1, X2, X3, Y, DY) -> (DX1, DX2, DX3) {
        SM = builtin_softmax(X1, X2, X3);
        TDY[i, 0 : X2, 1] = +(DY[i, j]);
        DX1 = DY - SM * TDY;
        DX2 = 0;
        DX3 = 0;
      } )***")},
    {"builtin_binary_crossentropy", ddef({"(-X2/X1 + (1-X2)/(1-X1))/X3", "log(1-X1) - log(X1)", "0"})},
    {"reverse_grad", std::make_shared<BoundFunction>(R"***(
      function (X1, X2, Y, DY) -> (DX1, DX2) {
        DX1 = -X2*DY;
        DX2 = 0;
    } )***")},
};

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
