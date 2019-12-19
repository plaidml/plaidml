// RUN: oplib-tblgen -gen-cpp-wrappers -I mlir/include %s | FileCheck --dump-input-on-failure %s

include "mlir/IR/OpBase.td"

def OpLib_Tblgen_Test_Dialect : Dialect {
  let name = "test";
  let cppNamespace = "pmlc::dialect::op";
}

class PML_Op<string mnemonic, list<OpTrait> traits> :
    Op<OpLib_Tblgen_Test_Dialect, mnemonic, traits>;

def PML_TensorOpWithIntArrayAttr : PML_Op<"multi_tensor_input_op", [NoSideEffect]> {
  let arguments = (ins AnyTensor:$test_input, ArrayAttr:$second_test_input);
  let results = (outs AnyTensor:$test_output);
}

// CHECK-LABEL: TensorOpWithIntArrayAttr wrapper
// CHECK: class TensorOpWithIntArrayAttr {
// CHECK: protected:
// CHECK:   edsl::Tensor test_input_;
// CHECK:   std::vector<int> second_test_input_;
// CHECK: private:
// CHECK:   explicit TensorOpWithIntArrayAttr(edsl::Tensor test_input) : test_input_(test_input) {}
// CHECK:   TensorOpWithIntArrayAttr second_test_input(std::vector<int> second_test_input) {
// CHECK:     second_test_input_ = second_test_input;
// CHECK:     return *this;
// CHECK:   }
// CHECK:   operator edsl::Tensor() const {
// CHECK:     auto args = edsl::make_tuple(test_input_, edsl::make_tuple(second_test_input_));
// CHECK:     return details::op("TensorOpWithIntArrayAttr", args).as_tensor();
// CHECK:   }
// CHECK: };
