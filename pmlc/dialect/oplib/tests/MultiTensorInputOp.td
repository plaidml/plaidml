// RUN: pmlc-tblgen -gen-cpp-wrappers -I llvm-project/mlir/include %s | FileCheck --dump-input-on-failure %s

include "mlir/IR/OpBase.td"

def OpLib_Tblgen_Test_Dialect : Dialect {
  let name = "test";
  let cppNamespace = "pmlc::dialect::op";
}

class PML_Op<string mnemonic, list<OpTrait> traits> :
    Op<OpLib_Tblgen_Test_Dialect, mnemonic, traits>;

def PML_MultiTensorInputOp : PML_Op<"multi_tensor_input_op", [NoSideEffect]> {
  let arguments = (ins AnyTensor:$test_input, AnyTensor:$second_test_input);
  let results = (outs AnyTensor:$test_output);
}

// CHECK-LABEL: MultiTensorInputOp wrapper
// CHECK: class MultiTensorInputOp {
// CHECK: protected:
// CHECK:   edsl::Tensor test_input_;
// CHECK:   edsl::Tensor second_test_input_;
// CHECK: private:
// CHECK:   MultiTensorInputOp(edsl::Tensor test_input, edsl::Tensor second_test_input) : test_input_(test_input), second_test_input_(second_test_input) {}
// CHECK:   operator edsl::Tensor() const {
// CHECK:     auto args = edsl::make_tuple(test_input_, second_test_input_);
// CHECK:     return details::op("MultiTensorInputOp", args).as_tensor();
// CHECK:   }
// CHECK: };
