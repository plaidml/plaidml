// RUN: oplib-tblgen -gen-op-lib-cpp-wrappers -I mlir/include %s | FileCheck --dump-input-on-failure %s

include "mlir/IR/OpBase.td"

def OpLib_Tblgen_Test_Dialect : Dialect {
  let name = "test";
  let cppNamespace = "pmlc::dialect::op";
}

class PML_Op<string mnemonic, list<OpTrait> traits> :
    Op<OpLib_Tblgen_Test_Dialect, mnemonic, traits>;

def PML_BasicTensorOp : PML_Op<"basic_tensor_op", [NoSideEffect]> {
  let arguments = (ins AnyTensor:$test_input);
  let results = (outs AnyTensor:$test_output);
}

// CHECK-LABEL: BasicTensorOp wrapper
// CHECK: class BasicTensorOp {
// CHECK: protected:
// CHECK:   edsl::Tensor test_input_;
// CHECK: private:
// CHECK:   explicit BasicTensorOp(edsl::Tensor test_input) : test_input_(test_input) {}
// CHECK:   operator edsl::Tensor() const {
// CHECK:     auto args = edsl::make_tuple(test_input_);
// CHECK:     return details::op("BasicTensorOp", args).as_tensor();
// CHECK:   }
// CHECK: };

