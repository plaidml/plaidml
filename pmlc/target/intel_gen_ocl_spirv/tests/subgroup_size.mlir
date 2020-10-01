// RUN: pmlc-opt --intel-gen-ocl-set-subgroup-size %s | FileCheck %s

spv.module @subgroup Physical64 OpenCL {
  spv.func @kernel() "None" attributes {spv.entry_point_abi = {local_size = dense<[16, 3, 3]> : vector<3xi32>}, workgroup_attributions = 0 : i64} {
    spv.Return
  }
  // CHECK: spv.ExecutionMode @kernel "SubgroupSize", 16
}

spv.module @no_subgroup Physical64 OpenCL {
  spv.func @kernel() "None" attributes {spv.entry_point_abi = {local_size = dense<[1, 3, 3]> : vector<3xi32>}, workgroup_attributions = 0 : i64} {
    spv.Return
  }
  // CHECK-NOT: spv.ExecutionMode @kernel "SubgroupSize"
}

spv.module @func_and_kernel Physical64 OpenCL {
  spv.func @func() "None" {
    spv.Return
  }

  spv.func @kernel() "None" attributes {spv.entry_point_abi = {local_size = dense<[32, 3, 3]> : vector<3xi32>}, workgroup_attributions = 0 : i64} {
    spv.Return
  }
  // CHECK: spv.ExecutionMode @kernel "SubgroupSize", 32
}

spv.module @only_func Physical64 OpenCL {
  spv.func @func() "None" {
    spv.Return
  }
  // CHECK-NOT: spv.ExecutionMode
}
