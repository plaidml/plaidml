#ifndef __VULKAN_OPS__
#define __VULKAN_OPS__

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif

include "mlir/Interfaces/SideEffectInterfaces.td"
include "pmlc/util/enums.td"
include "pmlc/util/interfaces.td"

def Vk_Dialect : Dialect {
  let name = "vk";
  let summary = "Vulkan Dialect";
  let cppNamespace = "pmlc::dialect::vulkan";
}

def VkBuffer : DialectType<Vk_Dialect,
  CPred<"$_self.isa<BufferType>()">,
  "VkBuffer">;

def VkShaderModule : DialectType<Vk_Dialect,
  CPred<"$_self.isa<ShaderModuleType>()">,
  "VkShaderModule">;

class Vk_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Vk_Dialect, mnemonic, traits> {
}

def Vk_CreateShaderModuleOp : Vk_Op<"create_shader_module"> {
  let arguments = (ins StrAttr:$binary);
  let results = (outs VkShaderModule:$module);
}

def Vk_Alloc : Vk_Op<"alloc"> {
  let results = (outs VkBuffer);
}

def Vk_Copy : Vk_Op<"copy"> {
  let arguments = (ins VkBuffer:$src);
  let results = (outs VkBuffer:$dst);
}

def Vk_Dispatch : Vk_Op<"dispatch"> {
  let arguments = (ins
    VkShaderModule:$module,
    StrAttr:$entry_point,
    Index:$workgroup_x,
    Index:$workgroup_y,
    Index:$workgroup_z
  );
  let assemblyFormat = [{
    type($module) $module `,` `entry_point` `=` $entry_point `,`
    `workgroup` `=` `[` $workgroup_x `,` $workgroup_y `,` $workgroup_z `]`
    attr-dict
  }];
}

def Vk_InitVulkanCall : Vk_Op<"initvulkancall"> {
  let results = (outs VkBuffer);
  let builders = [OpBuilder<
    "Builder builder, OperationState &result, Type resultType, "
    "FlatSymbolRefAttr callee",
    [{
      result.addAttribute("callee", callee);
      result.addTypes(resultType);
    }]>,
  ];
}

def Vk_CreateVulkanLaunchKernelAction : Vk_Op<"createVulkanLaunchKernelAction"> {
  let builders = [OpBuilder<
    "Builder builder, OperationState &result, FlatSymbolRefAttr callee, "
    "ArrayRef<Value> operands = {}",
    [{
      result.addAttribute("callee", callee);
    }]>,
  ];
}

def Vk_SetLaunchKernelAction : Vk_Op<"setLaunchKernelAction"> {
  let builders = [OpBuilder<
    "Builder builder, OperationState &result, FlatSymbolRefAttr callee, "
    "ArrayRef<Value> operands = {}",
    [{
      result.addAttribute("callee", callee);
    }]>,
  ];
}

def Vk_AddVulkanLaunchActionToSchedule : Vk_Op<"AddVulkanLaunchActionToSchedule"> {
  let builders = [OpBuilder<
    "Builder builder, OperationState &result, FlatSymbolRefAttr callee, "
    "ArrayRef<Value> operands = {}",
    [{
      result.addAttribute("callee", callee);
    }]>,
  ];
}

def Vk_SubmitCommandBuffers : Vk_Op<"SubmitCommandBuffers"> {
  let builders = [OpBuilder<
    "Builder builder, OperationState &result, FlatSymbolRefAttr callee, "
    "ArrayRef<Value> operands = {}",
    [{
      result.addAttribute("callee", callee);
    }]>,
  ];
}

def Vk_DeinitVulkan : Vk_Op<"DeinitVulkan"> {
  let builders = [OpBuilder<
    "Builder builder, OperationState &result, FlatSymbolRefAttr callee, "
    "ArrayRef<Value> operands = {}",
    [{
      result.addAttribute("callee", callee);
    }]>,
  ];
}

#endif // __VULKAN_OPS__
