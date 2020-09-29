#ifndef __PML_VULKAN_ENUMS__
#define __PML_VULKAN_ENUMS__

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif

def BUFFER_COPY_MODE : BitEnumAttr<
    "BUFFER_COPY_MODE",
    "Modes that control buffer copy between host and device",
    [
      BitEnumAttrCase<"NO_COPY", 0x0000>,
      BitEnumAttrCase<"HOST_TO_DEVICE", 0x0001>,
      BitEnumAttrCase<"DEVICE_TO_HOST", 0x0002>,
    ]> {
  let cppNamespace = "pmlc::conversion::gpu";
  let returnType = "BUFFER_COPY_MODE";
  let convertFromStorage = "static_cast<" # returnType # ">($_self.getValue().getZExtValue())";
}

#endif // __PML_VULKAN_ENUMS__
