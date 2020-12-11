#ifndef __PML_COMP_ENUMS__
#define __PML_COMP_ENUMS__

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif

def BufferCopyMode : BitEnumAttr<
    "BufferCopyMode",
    "Modes that control buffer copy between host and device",
    [
      BitEnumAttrCase<"NoCopy", 0x0000>,
      BitEnumAttrCase<"HostToDevice", 0x0001>,
      BitEnumAttrCase<"DeviceToHost", 0x0002>,
    ]> {
  let cppNamespace = "pmlc::dialect::comp";
  let returnType = "BufferCopyMode";
  let convertFromStorage = "static_cast<" # returnType # ">($_self.getValue().getZExtValue())";
}

#endif // __PML_COMP_ENUMS__
