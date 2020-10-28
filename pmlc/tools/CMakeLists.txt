add_subdirectory(pmlc-jit)
add_subdirectory(pmlc-opt)

add_custom_target(pmlc_tools_FileCheck ALL
  COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:FileCheck> FileCheck
  DEPENDS FileCheck
)
