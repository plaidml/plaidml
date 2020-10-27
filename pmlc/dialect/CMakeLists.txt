add_subdirectory(comp)
add_subdirectory(pxa)
add_subdirectory(stdx)
#v annoying b/c eltwise, need 'addt'l td's param'
#add_subdirectory(tile)

pml_cc_library(
  NAME
    all_dialects
  DEPS
    ::comp
    ::pxa
    ::stdx
)