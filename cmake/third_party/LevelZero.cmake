FetchContent_Declare(
  LevelZero
  URL      https://github.com/oneapi-src/level-zero/archive/v1.0.22.zip
  URL_HASH SHA256=ad9a757c5e07cd44d8e63019e63ba8a8ea19981ea4b7567d3099d3ef80567908
)
FetchContent_GetProperties(LevelZero)
if(NOT levelzero_POPULATED)
    FetchContent_Populate(LevelZero)
    set_property(GLOBAL PROPERTY LEVEL_ZERO_INCLUDE_DIRS ${levelzero_SOURCE_DIR}/include)
    add_subdirectory(${levelzero_SOURCE_DIR})
endif()
