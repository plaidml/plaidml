FetchContent_Declare(
  half
  URL      https://github.com/plaidml/depot/raw/master/half-1.11.0.zip
  URL_HASH SHA256=9e5ddb4b43abeafe190e780b5b606b081acb511e6edd4ef6fbe5de863a4affaf
)
FetchContent_MakeAvailable(half)

add_library(half INTERFACE)
target_include_directories(half INTERFACE ${half_SOURCE_DIR}/include)
