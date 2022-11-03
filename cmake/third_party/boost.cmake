set(BOOST_URL "https://boostorg.jfrog.io/artifactory/main/release/1.77.0/source/boost_1_77_0.tar.bz2" CACHE STRING "Boost download URL")
set(BOOST_URL_SHA256 "fc9f85fc030e233142908241af7a846e60630aa7388de9a5fafb1f3a26840854" CACHE STRING "Boost download URL SHA256 checksum")

include(FetchContent)
FetchContent_Declare(
  Boost
  URL ${BOOST_URL}
  URL_HASH SHA256=${BOOST_URL_SHA256}
)
FetchContent_GetProperties(Boost)

if(NOT Boost_POPULATED)
  message(STATUS "Fetching Boost")
  FetchContent_Populate(Boost)
  message(STATUS "Fetching Boost - done")
  set(BOOST_SOURCE ${boost_SOURCE_DIR})
endif()

# Define the header-only Boost target
add_library(Boost::boost INTERFACE IMPORTED GLOBAL)
target_include_directories(Boost::boost SYSTEM INTERFACE ${BOOST_SOURCE})

# Disable autolink
target_compile_definitions(Boost::boost INTERFACE BOOST_ALL_NO_LIB=1)
