FetchContent_Declare(
  googletest
  URL      https://github.com/google/googletest/archive/release-1.10.0.tar.gz
  URL_HASH SHA256=9dc9157a9a1551ec7a7e43daea9a694a0bb5fb8bec81235d8a1e6ef64c716dcb
)
set(gtest_force_shared_crt ON)
FetchContent_MakeAvailable(googletest)
