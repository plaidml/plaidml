def plaidml_deps(root=""):
  native.new_http_archive(
    name = "boost_archive",
    url = "https://storage.googleapis.com/external_build_repo/boost_1_63_0.tar.bz2",
    sha256 = "beae2529f759f6b3bf3f4969a19c2e9d6f0c503edcb2de4a61d1428519fcb3b0",
    build_file = root + "//bzl:boost.BUILD",
  )

  native.bind(name="boost", actual="@boost_archive//:boost")
  native.bind(name="boost_regex", actual="@boost_archive//:regex")
  native.bind(name="boost_system", actual="@boost_archive//:system")
  native.bind(name="boost_thread", actual="@boost_archive//:thread")

  native.new_git_repository(
    name="easylogging_repo",
    remote="https://github.com/easylogging/easyloggingpp",
    tag="v9.95.0",
    build_file = root + "//bzl:easylogging.BUILD",
  )

  native.bind(
    name="easylogging",
    actual="@easylogging_repo//:easylogging"
  )

  native.new_http_archive(
    name = "coinlp_archive",
    url = "https://storage.googleapis.com/external_build_repo/Clp-1.16.10-tz.tgz",
    sha256 = "897a97b8829f465d1282da4e5fa61ba3ff506e1b0051733e5dde5e2451dcaebe",
    strip_prefix = "Clp-1.16.10",
    build_file = root + "//bzl:coinlp.BUILD",
  )
  native.bind(name="coinlp", actual="@coinlp_archive//:coinlp")
  native.bind(name="clp", actual="@coinlp_archive//:clp")

  native.new_git_repository(
    name="gflags_repo",
    commit="ac1a925c2bdec48e010020df93732badf75970e9",
    remote="https://github.com/gflags/gflags",
    build_file = root + "//bzl:gflags.BUILD"
  )

  native.bind(name="gflags", actual="@gflags_repo//:gflags")

  native.new_http_archive(
    name="gmock_archive",
    url="https://github.com/google/googletest/archive/release-1.8.0.tar.gz",
    sha256="58a6f4277ca2bc8565222b3bbd58a177609e9c488e8a72649359ba51450db7d8",
    strip_prefix="googletest-release-1.8.0",
    build_file = root + "//bzl:gmock.BUILD",)

  native.bind(name="gtest", actual="@gmock_archive//:gtest")
  native.bind(name="gtest_main", actual="@gmock_archive//:gtest_main")

  native.new_http_archive(
    name="half_repo",
    url="https://storage.googleapis.com/external_build_repo/half-1.11.0.zip",
    sha256="9e5ddb4b43abeafe190e780b5b606b081acb511e6edd4ef6fbe5de863a4affaf",
    strip_prefix="half-1.11.0",
    build_file = root + "//bzl:half.BUILD",
  )

  native.bind( name="half", actual="@half_repo//:half")

  native.new_git_repository(
    name = "jsoncpp_git",
    remote = "https://github.com/open-source-parsers/jsoncpp.git",
    commit = "11086dd6a7eba04289944367ca82cea71299ed70",
    build_file = root + "//bzl:jsoncpp.BUILD",
  )

  native.bind(
    name = "jsoncpp",
    actual = "@jsoncpp_git//:jsoncpp",
  )

  native.new_git_repository(
    name = "minizip_repo",
    remote = "https://github.com/nmoinvaz/minizip.git",
    commit = "36089398a362a117105ebfcb3751a269c70ab3b7",
    build_file = root + "//bzl:minizip.BUILD",
  )
  native.bind(name="minizip", actual="@minizip_repo//:minizip")

  native.new_git_repository(
    name="opencl_headers_repo",
    commit="f039db6764d52388658ef15c30b2237bbda49803",
    remote="https://github.com/KhronosGroup/OpenCL-Headers.git",
    build_file = root + "//bzl:opencl_headers.BUILD"
  )

  native.bind(
    name="opencl_headers",
    actual="@opencl_headers_repo//:inc"
  )

  native.new_git_repository(
    name="opencl_icd_loader_repo",
    commit="6849f617e991e8a46eebf746df43032175f263b3",
    remote="https://github.com/KhronosGroup/OpenCL-ICD-Loader.git",
    build_file = root + "//bzl:opencl_icd_loader.BUILD"
  )

  native.bind(
    name="opencl_icd_loader",
    actual="@opencl_icd_loader_repo//:lib"
  )

  # zlib is required by grpc
  native.new_http_archive(
    name = "zlib_archive",
    url = "https://storage.googleapis.com/external_build_repo/zlib-1.2.8.tar.gz",
    sha256 = "36658cb768a54c1d4dec43c3116c27ed893e88b02ecfcb44f2166f9c0b7f2a0d",
    build_file = root + "//bzl:zlib.BUILD",
  )
  native.bind(name="zlib", actual="@zlib_archive//:zlib")

  # required by protobuf_python
  native.new_http_archive(
    name="six_archive",
    url="https://bitbucket.org/gutworth/six/get/1.10.0.zip",
    sha256="016c8313d1fe8eefe706d5c3f88ddc51bd78271ceef0b75e0a9b400b6a8998a9",
    build_file = root + "//bzl:six.BUILD",
    strip_prefix="gutworth-six-e5218c3f66a2")

  native.bind(name="six", actual="@six_archive//:six",)