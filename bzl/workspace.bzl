def plaidml_workspace():
    native.new_http_archive(
        name="boost_archive",
        url="https://dl.bintray.com/boostorg/release/1.66.0/source/boost_1_66_0.tar.gz",
        sha256="bd0df411efd9a585e5a2212275f8762079fed8842264954675a4fddc46cfcf60",
        build_file=str(Label("//bzl:boost.BUILD")),
        strip_prefix="boost_1_66_0",
    )

    native.new_git_repository(
        name="easylogging_repo",
        remote="https://github.com/easylogging/easyloggingpp",
        tag="v9.95.0",
        build_file=str(Label("//bzl:easylogging.BUILD")),
    )

    native.git_repository(
        name="com_github_gflags_gflags",
        commit="038cfcd1a08ea6638bb7a75c7f632a56e2fbae1e",
        remote="https://github.com/earhart/gflags",
    )

    native.new_http_archive(
        name="gmock_archive",
        url="https://github.com/google/googletest/archive/release-1.8.0.tar.gz",
        sha256="58a6f4277ca2bc8565222b3bbd58a177609e9c488e8a72649359ba51450db7d8",
        strip_prefix="googletest-release-1.8.0",
        build_file=str(Label("//bzl:gmock.BUILD")),
    )

    native.new_http_archive(
        name="half_repo",
        url="https://storage.googleapis.com/external_build_repo/half-1.11.0.zip",
        sha256="9e5ddb4b43abeafe190e780b5b606b081acb511e6edd4ef6fbe5de863a4affaf",
        strip_prefix="half-1.11.0",
        build_file=str(Label("//bzl:half.BUILD")),
    )

    native.new_git_repository(
        name="jsoncpp_git",
        remote="https://github.com/open-source-parsers/jsoncpp.git",
        commit="11086dd6a7eba04289944367ca82cea71299ed70",
        build_file=str(Label("//bzl:jsoncpp.BUILD")),
    )

    native.new_git_repository(
        name="minizip_repo",
        remote="https://github.com/nmoinvaz/minizip.git",
        commit="36089398a362a117105ebfcb3751a269c70ab3b7",
        build_file=str(Label("//bzl:minizip.BUILD")),
    )

    native.new_git_repository(
        name="opencl_headers_repo",
        commit="f039db6764d52388658ef15c30b2237bbda49803",
        remote="https://github.com/KhronosGroup/OpenCL-Headers.git",
        build_file=str(Label("//bzl:opencl_headers.BUILD")))

    native.new_git_repository(
        name="opencl_icd_loader_repo",
        commit="6849f617e991e8a46eebf746df43032175f263b3",
        remote="https://github.com/KhronosGroup/OpenCL-ICD-Loader.git",
        build_file=str(Label("//bzl:opencl_icd_loader.BUILD")))

    native.new_http_archive(
        name="zlib_archive",
        url="https://storage.googleapis.com/external_build_repo/zlib-1.2.8.tar.gz",
        sha256="36658cb768a54c1d4dec43c3116c27ed893e88b02ecfcb44f2166f9c0b7f2a0d",
        build_file=str(Label("//bzl:zlib.BUILD")),
    )

    configure_protobuf()

def configure_protobuf():
    native.new_http_archive(
        name="com_google_protobuf",
        url="https://github.com/google/protobuf/archive/v3.5.1.zip",
        sha256="1f8b9b202e9a4e467ff0b0f25facb1642727cdf5e69092038f15b37c75b99e45",
        strip_prefix="protobuf-3.5.1",
        build_file=str(Label("//bzl:protobuf.BUILD")))

    native.new_http_archive(
        name="six_archive",
        url="https://bitbucket.org/gutworth/six/get/1.10.0.zip",
        sha256="016c8313d1fe8eefe706d5c3f88ddc51bd78271ceef0b75e0a9b400b6a8998a9",
        build_file=str(Label("//bzl:six.BUILD")),
        strip_prefix="gutworth-six-e5218c3f66a2")

    native.bind(
        name="six",
        actual="@six_archive//:six",
    )
