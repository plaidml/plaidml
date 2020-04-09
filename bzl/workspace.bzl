# Copyright 2019 Intel Corporation

load("//bzl:conda_repo.bzl", "conda_repo")
load("//bzl:xsmm_repo.bzl", "xsmm_repo")
load("//vendor/bazel:repo.bzl", "http_archive")
load("//vendor/cuda:configure.bzl", "configure_cuda")
load("//vendor/cm:configure.bzl", "configure_cm")

# Sanitize a dependency so that it works correctly from code that includes it as a submodule.
def clean_dep(dep):
    return str(Label(dep))

def plaidml_workspace():
    configure_toolchain()

    http_archive(
        name = "bazel_latex",
        sha256 = "5119802a5fbe2f27914af455c59b4ecdaaf57c0bc6c63da38098a30d94f48c9a",
        strip_prefix = "bazel-latex-b6375d9df2952548c3371c0c865710655e8b1cc1",
        url = "https://github.com/plaidml/bazel-latex/archive/b6375d9df2952548c3371c0c865710655e8b1cc1.zip",
    )

    http_archive(
        name = "boost",
        url = "https://github.com/plaidml/depot/raw/master/boost_1_66_0.tar.gz",
        sha256 = "bd0df411efd9a585e5a2212275f8762079fed8842264954675a4fddc46cfcf60",
        strip_prefix = "boost_1_66_0",
        build_file = clean_dep("//bzl:boost.BUILD"),
    )

    http_archive(
        name = "com_github_google_benchmark",
        url = "https://github.com/google/benchmark/archive/v1.5.0.tar.gz",
        sha256 = "3c6a165b6ecc948967a1ead710d4a181d7b0fbcaa183ef7ea84604994966221a",
        strip_prefix = "benchmark-1.5.0",
    )

    http_archive(
        name = "easylogging",
        url = "https://github.com/muflihun/easyloggingpp/releases/download/v9.95.0/easyloggingpp_v9.95.0.tar.gz",
        sha256 = "4b1aebe19e383349c6e438aac357eccfabb0ce34430e872508ed8ee0d1629e0f",
        build_file = clean_dep("//bzl:easylogging.BUILD"),
    )

    http_archive(
        name = "gflags",
        url = "https://github.com/gflags/gflags/archive/e292e0452fcfd5a8ae055b59052fc041cbab4abf.zip",
        sha256 = "7d17922978692175c67ef5786a014df44bfbfe3b48b30937cca1413d4ff65f75",
        strip_prefix = "gflags-e292e0452fcfd5a8ae055b59052fc041cbab4abf",
    )

    http_archive(
        name = "gmock",
        url = "https://github.com/google/googletest/archive/release-1.8.0.tar.gz",
        sha256 = "58a6f4277ca2bc8565222b3bbd58a177609e9c488e8a72649359ba51450db7d8",
        strip_prefix = "googletest-release-1.8.0",
        build_file = clean_dep("//bzl:gmock.BUILD"),
    )

    http_archive(
        name = "half",
        url = "https://github.com/plaidml/depot/raw/master/half-1.11.0.zip",
        sha256 = "9e5ddb4b43abeafe190e780b5b606b081acb511e6edd4ef6fbe5de863a4affaf",
        strip_prefix = "half-1.11.0",
        build_file = clean_dep("//bzl:half.BUILD"),
    )

    http_archive(
        name = "jsoncpp",
        url = "https://github.com/open-source-parsers/jsoncpp/archive/11086dd6a7eba04289944367ca82cea71299ed70.zip",
        sha256 = "2099839a06c867a8b3abf81b3eb82dc0ed67207fd0d2b940b6cf2efef66fe7d8",
        strip_prefix = "jsoncpp-11086dd6a7eba04289944367ca82cea71299ed70",
        build_file = clean_dep("//bzl:jsoncpp.BUILD"),
    )

    http_archive(
        name = "jsonnet",
        url = "https://github.com/google/jsonnet/archive/v0.13.0.zip",
        sha256 = "e9f7095dd2a383001188aa622edaf82059732e11d74f8d0bfdfa84f2682dd547",
        strip_prefix = "jsonnet-0.13.0",
    )

    http_archive(
        name = "io_bazel_rules_jsonnet",
        sha256 = "d05d719c4738e8aac5f13b32f745ff4832b9638ecc89ddcb6e36c379a1ada025",
        strip_prefix = "rules_jsonnet-0.1.0",
        url = "https://github.com/bazelbuild/rules_jsonnet/archive/0.1.0.zip",
    )

    http_archive(
        name = "minizip",
        url = "https://github.com/nmoinvaz/minizip/archive/36089398a362a117105ebfcb3751a269c70ab3b7.zip",
        sha256 = "c47b06ad7ef10d01a8d415b1b8dfb3691dad6ed41b38756fbf8fd6c074480d0f",
        strip_prefix = "minizip-36089398a362a117105ebfcb3751a269c70ab3b7",
        build_file = clean_dep("//bzl:minizip.BUILD"),
    )

    http_archive(
        name = "opencl_headers",
        url = "https://github.com/KhronosGroup/OpenCL-Headers/archive/f039db6764d52388658ef15c30b2237bbda49803.zip",
        sha256 = "b2b813dd88a7c39eb396afc153070f8f262504a7f956505b2049e223cfc2229b",
        strip_prefix = "OpenCL-Headers-f039db6764d52388658ef15c30b2237bbda49803",
        build_file = clean_dep("//bzl:opencl_headers.BUILD"),
    )

    http_archive(
        name = "cm_headers",
        url = "https://github.com/intel/cm-compiler/releases/download/Release_20190717/Linux_C_for_Metal_Development_Package_20190717.zip",
        sha256 = "4549496e3742ade2ff13e804654cb4ee7ddabb3b95dbc1fdeb9ca22141f317d5",
        strip_prefix = "Linux_C_for_Metal_Development_Package_20190717",
        build_file = clean_dep("//bzl:cm_headers.BUILD"),
    )

    http_archive(
        name = "libva",
        url = "https://github.com/intel/libva/releases/download/2.5.0/libva-2.5.0.tar.bz2",
        sha256 = "3aa89cd369a506ac4dbe5de7c0ef5da4f3d220bf986403f02fa1f6f702af6878",
        strip_prefix = "libva-2.5.0",
        build_file = clean_dep("//bzl:libva.BUILD"),
    )

    http_archive(
        name = "pybind11",
        url = "https://github.com/pybind/pybind11/archive/v2.2.4.tar.gz",
        sha256 = "b69e83658513215b8d1443544d0549b7d231b9f201f6fc787a2b2218b408181e",
        strip_prefix = "pybind11-2.2.4",
        build_file = clean_dep("//bzl:pybind11.BUILD"),
    )

    http_archive(
        name = "rules_pkg",
        sha256 = "e46b4f5aa71d1037c7c8142e2fedb503127af4bbd9dbde4a742d119749f68a3f",
        strip_prefix = "rules_pkg-cb54c427343aa48c32e3c09ddcc8f6316cdbd5a6/pkg",
        url = "https://github.com/bazelbuild/rules_pkg/archive/cb54c427343aa48c32e3c09ddcc8f6316cdbd5a6.tar.gz",
    )

    http_archive(
        name = "rules_python",
        sha256 = "b5bab4c47e863e0fbb77df4a40c45ca85f98f5a2826939181585644c9f31b97b",
        strip_prefix = "rules_python-9d68f24659e8ce8b736590ba1e4418af06ec2552",
        url = "https://github.com/bazelbuild/rules_python/archive/9d68f24659e8ce8b736590ba1e4418af06ec2552.tar.gz",
    )

    http_archive(
        name = "zlib",
        url = "https://github.com/plaidml/depot/raw/master/zlib-1.2.8.tar.gz",
        sha256 = "36658cb768a54c1d4dec43c3116c27ed893e88b02ecfcb44f2166f9c0b7f2a0d",
        build_file = clean_dep("//bzl:zlib.BUILD"),
    )

    configure_protobuf()
    configure_cuda(name = "cuda")
    configure_cm(name = "cm")

    conda_repo(
        name = "com_intel_plaidml_conda_unix",
        env = clean_dep("//conda:unix.yml"),
        build_file = clean_dep("//conda:unix.BUILD"),
    )

    conda_repo(
        name = "com_intel_plaidml_conda_windows",
        env = clean_dep("//conda:windows.yml"),
        build_file = clean_dep("//conda:windows.BUILD"),
    )

    xsmm_repo(
        name = "xsmm",
        url = "https://github.com/hfp/libxsmm/archive/1.12.1.zip",
        sha256 = "451ec9d30f0890bf3081aa3d0d264942a6dea8f9d29c17bececc8465a10a832b",
        strip_prefix = "libxsmm-1.12.1",
        build_file = clean_dep("//bzl:xsmm.BUILD"),
    )

    http_archive(
        name = "tbb",
        url = "https://github.com/intel/tbb/archive/tbb_2019.zip",
        sha256 = "3bb395989ce4701fc5d486ab8db3d639e9f86111a761de358d8e02bb6ed1e076",
        strip_prefix = "oneTBB-tbb_2019",
        build_file = clean_dep("//vendor/tbb:tbb.BUILD"),
    )

    LLVM_COMMIT = "a21beccea2020f950845cbb68db663d0737e174c"
    LLVM_SHA256 = "73682f2b78c1c46621afb69b850e50c4d787f9c77fb3b53ac50fc42ffbac0493"
    LLVM_URL = "https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT)
    http_archive(
        name = "llvm-project",
        url = LLVM_URL,
        sha256 = LLVM_SHA256,
        strip_prefix = "llvm-project-" + LLVM_COMMIT,
        link_files = {
            clean_dep("//vendor/llvm:llvm.BUILD"): "llvm/BUILD.bazel",
            clean_dep("//vendor/mlir:mlir.BUILD"): "mlir/BUILD.bazel",
        },
        patches = [clean_dep("//vendor/mlir:mlir.patch")],
        override = "PLAIDML_LLVM_REPO",
    )

def configure_protobuf():
    http_archive(
        name = "com_google_protobuf",
        url = "https://github.com/protocolbuffers/protobuf/archive/v3.9.0.tar.gz",
        sha256 = "2ee9dcec820352671eb83e081295ba43f7a4157181dad549024d7070d079cf65",
        strip_prefix = "protobuf-3.9.0",
        build_file = clean_dep("//bzl:protobuf.BUILD"),
    )

    http_archive(
        name = "rules_cc",
        sha256 = "29daf0159f0cf552fcff60b49d8bcd4f08f08506d2da6e41b07058ec50cfeaec",
        strip_prefix = "rules_cc-b7fe9697c0c76ab2fd431a891dbb9a6a32ed7c3e",
        url = "https://github.com/bazelbuild/rules_cc/archive/b7fe9697c0c76ab2fd431a891dbb9a6a32ed7c3e.tar.gz",
    )

    http_archive(
        name = "rules_java",
        sha256 = "f5a3e477e579231fca27bf202bb0e8fbe4fc6339d63b38ccb87c2760b533d1c3",
        strip_prefix = "rules_java-981f06c3d2bd10225e85209904090eb7b5fb26bd",
        url = "https://github.com/bazelbuild/rules_java/archive/981f06c3d2bd10225e85209904090eb7b5fb26bd.tar.gz",
    )

    http_archive(
        name = "rules_proto",
        sha256 = "88b0a90433866b44bb4450d4c30bc5738b8c4f9c9ba14e9661deb123f56a833d",
        strip_prefix = "rules_proto-b0cc14be5da05168b01db282fe93bdf17aa2b9f4",
        url = "https://github.com/bazelbuild/rules_proto/archive/b0cc14be5da05168b01db282fe93bdf17aa2b9f4.tar.gz",
    )

    http_archive(
        name = "six_archive",
        url = "https://bitbucket.org/gutworth/six/get/1.10.0.zip",
        sha256 = "016c8313d1fe8eefe706d5c3f88ddc51bd78271ceef0b75e0a9b400b6a8998a9",
        strip_prefix = "gutworth-six-e5218c3f66a2",
        build_file = clean_dep("//bzl:six.BUILD"),
    )

    native.bind(
        name = "six",
        actual = "@six_archive//:six",
    )

def configure_toolchain():
    http_archive(
        name = "crosstool_ng_linux_x86_64_gcc_8.3.0",
        build_file = clean_dep("//toolchain:crosstool_ng/linux_x86_64.BUILD"),
        sha256 = "091f5732882a499c6b9fb5fcb895176d0c96e958236e16b61d1a9cafec4271ad",
        strip_prefix = "x86_64-unknown-linux-gnu",
        url = "https://github.com/plaidml/depot/raw/master/toolchain/gcc-8.3/x86_64-unknown-linux-gnu-20191010.tgz",
    )
