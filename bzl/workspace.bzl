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
        urls = [
            "https://mirror.bazel.build/github.com/plaidml/bazel-latex/archive/b6375d9df2952548c3371c0c865710655e8b1cc1.zip",
            "https://github.com/plaidml/bazel-latex/archive/b6375d9df2952548c3371c0c865710655e8b1cc1.zip",
        ],
        strip_prefix = "bazel-latex-b6375d9df2952548c3371c0c865710655e8b1cc1",
    )

    http_archive(
        name = "boost",
        sha256 = "bd0df411efd9a585e5a2212275f8762079fed8842264954675a4fddc46cfcf60",
        urls = [
            "https://mirror.bazel.build/boostorg.jfrog.io/artifactory/main/release/1.66.0/source/boost_1_66_0.tar.gz",
            "https://boostorg.jfrog.io/artifactory/main/release/1.66.0/source/boost_1_66_0.tar.gz",
        ],
        strip_prefix = "boost_1_66_0",
        build_file = clean_dep("//bzl:boost.BUILD"),
    )

    http_archive(
        name = "com_github_google_benchmark",
        sha256 = "3c6a165b6ecc948967a1ead710d4a181d7b0fbcaa183ef7ea84604994966221a",
        urls = [
            "https://mirror.bazel.build/github.com/google/benchmark/archive/v1.5.0.tar.gz",
            "https://github.com/google/benchmark/archive/v1.5.0.tar.gz",
        ],
        strip_prefix = "benchmark-1.5.0",
    )

    http_archive(
        name = "easylogging",
        sha256 = "4b1aebe19e383349c6e438aac357eccfabb0ce34430e872508ed8ee0d1629e0f",
        urls = [
            "https://mirror.bazel.build/github.com/muflihun/easyloggingpp/releases/download/v9.95.0/easyloggingpp_v9.95.0.tar.gz",
            "https://github.com/muflihun/easyloggingpp/releases/download/v9.95.0/easyloggingpp_v9.95.0.tar.gz",
        ],
        build_file = clean_dep("//bzl:easylogging.BUILD"),
    )

    http_archive(
        name = "gflags",
        sha256 = "7d17922978692175c67ef5786a014df44bfbfe3b48b30937cca1413d4ff65f75",
        urls = [
            "https://mirror.bazel.build/github.com/gflags/gflags/archive/e292e0452fcfd5a8ae055b59052fc041cbab4abf.zip",
            "https://github.com/gflags/gflags/archive/e292e0452fcfd5a8ae055b59052fc041cbab4abf.zip",
        ],
        strip_prefix = "gflags-e292e0452fcfd5a8ae055b59052fc041cbab4abf",
    )

    http_archive(
        name = "gmock",
        sha256 = "58a6f4277ca2bc8565222b3bbd58a177609e9c488e8a72649359ba51450db7d8",
        urls = [
            "https://mirror.bazel.build/github.com/google/googletest/archive/release-1.8.0.tar.gz",
            "https://github.com/google/googletest/archive/release-1.8.0.tar.gz",
        ],
        strip_prefix = "googletest-release-1.8.0",
        build_file = clean_dep("//bzl:gmock.BUILD"),
    )

    http_archive(
        name = "half",
        sha256 = "9e5ddb4b43abeafe190e780b5b606b081acb511e6edd4ef6fbe5de863a4affaf",
        urls = [
            "https://mirror.bazel.build/sourceforge.net/projects/half/files/half/1.11.0/half-1.11.0.zip/download",
            "https://sourceforge.net/projects/half/files/half/1.11.0/half-1.11.0.zip/download",
        ],
        strip_prefix = "half-1.11.0",
        build_file = clean_dep("//bzl:half.BUILD"),
    )

    http_archive(
        name = "jsoncpp",
        sha256 = "2099839a06c867a8b3abf81b3eb82dc0ed67207fd0d2b940b6cf2efef66fe7d8",
        urls = [
            "https://mirror.bazel.build/github.com/open-source-parsers/jsoncpp/archive/11086dd6a7eba04289944367ca82cea71299ed70.zip",
            "https://github.com/open-source-parsers/jsoncpp/archive/11086dd6a7eba04289944367ca82cea71299ed70.zip",
        ],
        strip_prefix = "jsoncpp-11086dd6a7eba04289944367ca82cea71299ed70",
        build_file = clean_dep("//bzl:jsoncpp.BUILD"),
    )

    http_archive(
        name = "jsonnet",
        sha256 = "e9f7095dd2a383001188aa622edaf82059732e11d74f8d0bfdfa84f2682dd547",
        urls = [
            "https://mirror.bazel.build/github.com/google/jsonnet/archive/v0.13.0.zip",
            "https://github.com/google/jsonnet/archive/v0.13.0.zip",
        ],
        strip_prefix = "jsonnet-0.13.0",
    )

    http_archive(
        name = "io_bazel_rules_jsonnet",
        sha256 = "d05d719c4738e8aac5f13b32f745ff4832b9638ecc89ddcb6e36c379a1ada025",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_jsonnet/archive/0.1.0.zip",
            "https://github.com/bazelbuild/rules_jsonnet/archive/0.1.0.zip",
        ],
        strip_prefix = "rules_jsonnet-0.1.0",
    )

    http_archive(
        name = "minizip",
        sha256 = "47355898c601ee005b171033016829b759087b559d7cf17114c71edf8aaf88c0",
        urls = [
            "https://mirror.bazel.build/github.com/nmoinvaz/minizip/archive/36089398a362a117105ebfcb3751a269c70ab3b7.zip",
            "https://github.com/nmoinvaz/minizip/archive/36089398a362a117105ebfcb3751a269c70ab3b7.zip",
        ],
        strip_prefix = "minizip-36089398a362a117105ebfcb3751a269c70ab3b7",
        build_file = clean_dep("//bzl:minizip.BUILD"),
    )

    http_archive(
        name = "opencl_headers",
        sha256 = "b2b813dd88a7c39eb396afc153070f8f262504a7f956505b2049e223cfc2229b",
        urls = [
            "https://mirror.bazel.build/github.com/KhronosGroup/OpenCL-Headers/archive/f039db6764d52388658ef15c30b2237bbda49803.zip",
            "https://github.com/KhronosGroup/OpenCL-Headers/archive/f039db6764d52388658ef15c30b2237bbda49803.zip",
        ],
        strip_prefix = "OpenCL-Headers-f039db6764d52388658ef15c30b2237bbda49803",
        build_file = clean_dep("//bzl:opencl_headers.BUILD"),
    )

    http_archive(
        name = "cm_headers",
        sha256 = "4549496e3742ade2ff13e804654cb4ee7ddabb3b95dbc1fdeb9ca22141f317d5",
        urls = [
            "https://mirror.bazel.build/github.com/intel/cm-compiler/releases/download/Release_20190717/Linux_C_for_Metal_Development_Package_20190717.zip",
            "https://github.com/intel/cm-compiler/releases/download/Release_20190717/Linux_C_for_Metal_Development_Package_20190717.zip",
        ],
        strip_prefix = "Linux_C_for_Metal_Development_Package_20190717",
        build_file = clean_dep("//bzl:cm_headers.BUILD"),
    )

    http_archive(
        name = "libva",
        sha256 = "3aa89cd369a506ac4dbe5de7c0ef5da4f3d220bf986403f02fa1f6f702af6878",
        urls = [
            "https://mirror.bazel.build/github.com/intel/libva/releases/download/2.5.0/libva-2.5.0.tar.bz2",
            "https://github.com/intel/libva/releases/download/2.5.0/libva-2.5.0.tar.bz2",
        ],
        strip_prefix = "libva-2.5.0",
        build_file = clean_dep("//bzl:libva.BUILD"),
    )

    http_archive(
        name = "pybind11",
        sha256 = "b69e83658513215b8d1443544d0549b7d231b9f201f6fc787a2b2218b408181e",
        urls = [
            "https://mirror.bazel.build/github.com/pybind/pybind11/archive/v2.2.4.tar.gz",
            "https://github.com/pybind/pybind11/archive/v2.2.4.tar.gz",
        ],
        strip_prefix = "pybind11-2.2.4",
        build_file = clean_dep("//bzl:pybind11.BUILD"),
    )

    http_archive(
        name = "rules_pkg",
        sha256 = "e46b4f5aa71d1037c7c8142e2fedb503127af4bbd9dbde4a742d119749f68a3f",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_pkg/archive/cb54c427343aa48c32e3c09ddcc8f6316cdbd5a6.tar.gz",
            "https://github.com/bazelbuild/rules_pkg/archive/cb54c427343aa48c32e3c09ddcc8f6316cdbd5a6.tar.gz",
        ],
        strip_prefix = "rules_pkg-cb54c427343aa48c32e3c09ddcc8f6316cdbd5a6/pkg",
    )

    http_archive(
        name = "rules_python",
        sha256 = "b5bab4c47e863e0fbb77df4a40c45ca85f98f5a2826939181585644c9f31b97b",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_python/archive/9d68f24659e8ce8b736590ba1e4418af06ec2552.tar.gz",
            "https://github.com/bazelbuild/rules_python/archive/9d68f24659e8ce8b736590ba1e4418af06ec2552.tar.gz",
        ],
        strip_prefix = "rules_python-9d68f24659e8ce8b736590ba1e4418af06ec2552",
    )

    http_archive(
        name = "zlib",
        sha256 = "c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1",
        urls = [
            "https://mirror.bazel.build/zlib.net/zlib-1.2.11.tar.gz",
            "https://zlib.net/zlib-1.2.11.tar.gz",
        ],
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
        sha256 = "451ec9d30f0890bf3081aa3d0d264942a6dea8f9d29c17bececc8465a10a832b",
        urls = [
            "https://mirror.bazel.build/github.com/hfp/libxsmm/archive/1.12.1.zip",
            "https://github.com/hfp/libxsmm/archive/1.12.1.zip",
        ],
        strip_prefix = "libxsmm-1.12.1",
        build_file = clean_dep("//bzl:xsmm.BUILD"),
    )

    http_archive(
        name = "tbb",
        sha256 = "3bb395989ce4701fc5d486ab8db3d639e9f86111a761de358d8e02bb6ed1e076",
        urls = [
            "https://mirror.bazel.build/github.com/intel/tbb/archive/tbb_2019.zip",
            "https://github.com/intel/tbb/archive/tbb_2019.zip",
        ],
        strip_prefix = "oneTBB-tbb_2019",
        build_file = clean_dep("//vendor/tbb:tbb.BUILD"),
    )

    http_archive(
        name = "llvm-project",
        sha256 = "73682f2b78c1c46621afb69b850e50c4d787f9c77fb3b53ac50fc42ffbac0493",
        urls = [
            "https://mirror.bazel.build/github.com/llvm/llvm-project/archive/a21beccea2020f950845cbb68db663d0737e174c.tar.gz",
            "https://github.com/llvm/llvm-project/archive/a21beccea2020f950845cbb68db663d0737e174c.tar.gz",
        ],
        strip_prefix = "llvm-project-a21beccea2020f950845cbb68db663d0737e174c",
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
        sha256 = "2ee9dcec820352671eb83e081295ba43f7a4157181dad549024d7070d079cf65",
        urls = [
            "https://mirror.bazel.build/github.com/protocolbuffers/protobuf/archive/v3.9.0.tar.gz",
            "https://github.com/protocolbuffers/protobuf/archive/v3.9.0.tar.gz",
        ],
        strip_prefix = "protobuf-3.9.0",
        build_file = clean_dep("//bzl:protobuf.BUILD"),
    )

    http_archive(
        name = "rules_cc",
        sha256 = "29daf0159f0cf552fcff60b49d8bcd4f08f08506d2da6e41b07058ec50cfeaec",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_cc/archive/b7fe9697c0c76ab2fd431a891dbb9a6a32ed7c3e.tar.gz",
            "https://github.com/bazelbuild/rules_cc/archive/b7fe9697c0c76ab2fd431a891dbb9a6a32ed7c3e.tar.gz",
        ],
        strip_prefix = "rules_cc-b7fe9697c0c76ab2fd431a891dbb9a6a32ed7c3e",
    )

    http_archive(
        name = "rules_java",
        sha256 = "f5a3e477e579231fca27bf202bb0e8fbe4fc6339d63b38ccb87c2760b533d1c3",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_java/archive/981f06c3d2bd10225e85209904090eb7b5fb26bd.tar.gz",
            "https://github.com/bazelbuild/rules_java/archive/981f06c3d2bd10225e85209904090eb7b5fb26bd.tar.gz",
        ],
        strip_prefix = "rules_java-981f06c3d2bd10225e85209904090eb7b5fb26bd",
    )

    http_archive(
        name = "rules_proto",
        sha256 = "88b0a90433866b44bb4450d4c30bc5738b8c4f9c9ba14e9661deb123f56a833d",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_proto/archive/b0cc14be5da05168b01db282fe93bdf17aa2b9f4.tar.gz",
            "https://github.com/bazelbuild/rules_proto/archive/b0cc14be5da05168b01db282fe93bdf17aa2b9f4.tar.gz",
        ],
        strip_prefix = "rules_proto-b0cc14be5da05168b01db282fe93bdf17aa2b9f4",
    )

    http_archive(
        name = "six_archive",
        sha256 = "ed17446c954bdf4dd7a705df85e9aab1338fad3ea40e7df4beda76c6e73c71b1",
        urls = [
            "https://mirror.bazel.build/github.com/benjaminp/six/archive/1.10.0.zip",
            "https://github.com/benjaminp/six/archive/1.10.0.zip",
        ],
        strip_prefix = "six-1.10.0",
        build_file = clean_dep("//bzl:six.BUILD"),
    )

    native.bind(
        name = "six",
        actual = "@six_archive//:six",
    )

def configure_toolchain():
    http_archive(
        name = "crosstool_ng_linux_x86_64_gcc_8.3.0",
        sha256 = "52bfebd923059613b8b592122005bc6cd7280a7a53987007ed75d50a6ae89925",
        urls = [
            "https://mirror.bazel.build/github.com/plaidml/depot/raw/master/toolchain/gcc-8.3/x86_64-unknown-linux-gnu-20191010.tgz",
            "https://github.com/plaidml/depot/raw/master/toolchain/gcc-8.3/x86_64-unknown-linux-gnu-20191010.tgz",
        ],
        strip_prefix = "x86_64-unknown-linux-gnu",
        build_file = clean_dep("//toolchain:crosstool_ng/linux_x86_64.BUILD"),
    )
