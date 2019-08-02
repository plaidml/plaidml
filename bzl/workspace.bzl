load("//bzl:conda_repo.bzl", "conda_repo")
load("//bzl:xsmm_repo.bzl", "xsmm_repo")
load("//vendor/cuda:configure.bzl", "configure_cuda")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")
load("//bzl:dev_repo.bzl", "dev_http_archive")

def plaidml_workspace():
    configure_toolchain()

    http_archive(
        name = "boost",
        url = "https://storage.googleapis.com/vertexai-depot/boost_1_66_0.tar.gz",
        sha256 = "bd0df411efd9a585e5a2212275f8762079fed8842264954675a4fddc46cfcf60",
        strip_prefix = "boost_1_66_0",
        build_file = Label("//bzl:boost.BUILD"),
    )

    http_archive(
        name = "easylogging",
        url = "https://github.com/muflihun/easyloggingpp/releases/download/v9.95.0/easyloggingpp_v9.95.0.tar.gz",
        sha256 = "4b1aebe19e383349c6e438aac357eccfabb0ce34430e872508ed8ee0d1629e0f",
        build_file = Label("//bzl:easylogging.BUILD"),
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
        build_file = Label("//bzl:gmock.BUILD"),
    )

    http_archive(
        name = "half",
        url = "https://storage.googleapis.com/external_build_repo/half-1.11.0.zip",
        sha256 = "9e5ddb4b43abeafe190e780b5b606b081acb511e6edd4ef6fbe5de863a4affaf",
        strip_prefix = "half-1.11.0",
        build_file = Label("//bzl:half.BUILD"),
    )

    http_archive(
        name = "jsoncpp",
        url = "https://github.com/open-source-parsers/jsoncpp/archive/11086dd6a7eba04289944367ca82cea71299ed70.zip",
        sha256 = "2099839a06c867a8b3abf81b3eb82dc0ed67207fd0d2b940b6cf2efef66fe7d8",
        strip_prefix = "jsoncpp-11086dd6a7eba04289944367ca82cea71299ed70",
        build_file = Label("//bzl:jsoncpp.BUILD"),
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
        build_file = Label("//bzl:minizip.BUILD"),
    )

    http_archive(
        name = "opencl_headers",
        url = "https://github.com/KhronosGroup/OpenCL-Headers/archive/f039db6764d52388658ef15c30b2237bbda49803.zip",
        sha256 = "b2b813dd88a7c39eb396afc153070f8f262504a7f956505b2049e223cfc2229b",
        strip_prefix = "OpenCL-Headers-f039db6764d52388658ef15c30b2237bbda49803",
        build_file = Label("//bzl:opencl_headers.BUILD"),
    )

    http_archive(
        name = "pybind11",
        url = "https://github.com/pybind/pybind11/archive/v2.2.4.tar.gz",
        sha256 = "b69e83658513215b8d1443544d0549b7d231b9f201f6fc787a2b2218b408181e",
        strip_prefix = "pybind11-2.2.4",
        build_file = Label("//bzl:pybind11.BUILD"),
    )

    http_archive(
        name = "zlib",
        url = "https://storage.googleapis.com/external_build_repo/zlib-1.2.8.tar.gz",
        sha256 = "36658cb768a54c1d4dec43c3116c27ed893e88b02ecfcb44f2166f9c0b7f2a0d",
        build_file = Label("//bzl:zlib.BUILD"),
    )

    http_file(
        name = "plantuml_jar",
        urls = ["https://storage.googleapis.com/vertexai-depot/plantuml.jar"],
        sha256 = "26d60e43c14106a3d220e33c2b2e073b2bce40b433ad3e5fa13c747f58e67ab6",
    )

    configure_protobuf()
    configure_cuda(name = "cuda")

    conda_repo(
        name = "com_intel_plaidml_conda",
        env = Label("//conda:plaidml.yml"),
    )

    conda_repo(
        name = "com_intel_plaidml_conda_analysis",
        env = "@com_intel_plaidml//conda:analysis.yml",
    )

    conda_repo(
        name = "com_intel_plaidml_conda_buildkite",
        env = "@com_intel_plaidml//conda:buildkite.yml",
    )

    conda_repo(
        name = "com_intel_plaidml_conda_ml_gpu",
        env = "@com_intel_plaidml//conda:ml_gpu.yml",
    )

    conda_repo(
        name = "com_intel_plaidml_conda_pytorch",
        env = Label("//conda:pytorch.yml"),
        build_file = Label("//conda:pytorch.BUILD"),
    )

    conda_repo(
        name = "com_intel_plaidml_conda_sphinx",
        env = Label("//conda:sphinx.yml"),
    )

    conda_repo(
        name = "com_intel_plaidml_conda_tools_unix",
        env = Label("//conda:tools-unix.yml"),
        build_file = Label("//conda:tools-unix.BUILD"),
    )

    conda_repo(
        name = "com_intel_plaidml_conda_tools_windows",
        env = Label("//conda:tools-windows.yml"),
        build_file = Label("//conda:tools-windows.BUILD"),
    )

    xsmm_repo(
        name = "xsmm",
        url = "https://github.com/hfp/libxsmm/archive/1.12.1.zip",
        sha256 = "451ec9d30f0890bf3081aa3d0d264942a6dea8f9d29c17bececc8465a10a832b",
        stripPrefix = "libxsmm-1.12.1",
        build_file = Label("//bzl:xsmm.BUILD"),
    )

    dev_http_archive(
        name = "llvm",
        url = "https://github.com/llvm-mirror/llvm/archive/b7d166cebcf619a3691eed3f994384aab3d80fa6.zip",
        sha256 = "61bc88374568cb0fc732432ec56bbab0e05c44b8f70f1f1868665204e9de2dba",
        strip_prefix = "llvm-b7d166cebcf619a3691eed3f994384aab3d80fa6",
        build_file = Label("//vendor/llvm:llvm.BUILD"),
    )

    dev_http_archive(
        name = "mlir",
        url = "https://github.com/plaidml/mlir/archive/4fcc9e1d9f8797a26512de1c640e46d24e07831c.zip",
        sha256 = "73cc16075b4190e739ada30e1338765590cfeb84588d057345de597ad3e36e48",
        strip_prefix = "mlir-4fcc9e1d9f8797a26512de1c640e46d24e07831c",
        build_file = Label("//vendor/mlir:mlir.BUILD"),
    )

def configure_protobuf():
    http_archive(
        name = "com_google_protobuf",
        url = "https://github.com/protocolbuffers/protobuf/archive/v3.9.0.tar.gz",
        sha256 = "2ee9dcec820352671eb83e081295ba43f7a4157181dad549024d7070d079cf65",
        strip_prefix = "protobuf-3.9.0",
        build_file = Label("//bzl:protobuf.BUILD"),
    )

    http_archive(
        name = "rules_cc",
        sha256 = "29daf0159f0cf552fcff60b49d8bcd4f08f08506d2da6e41b07058ec50cfeaec",
        strip_prefix = "rules_cc-b7fe9697c0c76ab2fd431a891dbb9a6a32ed7c3e",
        urls = ["https://github.com/bazelbuild/rules_cc/archive/b7fe9697c0c76ab2fd431a891dbb9a6a32ed7c3e.tar.gz"],
    )

    http_archive(
        name = "rules_java",
        sha256 = "f5a3e477e579231fca27bf202bb0e8fbe4fc6339d63b38ccb87c2760b533d1c3",
        strip_prefix = "rules_java-981f06c3d2bd10225e85209904090eb7b5fb26bd",
        urls = ["https://github.com/bazelbuild/rules_java/archive/981f06c3d2bd10225e85209904090eb7b5fb26bd.tar.gz"],
    )

    http_archive(
        name = "rules_proto",
        sha256 = "88b0a90433866b44bb4450d4c30bc5738b8c4f9c9ba14e9661deb123f56a833d",
        strip_prefix = "rules_proto-b0cc14be5da05168b01db282fe93bdf17aa2b9f4",
        urls = ["https://github.com/bazelbuild/rules_proto/archive/b0cc14be5da05168b01db282fe93bdf17aa2b9f4.tar.gz"],
    )

    http_archive(
        name = "six_archive",
        url = "https://bitbucket.org/gutworth/six/get/1.10.0.zip",
        sha256 = "016c8313d1fe8eefe706d5c3f88ddc51bd78271ceef0b75e0a9b400b6a8998a9",
        strip_prefix = "gutworth-six-e5218c3f66a2",
        build_file = Label("//bzl:six.BUILD"),
    )

    native.bind(
        name = "six",
        actual = "@six_archive//:six",
    )

def configure_toolchain():
    http_archive(
        name = "crosstool_ng_linux_x86_64_gcc_5.4.0",
        build_file = Label("//toolchain:crosstool_ng/linux_x86_64.BUILD"),
        sha256 = "dfbf72d78bfe876b2864f51ac740a54e5fd12e2b4a86c10514fb7accaa9640e6",
        strip_prefix = "x86_64-unknown-linux-gnu",
        url = "https://storage.googleapis.com/vertexai-depot/toolchain/gcc-5.4.0/x86_64-unknown-linux-gnu-20190419.tgz",
    )

    http_archive(
        name = "crosstool_ng_linux_x86_64_gcc_4.9.4",
        build_file = Label("//toolchain:crosstool_ng/linux_x86_64.BUILD"),
        sha256 = "28fc19c39683c3c1065058ea525eb9fbb20095249e69971215d9451184ab006f",
        strip_prefix = "x86_64-unknown-linux-gnu",
        url = "https://storage.googleapis.com/vertexai-depot/toolchain/gcc-4.9.4/x86_64-unknown-linux-gnu-20190617.tgz",
    )
