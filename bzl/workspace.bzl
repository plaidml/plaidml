load("//vendor/cuda:configure.bzl", "configure_cuda")
load("//bzl:conda_repo.bzl", "conda_repo")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")

def plaidml_workspace():
    http_archive(
        name = "bazel_skylib",
        url = "https://github.com/bazelbuild/bazel-skylib/archive/0.8.0.tar.gz",
        sha256 = "2ea8a5ed2b448baf4a6855d3ce049c4c452a6470b1efd1504fdb7c1c134d220a",
        strip_prefix = "bazel-skylib-0.8.0",
    )

    http_archive(
        name = "boost",
        url = "https://storage.googleapis.com/vertexai-depot/boost_1_66_0.tar.gz",
        sha256 = "bd0df411efd9a585e5a2212275f8762079fed8842264954675a4fddc46cfcf60",
        strip_prefix = "boost_1_66_0",
        build_file = str(Label("//bzl:boost.BUILD")),
    )

    http_archive(
        name = "easylogging",
        url = "https://github.com/muflihun/easyloggingpp/releases/download/v9.95.0/easyloggingpp_v9.95.0.tar.gz",
        sha256 = "4b1aebe19e383349c6e438aac357eccfabb0ce34430e872508ed8ee0d1629e0f",
        build_file = str(Label("//bzl:easylogging.BUILD")),
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
        build_file = str(Label("//bzl:gmock.BUILD")),
    )

    http_archive(
        name = "half",
        url = "https://storage.googleapis.com/external_build_repo/half-1.11.0.zip",
        sha256 = "9e5ddb4b43abeafe190e780b5b606b081acb511e6edd4ef6fbe5de863a4affaf",
        strip_prefix = "half-1.11.0",
        build_file = str(Label("//bzl:half.BUILD")),
    )

    http_archive(
        name = "jsoncpp",
        url = "https://github.com/open-source-parsers/jsoncpp/archive/11086dd6a7eba04289944367ca82cea71299ed70.zip",
        sha256 = "2099839a06c867a8b3abf81b3eb82dc0ed67207fd0d2b940b6cf2efef66fe7d8",
        strip_prefix = "jsoncpp-11086dd6a7eba04289944367ca82cea71299ed70",
        build_file = str(Label("//bzl:jsoncpp.BUILD")),
    )

    http_archive(
        name = "jsonnet",
        url = "https://github.com/google/jsonnet/archive/v0.12.1.tar.gz",
        sha256 = "257c6de988f746cc90486d9d0fbd49826832b7a2f0dbdb60a515cc8a2596c950",
        strip_prefix = "jsonnet-0.12.1",
    )

    http_archive(
        name = "minizip",
        url = "https://github.com/nmoinvaz/minizip/archive/36089398a362a117105ebfcb3751a269c70ab3b7.zip",
        sha256 = "c47b06ad7ef10d01a8d415b1b8dfb3691dad6ed41b38756fbf8fd6c074480d0f",
        strip_prefix = "minizip-36089398a362a117105ebfcb3751a269c70ab3b7",
        build_file = str(Label("//bzl:minizip.BUILD")),
    )

    http_archive(
        name = "opencl_headers",
        url = "https://github.com/KhronosGroup/OpenCL-Headers/archive/f039db6764d52388658ef15c30b2237bbda49803.zip",
        sha256 = "b2b813dd88a7c39eb396afc153070f8f262504a7f956505b2049e223cfc2229b",
        strip_prefix = "OpenCL-Headers-f039db6764d52388658ef15c30b2237bbda49803",
        build_file = str(Label("//bzl:opencl_headers.BUILD")),
    )

    http_archive(
        name = "opencl_icd_loader",
        url = "https://github.com/KhronosGroup/OpenCL-ICD-Loader/archive/6849f617e991e8a46eebf746df43032175f263b3.zip",
        sha256 = "1c82be071237ccce36753be9331de3c6b8d4f461d931b40bc070d6c0fb80ff83",
        strip_prefix = "OpenCL-ICD-Loader-6849f617e991e8a46eebf746df43032175f263b3",
        build_file = str(Label("//bzl:opencl_icd_loader.BUILD")),
    )

    http_archive(
        name = "zlib",
        url = "https://storage.googleapis.com/external_build_repo/zlib-1.2.8.tar.gz",
        sha256 = "36658cb768a54c1d4dec43c3116c27ed893e88b02ecfcb44f2166f9c0b7f2a0d",
        build_file = str(Label("//bzl:zlib.BUILD")),
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
        specs = {
            "env": str(Label("//conda:plaidml.yml")),
        },
    )

    conda_repo(
        name = "com_intel_plaidml_conda_sphinx",
        specs = {
            "env": str(Label("//conda:sphinx.yml")),
        },
    )

    http_archive(
        name = "mlir",
        url = "https://github.com/tensorflow/mlir/archive/2f17f4dfe8daf10789bcd7cf16e4112bd6bd2244.zip",
        sha256 = "fdf46ba9e43403731144db3fc0037905d01a3045e2f9da40dea9c44c7eb93fc2",
        strip_prefix = "mlir-2f17f4dfe8daf10789bcd7cf16e4112bd6bd2244",
        build_file = str(Label("//bzl:mlir.BUILD")),
    )

def configure_protobuf():
    http_archive(
        name = "com_google_protobuf",
        url = "https://github.com/protocolbuffers/protobuf/archive/v3.6.1.2.tar.gz",
        sha256 = "2244b0308846bb22b4ff0bcc675e99290ff9f1115553ae9671eba1030af31bc0",
        strip_prefix = "protobuf-3.6.1.2",
        build_file = str(Label("//bzl:protobuf.BUILD")),
    )

    http_archive(
        name = "six_archive",
        url = "https://bitbucket.org/gutworth/six/get/1.10.0.zip",
        sha256 = "016c8313d1fe8eefe706d5c3f88ddc51bd78271ceef0b75e0a9b400b6a8998a9",
        strip_prefix = "gutworth-six-e5218c3f66a2",
        build_file = str(Label("//bzl:six.BUILD")),
    )

    native.bind(
        name = "six",
        actual = "@six_archive//:six",
    )

def configure_llvm():
    http_archive(
        name = "llvm",
        url = "http://releases.llvm.org/8.0.8/llvm-8.0.0.src.tar.xz",
        sha256 = "8872be1b12c61450cacc82b3d153eab02be2546ef34fa3580ed14137bb26224c",
        strip_prefix = "llvm-8.0.0.src",
        build_file = str(Label("//vendor/llvm:llvm.BUILD")),
    )
