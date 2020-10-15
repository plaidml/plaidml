load("//vendor/bazel:repo.bzl", "http_archive")

# Sanitize a dependency so that it works correctly from code that includes it as a submodule.
def clean_dep(dep):
    return str(Label(dep))

def plaidml_tf_workspace():
    http_archive(
        name = "io_bazel_rules_closure",
        sha256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
        strip_prefix = "rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
        url = "https://github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",  # 2019-06-13
    )

    http_archive(
        name = "org_tensorflow",
        url = "https://github.com/tensorflow/tensorflow/archive/ffe4c23e975cc410dcbe99236f5ae2108ada2360.zip",
        sha256 = "051ba1c6bbb14348b2a80e78364fb7dbe06fc09a571a14077e8d033030546aba",
        strip_prefix = "tensorflow-ffe4c23e975cc410dcbe99236f5ae2108ada2360",
        link_files = {
            clean_dep("//vendor/tensorflow:third_party/py/python_configure.bzl"): "third_party/py/python_configure.bzl",
        },
    )

    http_archive(
        name = "tfhub_i3d_kinetics_400",
        url = "https://storage.googleapis.com/tfhub-modules/deepmind/i3d-kinetics-400/1.tar.gz",
        sha256 = "bafe29bb4528badad428207d8fe86ca2a998b5b1386b82e42175339f12ea2ff5",
        build_file = clean_dep("//vendor/tensorflow:tfhub_i3d_kinetics_400.BUILD"),
    )

    http_archive(
        name = "kinetics-i3d",
        url = "https://github.com/deepmind/kinetics-i3d/archive/efebe2eb948cb8a3d2601b6a7ee1af9986a4aedf.zip",
        sha256 = "4c7d9a32390a7b49865444beae96278c264b5eab8e111e63b8e20753e3ea3dde",
        strip_prefix = "kinetics-i3d-efebe2eb948cb8a3d2601b6a7ee1af9986a4aedf",
        build_file = clean_dep("//vendor/tensorflow:kinetics-i3d.BUILD"),
    )
