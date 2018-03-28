# Copyright Vertex.AI.

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "docsimages",
    srcs = glob(["*png"]),
    path = "docs"
)
genrule(
    name = "docs",
    srcs = [
        ":docsimages",
        "//:mddocs",
        "@plantuml_jar//file"
    ] + glob(
        ["**/*"],
    ),
    outs = ["docs.zip", "_build/html"],
    cmd = """
        set -x
        virtualenv --clear venv
        VIRTUAL_ENV_DISABLE_PROMPT=1
        if [ -e 'venv/bin/activate' ]; then
          . venv/bin/activate
        elif [ -e 'venv/Scripts/activate' ]; then
          . venv/Scripts/activate
        else
          echo 'Unable to enter virtual python environment'
          exit 1
        fi
        mkdir -p $(location _build/html)/docs
        cp -r $(locations :docsimages) $(location _build/html)/docs
        python -m pip install -r $(location requirements.txt)
        PLANTUMLSH="java -jar `pwd`/$(location @plantuml_jar//file)"  python -m sphinx -a -E -b html -c $$(dirname $(location conf.py)) $$(dirname $(location conf.py)) $(location _build/html)
        DOCSZIP=`pwd`/$(location docs.zip)
        pushd $(location _build/html)/../
        zip -q -r $${DOCSZIP} html
	popd
    """,
    local = 1,
)
