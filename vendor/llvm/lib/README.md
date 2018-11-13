# LLVM

First build the static libs:
`./t2 build @llvm//:static.so`

## macos_x86_64

```
cp private/bazel-bin/external/llvm/libsupport.lo public/plaidml/vendor/llvm/lib/macos_x86_64/
cp private/bazel-bin/external/llvm/liblib.lo public/plaidml/vendor/llvm/lib/macos_x86_64/
cp private/bazel-bin/external/llvm/libtargets.lo public/plaidml/vendor/llvm/lib/macos_x86_64/
```
