# Guide

## Toolchain Creation

Install crosstool-ng:

```
mkdir ~/src/vendor
cd ~/src/vendor
wget http://crosstool-ng.org/download/crosstool-ng/crosstool-ng-1.23.0.tar.xz
tar xvf crosstool-ng-1.23.0.tar.xz
cd crosstool-ng-1.23.0
./configure --prefix=/usr/local/
make -j16
sudo make install
```

Prepare a configuration:

```
mkdir ~/crosstool
cd ~/crosstool
mkdir linux_arm_32v7
cd linux_arm_32v7
ct-ng menuconfig
```

Select toolchain options.

```
ct-ng build
```

After about 15 minutes, the toolchain will land in a directory under `~/x-tools`.

Now we need to package it up and deploy it to someplace bazel can pull from:

```
cd ~/x-tools
tar cvzf arm-unknown-linux-gnueabihf.tgz arm-unknown-linux-gnueabihf/
gsutil cp arm-unknown-linux-gnueabihf.tgz gs://vertexai-depot/
shasum -a 256 *.tgz
```

You'll want the SHA256 digest when adding an entry to `workspace.bzl`.
