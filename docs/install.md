# Installation Instructions

PlaidML supports [Ubuntu\* Linux](#ubuntu-linux), [macOS\*](#macos) and
[Microsoft Windows\*](#windows) operating systems.

## Ubuntu Linux

If necessary, install Python's ``pip`` tool. OpenCL 1.2 or greater is also
required.

```
sudo add-apt-repository universe && sudo apt update
sudo apt install python3-pip
sudo apt install clinfo
```
Run ``clinfo``, and if it reports ``"Number of platforms" == 0``, you
can install a driver (GPU) or enable a CPU via one of these options:

* **Nvidia** -- For Nvidia GPUs, run:
```
sudo add-apt-repository ppa:graphics-drivers/ppa && sudo apt update
sudo apt install nvidia-modprobe nvidia-384 nvidia-opencl-icd-384 libcuda1-384
```

* **AMD** -- For AMD graphics cards, [download the AMDGPU PRO driver] and follow
the instructions provided by AMD for the chip.

* **Intel® Xeon® Processors OR Intel® Core™ Processors** -- In lieu of
installing specific drivers, you can [install nGraph with pip], or you can
[build the nGraph Library] with the cmake flag `-DNGRAPH_PLAIDML_ENABLE=TRUE`.


### Python

Although PlaidML can be run with Python2, we recommend Python3, as well as
judicious use of a [Virtualenv].  To create one just for using PlaidML:
```
python3 -m venv plaidml-venv
source plaidml-venv/bin/activate
```
### Keras\*

There are two ways to get Keras\* working on your system:

1. Isolate it to your `venv` as follows:
```
pip install -U plaidml-keras
```
2. Alternatively, install the PlaidML wheels system-wide with:
```
sudo -H pip install -U plaidml-keras
```

Finally, set up PlaidML to use a preferred computing device:
```
plaidml-setup
```
You can test the installation by running MobileNet in `plaidbench`. Remember to
use ``sudo -H`` if you're working outside of a virtual environment.
```
pip install plaidml-keras plaidbench
plaidbench keras mobilenet
```

You can adapt any Keras\* code by using the PlaidML backend instead of the
TensorFlow\*, CNTK\*, or Theano\* backend that you'd normally use; simply change
the Keras\* backend to ``plaidml.keras.backend``. You can do this by modifying
``~/.keras/keras.json`` so that the backend line reads ``"backend":
"plaidml.keras.backend"`` If this file does not exist, see the [Backend
instructions for Keras]. If you don't need anything special in your Keras\*
settings, you can set the ``~/.keras/keras.json`` as follows:

```json
{
   "epsilon": 1e-07,
   "floatx": "float32",
   "image_data_format": "channels_last",
   "backend": "plaidml.keras.backend"
}
```

Another option is to globally set the ``KERAS_BACKEND`` environment variable
to `plaidml.keras.backend`.

A monkey-patch technique involving ``plaidml.keras.install_backend()`` may still
work, but should be considered deprecated in favor of the above methods.

## macOS

A computer listed on [Apple's compatibility list] with support for OpenCL 1.2 is
required; those from 2011 and later usually fit this requirement.

### Python

Although PlaidML can be run with Python2, we recommend Python3, as well as
judicious use of a [Virtualenv].  To create one just for using PlaidML:
```
python3 -m venv plaidml-venv
source plaidml-venv/bin/activate
```
### Keras\*

To install PlaidML with Keras\*, run the following:
```
pip install -U plaidml-keras
```

Finally, set up PlaidML to use a preferred computing device:
```
plaidml-setup
```
PlaidML should now be installed! You can test the installation by running
MobileNet in [plaidbench].
```
pip install plaidml-keras plaidbench
plaidbench keras mobilenet
```

## Windows\*

These instructions assume Windows\* 10 without Python installed; adapt
accordingly.

1. First install [Chocolatey] by starting an Administrator PowerShell
   and running:
```
Set-ExecutionPolicy Bypass -Scope Process -Force; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
```
You'll likely need to reboot your shell at this point.

1. Install Python:
```
choco install -y python git vcredist2015
```
1. Switch to an unprivileged PowerShell to install and set up PlaidML with
Keras\*
```
pip install -U plaidml-keras
plaidml-setup
```

PlaidML should now be installed! You can test the installation by running
MobileNet in [plaidbench].

```
pip install plaidml-keras plaidbench
plaidbench keras mobilenet
```

[download the AMDGPU PRO driver]: http://support.amd.com/en-us/kb-articles/Pages/AMDGPU-PRO-Driver-for-Linux-Release-Notes.aspx
[Backend instructions for Keras]: https://keras.io/backend
[Apple's compatibility list]: https://support.apple.com/en-us/HT202823
[Intel® SDK for OpenCL™ Applications]: https://software.intel.com/en-us/intel-opencl
[install nGraph with pip]: https://github.com/NervanaSystems/ngraph/blob/master/README.md#quick-start
[build the nGraph Library]: https://ngraph.nervanasys.com/docs/latest/buildlb.html
[Virtualenv]: https://virtualenv.pypa.io/en/stable
[plaidbench]: https://github.com/plaidml/plaidbench
[Chocolatey]: https://chocolatey.org/
