## Installation Instructions

### Ubuntu Linux
If necessary, install Python's 'pip' tool.
```
sudo add-apt-repository universe && sudo apt update
sudo apt install python-pip
```
Make sure your system has OpenCL.
```
sudo apt install clinfo
clinfo
```
If clinfo reports "Number of platforms" == 0, you must install a driver.

If you have an NVIDIA graphics card:
```
sudo add-apt-repository ppa:graphics-drivers/ppa && sudo apt update
sudo apt install nvidia-modprobe nvidia-384 nvidia-opencl-icd-384 libcuda1-384
```
If you have an AMD card, [download the AMDGPU PRO driver and install](http://support.amd.com/en-us/kb-articles/Pages/AMDGPU-PRO-Driver-for-Linux-Release-Notes.aspx) according to AMD's instructions.

Best practices for python include judicious usage of [Virtualenv](https://virtualenv.pypa.io/en/stable/), and we certainly recommend creating one just for plaidml:
```
virtualenv plaidml-venv
source ./plaidml-venv/bin/activate
pip install -U plaidml-keras
```

Alternatively, install the PlaidML wheels system-wide:
```
sudo -H pip install -U plaidml-keras
```

Next, setup PlaidML to use your preferred computing device:
```
plaidml-setup
```

You can test your installation by running MobileNet in [plaidbench](https://github.com/plaidml/plaidbench):
(Remember to use sudo -H if you're not using a Virtualenv)
```
pip install plaidml-keras plaidbench
plaidbench keras mobilenet
```

You can adapt any Keras code by using the PlaidML backend instead of the TensorFlow, CNTK, or Theano backend that you 
normally use.

Simply insert this code **BEFORE you `import keras`**:
```
# Install the plaidml backend
import plaidml.keras
plaidml.keras.install_backend()
```

### macOS

You need a computer listed on Apple's [compatibility list](https://support.apple.com/en-us/HT202823) as having OpenCL 1.2 support (most machines 2011 and later).

Best practices for python include judicious usage of [Virtualenv](https://virtualenv.pypa.io/en/stable/), and we certainly recommend creating one just for plaidml:
```
virtualenv plaidml-venv
. plaidml-venv/bin/activate
```

Install PlaidML with Keras:
```
pip install -U plaidml-keras
```

Next, setup PlaidML to use your preferred computing device:
```
plaidml-setup
```

PlaidML should now be installed! You can test your installation by running MobileNet in [plaidbench](https://github.com/plaidml/plaidbench):
```
pip install plaidml-keras plaidbench
plaidbench keras mobilenet
```

### Windows

These instructions assume Windows 10 without python installed; adapt accordingly. First install Chocolatey by starting an Administrator PowerShell and running
```
Set-ExecutionPolicy Bypass -Scope Process -Force; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
```
You likely will need to reboot your shell; then you can install Python:
```
choco install -y python git vcredist2015
```

You can now switch to an unpriviledged PowerShell to install and set up PlaidML with Keras:
```
pip install -U plaidml-keras
plaidml-setup
```

PlaidML should now be installed! You can test your installation by running MobileNet in [plaidbench](https://github.com/plaidml/plaidbench):
```
pip install plaidml-keras plaidbench
plaidbench keras mobilenet
```
