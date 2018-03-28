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
git clone https://github.com/plaidml/plaidbench.git
cd plaidbench
pip install -r requirements.txt
python plaidbench.py mobilenet
```

You can adapt any Keras code by using the PlaidML backend instead of the TensorFlow, CNTK, or Theano backend that you 
normally use.

Simply insert this code **BEFORE you `import keras`**:
```
# Install the plaidml backend
import plaidml.keras
plaidml.keras.install_backend()
```


