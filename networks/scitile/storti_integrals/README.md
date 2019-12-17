
Instructions
-------------------------------------------------------------------
How to run the EDSL examples
-------------------------------------------------------------------

1. Create a python environment 

```
virtualenv plaidml-venv
source plaidml-venv/bin/activate
```

2. Download and install plaidml2

  * Go to: https://buildkite.com/plaidml/plaidml-plaidml/builds?branch=master
  * Click on the platform that you are using 
  * Click on artifacts 
  * download the plaidml2 wheel for your platform

```
pip install /path/to/the/downloaded/wheel
```

4. Run plaidml2 setup

```
plaidml2-setup
```

4. Grab the EDSL example code that you would like to run

```
wget https://raw.githubusercontent.com/plaidml/plaidml/storti_integrals/networks/scitile/storti_integrals/op.py
wget https://raw.githubusercontent.com/plaidml/plaidml/storti_integrals/networks/scitile/storti_integrals/torus.py
```

5. run the example 

```
python torus.py 
```

6. depending on which example you choose you run you might need additinal dependencies like matplotlib

```
pip install matplotlib
```