
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

3. Grab the EDSL example code that you would like to run

```
wget https://raw.githubusercontent.com/plaidml/plaidml/master/networks/scitile/uw_toroidal_shell/uw_toroidal_shell.py
```

4. run the example 

```
python uw_toroidal_shell.py 
```
