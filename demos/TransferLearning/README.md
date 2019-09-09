# Transfer Learning Demo

## Getting Started
Use of a virtual enviornment is highly recommended although not required.
```bash
virtualenv .venv
source .venv/bin/activate
```

`pip install -r requirements.txt` will install the following packages necessary for this demo to run.
- ngraph_tensorflow_bridge
- tensorflow==1.14.0
- plaidml-keras
- pillow
- matplotlib
- jupyter

## Running the demo through the Jupyter Notebook

Open the Jupyter Notebook using the following command:

```python
jupyter notebook TransferLearningDemo.ipynb
```

## Running the demo through the command line
```
usage: TransferLearningDemo [-h] [--gui] [--training]
                            [--network_type NETWORK_TYPE] [--backend BACKEND]
                            [--quiet]

optional arguments:
  -h, --help            show this help message and exit
  --gui                 shows the GUI of the demo
  --training            performs the training phase of the demo
  --network_type NETWORK_TYPE
                        selects the network used for training/classification
                        [ResNet50]/MobileNet V2
  --backend BACKEND     selects the backend used for training/classification
                        (run ngraph_bridge.list_backends() for full list)
  --quiet               disables most logging
```
