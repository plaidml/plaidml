## Transfer Learning Demo

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

For PlaidML devices, run `plaidml-setup` before starting this demo to select the Plaidml backend device.

Once the interface is presented, setup the model and network along with epoch and batch size values then hit the Train button to start. The demo will initialize test images and then start with a prediction with the model selected using pretrained `imagenet` weights. The demo will then automaticall proceed into the training phase followed by another predition phase based on the trained model.

The demo will present 9 randomly selected test images where the initial prediction incorrectly guessed and then the same 9 image predictions after the model has been trained.

## Running the demo through the Jupyter Notebook

Open the Jupyter Notebook using the following command:

```python
jupyter notebook TransferLearningDemo.ipynb
```

## Running the demo through the command line
```
usage: TransferLearningDemo [-h] [--gui] [--training] 
                            [--network_type NETWORK_TYPE] [--backend BACKEND]
                            [--warmup] [--workers] [--quiet]

optional arguments:
  -h, --help            show this help message and exit
  --gui                 shows the GUI of the demo
  --training            performs the training phase of the demo
  --network_type NETWORK_TYPE
                        selects the network used for training/classification
                        [ResNet50]/MobileNet V2
  --backend BACKEND     selects the backend used for training/classification [CPU]/PLAIDML/TF]
  --warmup              train with a warmup run first
  --workers             identify the number of workers
  --quiet               disables most logging
```
