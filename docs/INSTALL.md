# Installation

### Requirements
All the codes are tested in the following environment:
* Linux (tested on Ubuntu 18.04)
* Python 3.6+
* PyTorch >= 1.1
* CUDA >= 9.0
* [`spconv v1.0 (commit 8da6f96)`](https://github.com/traveller59/spconv/tree/8da6f967fb9a054d8870c3515b1b44eca2103634) or [`spconv v1.2`](https://github.com/traveller59/spconv)
* NVIDIA Docker: 2.5.0 or higher (if using our provided docker image)


### Install `pcdet v0.3`
If the OpenPCDet detector does not work, try reinstalling within the docker container using the following steps. The `setup.py` file will install the pcdet library.

```shell
pip install -r requirements.txt 
python setup.py develop
```
