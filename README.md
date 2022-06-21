<div align="center">    
 
# LightningANI

<!-- [![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539) -->
<!-- [![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018) -->
<!-- [![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018) -->
<!-- [![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)   -->
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)


<!--  
Conference   
-->   
</div>
 
## Description   
Pytorch Lightning template for creating neural network potentials (NNP) with torchANI 

## How to run   
First, install dependencies   
```bash
# clone project   
git clone git@github.com:rschireman/LightningANI.git

# install project   
cd ANI-NNP-pl
pip install -e .   
pip install -r requirements.txt
 ```   
 Next, navigate to any file and run it.   
 ```bash
# module folder
cd project

# run module 
python nnp_delayed_force_training.py --data_dir /path/to/dataset    
```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from pytorch_lightning import Trainer
from lightning_ani.nnp import NNPLightningModel
from lightning_ani.nnp_data_module import NNPDataModule
import wandb
from pytorch_lightning.loggers import WandbLogger
import numpy as np

wandb.init(project="BTBT-NPT-rocm-pl-batch_size-sweep")
wandb.run.name = "batch_size = 512"
wandb.run.save()
wandb_logger = WandbLogger()
data = NNPDataModule(data_dir="BTBT-NPT-300K-and-100K.h5", batch_size=512)
aev_dim = data.get_aev_dim()
aev_computer = data.aev_computer
model = NNPLightningModel(aev_computer=aev_computer, aev_dim=aev_dim,learning_rate=1e-5,force_coefficient=10,batch_size=512)
trainer = Trainer(max_epochs=10000,gpus=1,logger=wandb_logger)
trainer.fit(model,data)
```

<!-- ### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```    -->
