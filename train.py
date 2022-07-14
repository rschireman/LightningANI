from pytorch_lightning import Trainer
from lightning_ani.nnp import NNPLightningModel
from lightning_ani.nnp_data_module import NNPDataModule
import numpy as np



data = NNPDataModule(data_dir=".\\tests\\test.h5", batch_size=512)
aev_dim = data.get_aev_dim()
aev_computer = data.aev_computer
model = NNPLightningModel(aev_computer=aev_computer, aev_dim=aev_dim, learning_rate=1e-5, force_coefficient=10, batch_size=512)
trainer = Trainer(max_epochs=1000,gpus=1)
trainer.fit(model,data)