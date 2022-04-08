from pytorch_lightning import Trainer, seed_everything
from NNP.nnp import NNPLightningModel
from NNP.nnp import NNPDataModule


def test_lit_classifier():
    seed_everything(1234)

    model = NNPLightningModel()
    data = NNPDataModule(data_dir="BTBT-NPT-300K-and-100K.h5'")
   
