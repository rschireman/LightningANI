from pytorch_lightning import Trainer, seed_everything
from lightning_ani.nnp import NNPLightningModel
from lightning_ani.nnp_data_module import NNPDataModule
from lightning_ani.nnp_sigmoid import NNPLightningSigmoidModel
from lightning_ani.nnp import NNPLightningModelDF

def test_nnp():
    seed_everything(1234)
    data = NNPDataModule(data_dir="BTBT-NPT-300K-and-100K.h5", batch_size=32)
    aev_dim = data.get_aev_dim()
    aev_computer = data.aev_computer
    model = NNPLightningModel(aev_computer=aev_computer, aev_dim=aev_dim,learning_rate=1e-6)
    trainer = Trainer(max_epochs=1)
    assert data is not None and model is not None and trainer is not None

