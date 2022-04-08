from pytorch_lightning import Trainer, seed_everything
from NNP.nnp import NNPLightningModel
from NNP.nnp import NNPDataModule


def test_lit_classifier():
    seed_everything(1234)

    data = NNPDataModule(data_dir="./BTBT-NPT-300K-and-100K.h5'",batch_size=32)
    aev_dim = data.get_aev_dim()
    aev_computer = data.aev_computer
    model = NNPLightningModel(aev_computer=aev_computer, aev_dim=aev_dim,learning_rate=1e-5)
    print(data)
    trainer = Trainer(max_epochs=2)
    trainer.fit(model, data)
   
