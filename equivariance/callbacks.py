from typing import Union, Optional, Any, Dict
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch as ch
from torch.nn.modules.loss import _Loss
from attack.attack_steps import AttackerStep
from attack.losses import BaseLoss, CompositeLoss

class VectorCallback(Callback):
    """
    Callback passed to Trainer to (dynamically) set vector params.
    This allows a general forward method to be shared between 
    InvertedRepWrapper and StretchedInvertedRepWrapper
    """
    def __init__(self,
                 vector_type: str,
                 vector_params: Dict,
                 stretch_factor: float
                 ):
        ## See equivariance.stretched_inversion_callback.StretchedInvertedRepWrapper's 
        ## forward function for details of args
        super().__init__()
        self.vector_type = vector_type
        self.vector_params = vector_params
        self.stretch_factor = stretch_factor
    
    def get_vector_kwargs(self):
        return {
            'vector_type': self.vector_type,
            'vector_params': self.vector_params, 
            'stretch_factor': self.stretch_factor
        }

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pl_module.__setattr__('vector_kwargs', self.get_vector_kwargs())
    
    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pl_module.__setattr__('vector_kwargs', self.get_vector_kwargs())        
    
    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pl_module.__setattr__('vector_kwargs', self.get_vector_kwargs())
    
    def on_predict_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pl_module.__setattr__('vector_kwargs', self.get_vector_kwargs())
    
    