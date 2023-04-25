import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="7"

from omnidata.omnidata_tools.torch.data.omnidata_dataset import OmnidataDataset
from transformers import ViTFeatureExtractor, ViTModel,  ViTConfig, ViTPreTrainedModel, TrainingArguments, Trainer, EarlyStoppingCallback
from torchvision import transforms
import timm
import transformers
import torch
import torch.nn as nn
import torchvision.transforms as T
from typing import Dict, List, Optional, Set, Tuple, Union
import math
import wandb
from datasets import load_dataset
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import datetime

#___Params
dataset_name='cifar10'
pretrained_model_name="vit_base_patch32_224_in21k"
schedule_name='linear'
num_train_epochs=100
warmup_epochs=2 #if linear
batch_size=64
test=True
report_to="wandb"

#_____

norm_mean=torch.tensor([0.5, 0.5, 0.5])
norm_std=torch.tensor([0.5, 0.5, 0.5])



class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset,model):
        self.dataset=dataset
        self.model=model
        config = resolve_data_config({}, model=self.model)
        self.transform = create_transform(**config)
        
    def __getitem__(self, idx):
        item={}
        item['x']=self.transform(self.dataset[idx]['pixel_values'])
        item['labels']=self.transform(self.dataset[idx]['pixel_values'])
        return item

    def __len__(self):
        return len(self.dataset)

class vit2Image(nn.Module):
    def __init__(self, pretrained_model_name):
        super().__init__()
        
        self.vit= timm.create_model(pretrained_model_name, pretrained=True) #torch.Size([128, 384])
        self.vit.reset_classifier(0,'')#torch.Size([128, 197, 384])
        
        hidden_size=768
        self.num_channels=3
        encoder_stride=32
        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_size,
                out_channels=encoder_stride**2 * self.num_channels,
                kernel_size=1,
            ),
            nn.PixelShuffle(encoder_stride),
        )

        # Initialize weights and apply final processing
        #self.post_init()
        
    def forward(self,
        x: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        
        if(len(x.size())==5):
            x=torch.squeeze(x, 1)
        if(len(labels.size())==5):
            labels=torch.squeeze(labels, 1)
        sequence_output=self.vit(x)
        sequence_output = sequence_output[:, 1:]
        
        batch_size, sequence_length, num_channels = sequence_output.shape
        height = width = math.floor(sequence_length**0.5)
        sequence_output = sequence_output.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)
        
        reconstructed_pixel_values = self.decoder(sequence_output)
        loss=None
        if(labels is not None):
            loss = nn.functional.l1_loss(labels, reconstructed_pixel_values, reduction="none")
            # TODO
            loss=loss.sum()/ (reconstructed_pixel_values.size(dim=-1) + 1e-5) / self.num_channels / batch_size
       
        return loss, reconstructed_pixel_values
   
#Model
model=vit2Image(pretrained_model_name)
for param in model.vit.parameters():
    param.requires_grad = True
model.to(torch.device('cuda:0'))

# Load dataset
dataset = load_dataset(dataset_name)

if(test==False):
    data_split=dataset['train'].train_test_split(test_size=0.1)
    data_train=data_split['train']
    data_val=data_split['test']
else:
    data_train=dataset['train']
data_test=dataset['test']

_augmentations = transforms.Compose([transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.25,.25,.25),
            transforms.RandomRotation(2)])

def augmentations(examples):
    examples["pixel_values"] = [_augmentations(image.convert("RGB")) for image in examples['img']]
    return examples


data_train.set_transform(augmentations)
data_train=CustomDataset(data_train,model.vit)
data_test.set_transform(augmentations)
data_test=CustomDataset(data_test,model.vit)

if(test==False):
    data_val.set_transform(augmentations)
    data_val=CustomDataset(data_val,model.vit)




run_name=datetime.datetime.now().strftime("%Y%m%d%H_%M_%S")
if(test==True):
    run_name=run_name+"Final"
wandb.init(project="effects_of_finetuning", name=run_name)
training_args = TrainingArguments(
    num_train_epochs=num_train_epochs,
    max_grad_norm=1,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="epoch",
    logging_strategy="steps",
    logging_steps=1,
    save_strategy="epoch",
    remove_unused_columns=False,
    report_to=report_to,
    output_dir='./_tmp_wandb',
    load_best_model_at_end=True,
    save_total_limit=3
)

optimizer=torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.9,weight_decay=1e-4)
if(schedule_name=='cosine'):
    total_steps=120
    warmup_steps=100
    lr_scheduler=transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,num_training_steps=total_steps)
else:
    total_steps=len(data_train)/batch_size*num_train_epochs
    warmup_steps=warmup_epochs*len(data_train)/batch_size
    lr_scheduler=transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,num_training_steps=total_steps)
    
trainer = Trainer(
    model=model,
    optimizers=(optimizer,lr_scheduler),
    args=training_args,
    train_dataset=data_train,
    eval_dataset=data_val if test==False else data_test,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=1e-3)]
)

train_results = trainer.train()

trainer.save_model("./final_models/"+run_name)


