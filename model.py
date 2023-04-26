from torchvision import transforms
from torchvision.models.resnet import resnet152
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.cuda.amp import autocast
from torch.utils.data.distributed import DistributedSampler

from pytorch_lightning.loggers import WandbLogger
import os


# config
EPOCHS = 2
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
PRINT_FREQ = 50
TRAIN_BATCH=128
VAL_BATCH=128
imagenet_mean_RGB = [0.47889522, 0.47227842, 0.43047404]
imagenet_std_RGB = [0.229, 0.224, 0.225]
local_rank = int(os.environ['LOCAL_RANK'])

class IMAGENETDataModule(pl.LightningDataModule):
    def __init__(self, train_batch_size, val_batch_size, local_rank,data_dir: str = './'):
        super().__init__()
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.local_rank = local_rank

        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean_RGB, imagenet_std_RGB),
        ])
        
        self.transform_val = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean_RGB, imagenet_std_RGB),
        ])
        
        self.dims = (3, 224, 224)
        self.num_classes = 1000

    def setup(self, stage='fit'):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.imagenet_train = datasets.ImageFolder(root='data/train', transform=self.transform_train)
            self.imagenet_val = datasets.ImageFolder(root='data/val',  transform=self.transform_val)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.imagenet_test = datasets.ImageFolder(root='data/test',  transform=self.transform_val)

    def train_dataloader(self):
        train_sampler = DistributedSampler(self.imagenet_train, num_replicas=2, rank=self.local_rank)
        return DataLoader(self.imagenet_train, batch_size=self.train_batch_size, num_workers=2, shuffle=False, sampler=train_sampler)
    
    def val_dataloader(self):
        val_sampler = DistributedSampler(self.imagenet_val, num_replicas=2, rank=self.local_rank)
        return DataLoader(self.imagenet_val, batch_size=self.val_batch_size, num_workers=2, sampler=val_sampler)

    def test_dataloader(self):
        return DataLoader(self.imagenet_test, batch_size=self.batch_size, num_workers = 2)

dm = IMAGENETDataModule(TRAIN_BATCH, VAL_BATCH, local_rank)
dm.setup()

train_data = dm.imagenet_train
val_data = dm.imagenet_val

train_loader = dm.train_dataloader()
val_loader = dm.val_dataloader()

train_batch = next(iter(train_loader))
val_batch = next(iter(val_loader))

train_imgs, train_labels = train_batch
val_imgs, val_labels = val_batch

print("Train images shape:", train_imgs.shape)
print("Train labels shape:", train_labels.shape)

print("Validation images shape:", val_imgs.shape)
print("Validation labels shape:", val_labels.shape)

MODEL_CKPT_PATH = 'model/'
MODEL_CKPT = 'model/model-{epoch:02d}-{val_loss:.2f}'

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    filename=MODEL_CKPT ,
    save_top_k=3,
    mode='min')

# Samples required by the custom ImagePredictionLogger callback to log image predictions.
val_samples = next(iter(dm.val_dataloader()))
val_imgs, val_labels = val_samples[0], val_samples[1]
val_imgs.shape, val_labels.shape

early_stop_callback = EarlyStopping(
   monitor='val_loss',
   patience=3,
   verbose=False,
   mode='min'
)

# model
class LitResnet152(LightningModule):
    def __init__(self, learning_rate, momentum, weight_decay):
        super().__init__()
        self.nn = resnet152(pretrained = True)
        self.nn.fc = nn.Linear(self.nn.fc.in_features, 1000)
        self.lr = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss().cuda(GPU)
    
    def forward(self, x):
        return self.nn.forward(x)
    
    def training_step(self, batch, batch_idx):
        x,y = batch
        #AMP
        with autocast():
            logits = self.nn(x)
            loss = self.criterion(logits, y)
        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=1000)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=False)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=False)
        if batch_idx % PRINT_FREQ == 0:
          print("train step! " + str(batch_idx) + " train loss: " + str(loss.item()) + " train acc " + str(acc.item()))        
        return loss     
        
    def validation_step(self, batch, batch_idx):
        x,y = batch
        with autocast():
            logits = self.nn(x)
            loss = self.criterion(logits, y) 
        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=1000)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        if batch_idx % PRINT_FREQ == 0:
          print("val step! " + str(batch_idx) + " val loss: " + str(loss.item()) + " val acc " + str(acc.item()))
        return loss  
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.nn.parameters(), self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        return optimizer

# Set up TensorBoard logger
wandb_logger = WandbLogger(project='h9')
wandb_logger.experiment.config["batch_size"] = TRAIN_BATCH

# model = resnet18(pretrained = False, progress  = True)
model = LitResnet152(LR, MOMENTUM, WEIGHT_DECAY)

# Initialize a trainer
trainer = pl.Trainer(max_epochs=EPOCHS, gpus=1, accelerator='ddp_spawn', logger=wandb_logger)

# train the model
trainer.fit(model, dm,train_dataloaders=train_loader)