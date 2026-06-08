import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import TensorDataset, DataLoader, random_split

from SpecML import SpecML
from Tokeniser import patch_size, step_size, P, V, X
import time
import torch.nn as nn


start_time = time.time()
#------------------------------------------TRAINING PARAMETERS----------------------------------------------------#

N_STEPS_PER_RESTART = 5000  # gradient steps
BATCH_SIZE = 1024  # spectra per batch
LR = 1e-4  # AdamW learning rate
WEIGHT_DECAY = 0.01  # AdamW weight decay
BETAS = (0.9, 0.95)  # AdamW β₁, β₂
GRAD_CLIP = 1.0  # gradient clip max norm
SCHED_ETA_MIN = 1e-6  # minimum LR after annealing
NUM_RESTARTS = 6  # number of annealing cycles
WARMUP_STEPS = 2000
N_STEPS = N_STEPS_PER_RESTART * NUM_RESTARTS
TRAIN_VAL_SPLIT = 0.9  # fraction of data used for training

#-------------------------------------------------MASKING-----------------------------------------------------------#

CHUNK_WIDTH = int(np.floor(2.5 * patch_size / step_size)) # chunk width in tokens
MASK_RATIO = 0.5  # fraction of max_chunks to mask per spectrum

def apply_chunk_mask_batch(
    y_b: torch.Tensor,
    v_b: torch.Tensor,
    mask_ratio: float = MASK_RATIO,
    chunk_width: int = CHUNK_WIDTH,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask one batch entirely on device"""
    B, T, _ = y_b.shape
    device = y_b.device
    M = torch.zeros(B, T, dtype=torch.bool, device=device)

    valid_lens = v_b.sum(dim=1).int()
    n_chunks = valid_lens // chunk_width
    k = ((((n_chunks + 1) // 2) * mask_ratio).int()).clamp(min=1)
    section = valid_lens // k.clamp(min=1)
    usable = (valid_lens >= chunk_width) & (section >= chunk_width)

    offsets = torch.arange(chunk_width, device=device)
    k_max = int(k[usable].max()) if usable.any() else 0

    for j in range(k_max):
        active = torch.where(usable & (k > j))[0]
        sec = section[active]
        u = torch.rand(len(active), device=device)
        starts = (j * sec + u * (sec - chunk_width + 1)).int()
        M[active[:, None], starts[:, None] + offsets] = True

    x_b = y_b.clone()
    x_b[M] = 0.0
    return x_b, M


#-------------------------------------------------LOSS-----------------------------------------------------------#

def mse_loss(Y, Yhat, M):
    err = ((Y - Yhat) ** 2).sum(dim=-1)  # [B, T]  squared L2 norm over patch dim
    return err[M].mean()  # mean over masked positions only


#-------------------------------------------------SCHEDULER-----------------------------------------------------------#

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, T_0, eta_min=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.T_0 = T_0
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * self.last_epoch / max(1, self.warmup_steps)
                    for base_lr in self.base_lrs]
        progress = (self.last_epoch - self.warmup_steps) / max(1, self.T_0 - self.warmup_steps)
        cosine = 0.5 * (1 + np.cos(np.pi * (progress % 1.0)))
        return [self.eta_min + (base_lr - self.eta_min) * cosine
                for base_lr in self.base_lrs]


#-------------------------------------------------LIGHTNING MODULE-----------------------------------------------------------#

class SpecMLLit(pl.LightningModule):
    def __init__(self, patch_dim, P_enc: np.ndarray):
        super().__init__()
        self.model = SpecML(patch_dim=patch_dim)
        # Register P as a buffer so it moves to the correct device automatically
        self.register_buffer('P', torch.from_numpy(P_enc).float())

    def training_step(self, batch, batch_idx):
        y_b, v_b = batch
        x_b, m_b = apply_chunk_mask_batch(y_b, v_b)
        Yhat = self.model(x_b, v_b, self.P)
        loss = mse_loss(y_b, Yhat, m_b)
        self.log('train_loss', loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_b, v_b = batch
        x_b, m_b = apply_chunk_mask_batch(y_b, v_b)
        Yhat = self.model(x_b, v_b, self.P)
        loss = mse_loss(y_b, Yhat, m_b)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.model.parameters(), lr=LR,
            weight_decay=WEIGHT_DECAY, betas=BETAS,
        )
        scheduler = WarmupCosineScheduler(
            opt, warmup_steps=WARMUP_STEPS,
            T_0=N_STEPS_PER_RESTART, eta_min=SCHED_ETA_MIN,
        )
        return {
            'optimizer': opt,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'},
        }


#-------------------------------------------------TRAINING-----------------------------------------------------------#

if __name__ == '__main__':
    dataset = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(V).bool(),
    )
    n_train = int(len(dataset) * TRAIN_VAL_SPLIT)
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    lit_model = SpecMLLit(patch_dim=patch_size + 2, P_enc=P)

    checkpoint_cb = ModelCheckpoint(
        dirpath='checkpoints/',
        filename='specml-step{step}-val{val_loss:.4f}',
        every_n_train_steps=N_STEPS_PER_RESTART,  # save at the end of each annealing restart
        save_top_k=3,
        monitor='val_loss',
        mode='min',
        save_last=True,  # always keep checkpoints/last.ckpt
    )
    early_stop_cb = EarlyStopping(
        monitor='val_loss',
        patience=50,  # high patience for self-supervised training where loss oscillates with cosine restarts
        mode='min',
    )

    logger = CSVLogger(save_dir='outputs/', name='specml')

    trainer = pl.Trainer(
        max_epochs=1000,
        gradient_clip_val=GRAD_CLIP,
        callbacks=[checkpoint_cb, early_stop_cb],
        log_every_n_steps=10,
        logger=logger,
    )

    # To resume from a checkpoint, pass: ckpt_path='checkpoints/last.ckpt'
    trainer.fit(lit_model, train_loader, val_loader)

    # Save final weights in original flat format
    torch.save(lit_model.model.state_dict(), 'SpecML.pt')

end_time = time.time() - start_time
print(end_time)
