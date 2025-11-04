import os
from pathlib import Path
from typing import Dict, Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, nx_modes, ny_modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nx_modes = nx_modes
        self.ny_modes = ny_modes

        self.weights = nn.Parameter(
            torch.randn(in_channels, out_channels, nx_modes, ny_modes, 2) * 0.01
        )

    def forward(self, x):
        B, in_c, H, W = x.shape
        x_ft = torch.fft.rfft2(x, s=(H,W), norm='ortho')

        out_ft = torch.zeros(B, self.out_channels, H, W//2+1,
                             dtype=x_ft.dtype, device=x.device)

        nx = min(self.nx_modes, H)
        ny = min(self.ny_modes, x_ft.shape[3])

        x_slice = x_ft[:, :, :nx, :ny]
        w_slice = self.weights[:, :, :nx, :ny, :]

        out_complex = complex_mul2d_broadcast(x_slice, w_slice)
        out_sum = out_complex.sum(dim=1)

        out_ft[:, :, :nx, :ny] = out_sum
        x = torch.fft.irfft2(out_ft, s=(H,W), norm='ortho')
        return x

def complex_mul2d_broadcast(a, w):

    ar = a.real.unsqueeze(2)
    ai = a.imag.unsqueeze(2)
    wr = w[...,0]
    wi = w[...,1]

    real = ar*wr - ai*wi
    imag = ar*wi + ai*wr
    return torch.complex(real, imag)

from task import build_input_tensor


class _GNAct(nn.Module):
    def __init__(self, c):
        super().__init__(); self.gn = nn.GroupNorm(max(1, c // 8), c); self.act = nn.SiLU(inplace=True)
    def forward(self, x): return self.act(self.gn(x))

class _DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), _GNAct(cout),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), _GNAct(cout),
        )
    def forward(self, x): return self.net(x)

class _Down(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
            _GNAct(cout),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            _GNAct(cout),
        )
    def forward(self, x): return self.net(x)

class _Up(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = _DoubleConv(cin, cout)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNetSmall(nn.Module):
    def __init__(self, in_ch, base=32, out_ch=1, img_size=None):
        super().__init__()
        self.enc1 = _DoubleConv(in_ch, base)
        self.enc2 = _Down(base, base*2)
        self.enc3 = _Down(base*2, base*4)
        self.bott = _Down(base*4, base*8)
        self.up3  = _Up(base*8 + base*4, base*4)
        self.up2  = _Up(base*4 + base*2, base*2)
        self.up1  = _Up(base*2 + base,   base)
        self.head = nn.Conv2d(base, out_ch, 1)

        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        b  = self.bott(s3)
        x  = self.up3(b, s3)
        x  = self.up2(x, s2)
        x  = self.up1(x, s1)
        return self.head(x)


class TinyCNN(nn.Module):
    def __init__(self, in_ch, img_size=(10,10), fc_hidden=128, channel=16):
        super().__init__()
        H, W = img_size
        self.H, self.W = H, W
        self.feature = nn.Sequential(
            nn.Conv2d(in_ch, channel, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channel, channel, 3, padding=1), nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(channel * H * W, fc_hidden), nn.ReLU(),
            nn.Linear(fc_hidden, H * W)
        )

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0)
        feat = self.feature(x)
        out  = self.fc(self.flatten(feat))
        return out.view(-1, 1, self.H, self.W)

class FNO2d(nn.Module):

    def __init__(self, in_ch=5, out_channels=1, modes=16, width=64, img_size=(32,32)):
        super().__init__()
        self.modes = modes
        self.width = width

        self.fc0 = nn.Linear(in_ch, width)

        self.conv1 = SpectralConv2d(width, width, modes, modes)
        self.conv2 = SpectralConv2d(width, width, modes, modes)
        self.conv3 = SpectralConv2d(width, width, modes, modes)
        self.conv4 = SpectralConv2d(width, width, modes, modes)

        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)
        self.w4 = nn.Conv2d(width, width, 1)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        B, in_c, H, W = x.shape 
        x = x.permute(0,2,3,1)  
        x = self.fc0(x)         
        x = x.permute(0,3,1,2)

        x1 = self.conv1(x)
        x = self.w1(x) + x1
        x = nn.GELU()(x)

        x2 = self.conv2(x)
        x = self.w2(x) + x2
        x = nn.GELU()(x)

        x3 = self.conv3(x)
        x = self.w3(x) + x3
        x = nn.GELU()(x)

        x4 = self.conv4(x)
        x = self.w4(x) + x4
        x = nn.GELU()(x)

        x = x.permute(0,2,3,1)
        x = self.fc1(x)
        x = nn.GELU()(x)
        x = self.fc2(x)
        x = x.permute(0,3,1,2)
        return x

class PDEModel(pl.LightningModule):
    def __init__(self, in_ch, lr=1e-3, img_size=(10,10), net=None):
        super().__init__()
        self.save_hyperparameters()
        self.net = net
        self.lr = lr
    def forward(self, x): return self.net(x)
    def training_step(self, batch, _):
        x, y = batch
        pred = self(x)
        loss = F.l1_loss(pred, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True); return loss
    def validation_step(self, batch, _):
        x, y = batch
        pred = self(x)
        loss = F.l1_loss(pred, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class PDEFieldDataset(Dataset):
    def __init__(self, samples, cache: bool = False, build_fn=None):
        self.samples = samples
        self.cache = cache
        self.build_fn = build_fn or build_input_tensor
        self._cached = []
        if cache:
            for s in samples:
                self._cached.append(self.build_fn(s))
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        if self.cache: return self._cached[i]
        return self.build_fn(self.samples[i])

class PDEDataModule(pl.LightningDataModule):
    def __init__(self, samples, batch_size=8, num_workers=8, val_ratio=0.1, pin_memory=True, cache=False, build_fn=None):
        super().__init__()
        self.samples = samples
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.pin_memory = pin_memory
        self.cache = cache
        self.build_fn = build_fn
    def setup(self, stage=None):
        full = PDEFieldDataset(self.samples, cache=self.cache, build_fn=self.build_fn)
        n_val = max(1, int(len(full) * self.val_ratio))
        n_train = len(full) - n_val
        self.ds_train, self.ds_val = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=self.num_workers>0)
    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=self.num_workers>0)

def get_model_lightning(model_class, samples, epochs=500, batch=8, lr=1e-2,
                        path="best.ckpt", train=True,
                        val_ratio=0.1, num_workers=8, patience=20,
                        accelerator="auto", devices="auto",
                        ckpt_dir="./checkpoints",
                        cache=False,
                        build_fn=None):

    x0, y0 = (build_fn or build_input_tensor)(samples[0])
    in_ch = x0.shape[0]; H, W = y0.shape[-2:]

    if train:
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        dm = PDEDataModule(samples, batch_size=batch, num_workers=num_workers, val_ratio=val_ratio, cache=cache, build_fn=build_fn)
        model = model_class(in_ch=in_ch, img_size=(H, W))
        model = PDEModel(in_ch=in_ch, lr=lr, img_size=(H, W), net=model)
        ckpt_cb = ModelCheckpoint(dirpath=ckpt_dir, filename="best", monitor="val_loss", mode="min", save_top_k=1)
        es_cb = EarlyStopping(monitor="val_loss", mode="min", patience=patience)
        trainer = pl.Trainer(max_epochs=epochs, accelerator=accelerator, devices=devices,
                             callbacks=[ckpt_cb, es_cb], default_root_dir=ckpt_dir,
                             log_every_n_steps=10, deterministic=False)
        trainer.fit(model, dm)
        best_path = ckpt_cb.best_model_path or os.path.join(ckpt_dir, "best.ckpt")
        if not ckpt_cb.best_model_path:
            trainer.save_checkpoint(best_path)
        print(f"Model saved to {best_path}")
        best = PDEModel.load_from_checkpoint(best_path, in_ch=in_ch, lr=lr, img_size=(H, W))
        best.eval()
        return best.net
    else:
        lit = PDEModel.load_from_checkpoint(path, in_ch=in_ch, lr=lr, img_size=(H, W), strict=False)
        lit.eval()
        return lit.net


def predict(model: nn.Module, test_sample: Dict[str, Any],
            device='cuda' if torch.cuda.is_available() else 'cpu',
            build_fn=None,
            requires_grad: bool = False):
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    x, _ = (build_fn or build_input_tensor)(test_sample)
    x = x.unsqueeze(0).to(device)

    if requires_grad:
        x.requires_grad_(True)
        pred = model(x).squeeze(0)
        return pred
    else:
        with torch.no_grad():
            pred = model(x).cpu().squeeze(0)
        return pred
