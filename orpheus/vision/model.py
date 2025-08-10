import torch
import os
import torch.nn as nn
from einops import repeat
import pytorch_lightning as pl
import torch.optim as optim
from torchmetrics import ConcordanceCorrCoef, PearsonCorrCoef, MeanSquaredError


class TileTransformer(pl.LightningModule):
    def __init__(
        self,
        lr=2e-5,
        warm_up=1000,
        lr_decay=0.9999,
        layers=2,
        input_dim=768,
        decay=2e-5,
        dropout=0.1,
        latent_dim=512,
        heads=8,
        preds_output_dir="preds/visual"
    ):
        super().__init__()
        assert latent_dim % heads == 0
        self.warm_up_step = warm_up
        self.lr_decay = lr_decay
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.learning_rate = lr
        self.weight_decay = decay
        self.preds_output_dir = preds_output_dir
        self.save_hyperparameters()

        self.latent_embedder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
        )

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=heads,
                dim_feedforward=latent_dim,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=layers,
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, latent_dim))
        
        self.reg_head = nn.Sequential(nn.LayerNorm(self.latent_dim),
                               nn.Linear(self.latent_dim, 1))

        # Will be set in setup()
        self.pos_weight = None        
                                 
        setattr(self, "train_concordance", ConcordanceCorrCoef())
        setattr(self, "train_pearson", PearsonCorrCoef())
        setattr(self, "train_mse", MeanSquaredError())
        setattr(self, "val_concordance", ConcordanceCorrCoef())
        setattr(self, "val_pearson", PearsonCorrCoef())
        setattr(self, "val_mse", MeanSquaredError())


    def setup(self, stage: str):
        """Compute pos_weight = (#neg / #pos) from the training split once we have the datamodule."""
        if self.pos_weight is not None:
            return

        dm = getattr(self.trainer, "datamodule", None)
        if dm is None:
            return  # nothing we can do

        pos = 0
        neg = 0

        # Try to grab labels from a training dataset dataframe if exposed
        y_iterable = None
        for cand in ["train_dataset", "dataset_train", "train_set"]:
            if hasattr(dm, cand):
                ds = getattr(dm, cand)
                # Try common patterns for labels
                if hasattr(ds, "df") and self.label_column in getattr(ds, "df").columns:
                    y_iterable = getattr(ds, "df")[self.label_column].values
                elif hasattr(ds, "labels"):
                    y_iterable = getattr(ds, "labels")
                break

        # Fallback: iterate one epoch of the train dataloader just to count
        if y_iterable is None and hasattr(dm, "train_dataloader"):
            try:
                for batch in dm.train_dataloader():
                    y_b = batch["y"]
                    # assume y is 0/1 (float or long); move to cpu for counting
                    y_b = (y_b.detach().cpu().numpy().reshape(-1) > 0.5).astype(int)
                    pos += int(y_b.sum())
                    neg += int((1 - y_b).sum())
            except Exception:
                pass

        # If we found a direct iterable of labels, count from it
        if y_iterable is not None:
            import numpy as np
            y_arr = np.asarray(y_iterable).astype(float).reshape(-1)
            # treat >=0.5 as positive
            y_bin = (y_arr >= 0.5).astype(int)
            pos = int(y_bin.sum())
            neg = int((len(y_bin) - y_bin.sum()))

        # Safety: avoid division by zero
        if pos == 0:
            # All negative -> give small positive weight
            ratio = 1.0
        elif neg == 0:
            # All positive -> heavy weight so they still contribute in practice
            ratio = 1.0
        else:
            ratio = neg / pos

        # Store as tensor on correct device later in forward/loss
        self.pos_weight = torch.tensor([ratio], dtype=torch.float32)
        self.log("train_pos_weight", self.pos_weight.item(), prog_bar=True)

    
    def add_cls_token(self, x):
        if x.shape[0] > 1:
            cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=x.shape[0])
        else:
            cls_tokens = self.cls_token
        x = torch.cat((cls_tokens, x), dim=1)
        return x
    
    def training_step(self, batch, batch_idx):
        y_hat = self.forward(batch['x'])['y_hat']
        loss = self.calculate_loss(y_hat, batch['y'].view(-1, 1))
        self.log_all(loss, y_hat, batch['y'], "train")
        return loss
    
    # <-- not static anymore
    def calculate_loss(y_hat, y):
        # Ensure tensors are float and on same device
        y = y.to(y_hat.device, dtype=torch.float32)
        pos_w = self.pos_weight.to(y_hat.device) if self.pos_weight is not None else None
        return nn.functional.binary_cross_entropy_with_logits(
            y_hat, y, reduction="mean", pos_weight=pos_w
        )

    def validation_step(self, batch, batch_idx):
        y_hat = self.forward(batch['x'])['y_hat']
        loss = self.calculate_loss(y_hat, batch['y'].view(-1, 1))
        self.log_all(loss, y_hat, batch['y'], "val")

    def log_all(self, loss, y_hat, y, subset):
        logging_kwargs = {"on_step": True,
                          "on_epoch": True,
                          "sync_dist": True,
                          "batch_size": y.shape[0]}

        mse_metric = getattr(self, f"{subset}_mse")
        mse_metric(y_hat.reshape(-1), y.reshape(-1))
        self.log(f"{subset}_mse", mse_metric, **logging_kwargs)
        
        concordance_metric = getattr(self, f"{subset}_concordance")
        concordance_metric(y_hat.reshape(-1), y.reshape(-1))
        self.log(f"{subset}_concordance", concordance_metric, **logging_kwargs)

        pearson_metric = getattr(self, f"{subset}_pearson")
        pearson_metric(y_hat.reshape(-1), y.reshape(-1))
        self.log(f"{subset}_pearson", pearson_metric, **logging_kwargs)

        self.log(f"{subset}_loss", loss, **logging_kwargs)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        output = self.forward(batch['x'])
        y_hat, emb = output['y_hat'], output['emb']
        emb_output_filenames = batch["output_visual_embedding_path"]
        if not os.path.exists(os.path.dirname(emb_output_filenames[0])):
            os.makedirs(os.path.dirname(emb_output_filenames[0]))
        for i, emb_output_filename in enumerate(emb_output_filenames):
            torch.save(emb[i].cpu(), emb_output_filename)
        for i, y_hat_i in enumerate(y_hat):
            prediction_file_name = os.path.join(self.preds_output_dir, batch["split"][i], f"{batch['case_id'][i]}.pt")
            if not os.path.exists(os.path.dirname(prediction_file_name)):
                os.makedirs(os.path.dirname(prediction_file_name))
            torch.save(y_hat_i.cpu(), prediction_file_name)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        def calc_lr(epoch):
                step = self.trainer.global_step
                if step < self.warm_up_step:
                    lr_scale = float(step) / self.warm_up_step
                else:
                    lr_scale = self.lr_decay ** (step - self.warm_up_step)
                return lr_scale

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=calc_lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "reduce_on_plateau": False,
            },
            "monitor": "val_loss",
        }

    def forward(self, x):
        x = self.latent_embedder(x)
        x = self.add_cls_token(x)
        x = self.transformer(x)
        z = x[:, 0]
        return {'emb': z, 'y_hat': self.reg_head(z)}
