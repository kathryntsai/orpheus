import torch
import os
import torch.nn as nn
from einops import repeat
import pytorch_lightning as pl
import torch.optim as optim
from torchmetrics import ConcordanceCorrCoef, PearsonCorrCoef, MeanSquaredError

# Regression metrics
from torchmetrics import ConcordanceCorrCoef, PearsonCorrCoef, MeanSquaredError
# Binary metrics
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision, BinaryAccuracy

class TileTransformer(pl.LightningModule):
    """
    Transformer over tile embeddings with task-specific heads/losses.

    Args
    ----
    task: "regression" | "binary" | "survival"
        Selects the loss + metrics and how labels are read from batches.
    label_column: str
        Used only for binary task to auto-compute pos_weight from training data.
        Set this to the column name in your CSV (e.g., "y" or "class" or "score" if 0/1).
    censor_is_event: bool
        For survival task, set True if your 'censor' column means event=1.
        Set False if 'censor' means censored=1 (then event = 1 - censor).
    """
    
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
        task: str = "regression",          # "regression" | "binary" | "survival"
        label_column: str = "y",
        censor_is_event: bool = True,
    ):
        super().__init__()
        assert latent_dim % heads == 0, "latent_dim must be divisible by heads"
        assert task in {"regression", "binary", "survival"}, "Invalid task"

        # Hyperparameters / config
        self.warm_up_step = warm_up
        self.lr_decay = lr_decay
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.learning_rate = lr
        self.weight_decay = decay
        self.preds_output_dir = preds_output_dir
        self.save_hyperparameters()

        # Projection into transformer space
        self.latent_embedder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
        )

        # Transformer encoder
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

        # CLS token and head (1-dim output)
        self.cls_token = nn.Parameter(torch.randn(1, 1, latent_dim))

        # Binary pos_weight (computed in setup from training labels if task == "binary")
        self.reg_head = nn.Sequential(nn.LayerNorm(self.latent_dim),
                               nn.Linear(self.latent_dim, 1))

        # Will be set in setup()
        self.pos_weight = None        # torch.Tensor or None

        # Metrics
        if self.task == "regression":
            self.train_concordance = ConcordanceCorrCoef()
            self.train_pearson = PearsonCorrCoef()
            self.train_mse = MeanSquaredError()
            self.val_concordance = ConcordanceCorrCoef()
            self.val_pearson = PearsonCorrCoef()
            self.val_mse = MeanSquaredError()
        elif self.task == "binary":
            self.train_auroc = BinaryAUROC()
            self.train_ap = BinaryAveragePrecision()
            self.train_acc = BinaryAccuracy()
            self.val_auroc = BinaryAUROC()
            self.val_ap = BinaryAveragePrecision()
            self.val_acc = BinaryAccuracy()
        else:
            # survival: log only loss by default (c-index could be added with an epoch aggregator)
            pass

    # ---------- utils ----------

    def add_cls_token(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] or [T, D] when B==1 from dataset; dataloader collates B
        if x.shape[0] > 1:
            cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=x.shape[0])
        else:
            cls_tokens = self.cls_token
        x = torch.cat((cls_tokens, x), dim=1)
        return x

    # ---------- PL lifecycle ----------
    def setup(self, stage: str):
        """If binary, compute pos_weight = (#neg / #pos) from the training split."""
        if self.task != "binary" or self.pos_weight is not None:
            return

        if self.pos_weight is not None:
            return

        dm = getattr(self.trainer, "datamodule", None)
        if dm is None:
            return  # nothing we can do

        # Try reading labels directly from the train dataset's dataframe
        pos = 0
        neg = 0
        ds = getattr(dm, "train_ds", None)
        if ds is not None and hasattr(ds, "df") and self.label_column in ds.df.columns:
            y = ds.df[self.label_column].to_numpy()
            import numpy as np
            y = np.asarray(y).astype(float).reshape(-1)
            y_bin = (y >= 0.5).astype(int)
            pos = int(y_bin.sum())
            neg = int(len(y_bin) - pos)

        # Try to grab labels from a training dataset dataframe if exposed
        # y_iterable = None
        # for cand in ["train_dataset", "dataset_train", "train_set"]:
        #     if hasattr(dm, cand):
        #         ds = getattr(dm, cand)
        #         # Try common patterns for labels
        #         if hasattr(ds, "df") and self.label_column in getattr(ds, "df").columns:
        #             y_iterable = getattr(ds, "df")[self.label_column].values
        #         elif hasattr(ds, "labels"):
        #             y_iterable = getattr(ds, "labels")
        #         break


        # Fallback: light scan over dataloader
        if pos + neg == 0 and hasattr(dm, "train_dataloader"):
            try:
                for i, batch in enumerate(dm.train_dataloader()):
                    y_b = batch.get("y", None)
                    if y_b is None:
                        continue
                    y_b = (y_b.detach().cpu().numpy().reshape(-1) >= 0.5).astype(int)
                    pos += int(y_b.sum())
                    neg += int((1 - y_b).sum())
                    if i > 50:  # avoid scanning full epoch
                        break
            except Exception:
                pass

        # Fallback: iterate one epoch of the train dataloader just to count
        # if y_iterable is None and hasattr(dm, "train_dataloader"):
        #    try:
        #        for batch in dm.train_dataloader():
        #            y_b = batch["y"]
        #            # assume y is 0/1 (float or long); move to cpu for counting
        #            y_b = (y_b.detach().cpu().numpy().reshape(-1) > 0.5).astype(int)
        #            pos += int(y_b.sum())
        #            neg += int((1 - y_b).sum())
        #    except Exception:
        #        pass

        # If we found a direct iterable of labels, count from it
        # if y_iterable is not None:
        #    import numpy as np
        #    y_arr = np.asarray(y_iterable).astype(float).reshape(-1)
        #    # treat >=0.5 as positive
        #    y_bin = (y_arr >= 0.5).astype(int)
        #    pos = int(y_bin.sum())
        #    neg = int((len(y_bin) - y_bin.sum()))

        # Safety: avoid division by zero
        # if pos == 0:
        #    # All negative -> give small positive weight
        #    ratio = 1.0
        # elif neg == 0:
        #    # All positive -> heavy weight so they still contribute in practice
        #    ratio = 1.0
        # else:
        #    ratio = neg / pos

        ratio = 1.0 if pos == 0 or neg == 0 else (neg / pos)
        
        # Store as tensor on correct device later in forward/loss
        self.pos_weight = torch.tensor([ratio], dtype=torch.float32)
        # Logging inside setup can be flaky; it's okay if this doesn't show up every time.
        try:
            self.log("train_pos_weight", self.pos_weight.item(), prog_bar=True, on_step=False, on_epoch=True)
        except Exception:
            pass

    # ---------- forward / loss / step ----------
    def forward(self, x: torch.Tensor):
        # x shape expected: [B, T, input_dim]
        x = self.latent_embedder(x)
        x = self.add_cls_token(x)
        x = self.transformer(x)
        z = x[:, 0]  # CLS embedding
        y_hat = self.reg_head(z)  # logits/regression/risk
        return {"emb": z, "y_hat": y_hat}

    def calculate_loss(self, y_hat: torch.Tensor, batch: dict) -> torch.Tensor:
        """
        task-specific loss:
          - regression: MSE between y_hat and y
          - binary: BCEWithLogits(y_hat, y) with pos_weight
          - survival: negative Cox partial log-likelihood (uses tte, censor/event)
        """
        if self.task == "regression":
            y = batch["y"].view(-1, 1).to(y_hat.device, dtype=torch.float32)
            return nn.functional.mse_loss(y_hat, y, reduction="mean")

        if self.task == "binary":
            y = batch["y"].view(-1, 1).to(y_hat.device, dtype=torch.float32)
            pw = self.pos_weight if self.pos_weight is not None else None
            return nn.functional.binary_cross_entropy_with_logits(y_hat, y, pos_weight=pw)

        # survival
        tte = batch["tte"].view(-1).to(y_hat.device, dtype=torch.float32)     # time-to-event
        censor = batch["censor"].view(-1).to(y_hat.device, dtype=torch.float32)
        # 'event' must be 1 for observed event, 0 for censored
        event = censor if self.censor_is_event else (1.0 - censor)
        risk = y_hat.view(-1)  # linear predictor

        return self._cox_ph_loss(risk, tte, event)


    @staticmethod
    def _cox_ph_loss(risk: torch.Tensor, tte: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        """
        Negative Cox partial log-likelihood with log-sum-exp trick.
        risk: (N,) linear predictors
        tte:  (N,) times
        event:(N,) 1 if event, 0 if censored
        """
        # sort by increasing time so risk sets are suffixes
        order = torch.argsort(tte, descending=False)
        risk = risk[order]
        event = event[order]

        # logsumexp over suffixes via reversed cumsum in the log-space
        # For each i, riskset R_i = {j: t_j >= t_i} corresponds to indices i..N-1
        lse_rev = torch.logcumsumexp(risk.flip(0), dim=0).flip(0)
        # Partial log-likelihood: sum_{i: event_i=1} (risk_i - logsumexp(riskset_i))
        pll = (risk - lse_rev) * event
        # Normalize by number of events to make scale comparable
        denom = event.sum().clamp_min(1.0)
        return -pll.sum() / denom
    
    def training_step(self, batch, batch_idx):
        out = self.forward(batch["x"])
        y_hat = out["y_hat"]
        loss = self.calculate_loss(y_hat, batch)
        self._log_all(loss, y_hat, batch, subset="train")
        return loss
    
    def validation_step(self, batch, batch_idx):
        out = self.forward(batch["x"])
        y_hat = out["y_hat"]
        loss = self.calculate_loss(y_hat, batch)
        self._log_all(loss, y_hat, batch, subset="val")
        
    def _log_all(self, loss, y_hat, batch, subset: str):
        logging_kwargs = dict(on_step=True, on_epoch=True, sync_dist=True, batch_size=batch["x"].shape[0])
        self.log(f"{subset}_loss", loss, **logging_kwargs)

        if self.task == "regression":
            y = batch["y"].view(-1, 1).to(y_hat.device, dtype=torch.float32)
            self.__getattr__(f"{subset}_mse")(y_hat.reshape(-1), y.reshape(-1))
            self.log(f"{subset}_mse", self.__getattr__(f"{subset}_mse"), **logging_kwargs)

            self.__getattr__(f"{subset}_concordance")(y_hat.reshape(-1), y.reshape(-1))
            self.log(f"{subset}_concordance", self.__getattr__(f"{subset}_concordance"), **logging_kwargs)

            self.__getattr__(f"{subset}_pearson")(y_hat.reshape(-1), y.reshape(-1))
            self.log(f"{subset}_pearson", self.__getattr__(f"{subset}_pearson"), **logging_kwargs)

        elif self.task == "binary":
            y = batch["y"].view(-1, 1).to(y_hat.device, dtype=torch.float32)
            p = torch.sigmoid(y_hat.detach())
            # torchmetrics expects preds as probabilities for AUROC/AP; labels as int
            y_int = (y >= 0.5).to(torch.int)
            self.__getattr__(f"{subset}_auroc").update(p, y_int)
            self.__getattr__(f"{subset}_ap").update(p, y_int)
            self.__getattr__(f"{subset}_acc").update((p >= 0.5).to(torch.int), y_int)

            self.log(f"{subset}_auroc", self.__getattr__(f"{subset}_auroc"), **logging_kwargs)
            self.log(f"{subset}_ap",    self.__getattr__(f"{subset}_ap"),    **logging_kwargs)
            self.log(f"{subset}_acc",   self.__getattr__(f"{subset}_acc"),   **logging_kwargs)

        else:
            # survival: log extra context if useful
            try:
                event = batch["censor"].float() if self.censor_is_event else (1.0 - batch["censor"].float())
                self.log(f"{subset}_events", event.mean(), **logging_kwargs)
            except Exception:
                pass

    # ---------- predict ----------

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        output = self.forward(batch['x'])
        y_hat, emb = output['y_hat'], output['emb']
        emb_output_filenames = batch["output_visual_embedding_path"]
        # Save embeddings
        if not os.path.exists(os.path.dirname(emb_output_filenames[0])):
            os.makedirs(os.path.dirname(emb_output_filenames[0]))
        for i, emb_output_filename in enumerate(emb_output_filenames):
            torch.save(emb[i].cpu(), emb_output_filename)
        # Save predictions
        for i, y_hat_i in enumerate(y_hat):
            base = os.path.join(self.preds_output_dir, batch["split"][i], f"{batch['case_id'][i]}")
            os.makedirs(os.path.dirname(base), exist_ok=True)
            if self.task == "binary":
                torch.save(y_hat_i.detach().cpu(), base + ".logits.pt")
                torch.save(torch.sigmoid(y_hat_i).detach().cpu(), base + ".probs.pt")
            else:
                # regression or survival risk
                torch.save(y_hat_i.detach().cpu(), base + ".pt")
                
    # ---------- optim ----------
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



