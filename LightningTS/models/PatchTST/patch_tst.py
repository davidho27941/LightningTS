# TODO: Create `LoghtningModule` based object for PatchTST.

import torch
import numpy as np
import pandas as pd

import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
import torch.nn.functional as F

from torch import Tensor
from typing import Optional

from .layers import series_decomp
from .backbone import PatchTST_backbone


class PatchTSTModel(nn.Module):
    def __init__(
        self,
        configs,
    ):

        super().__init__()

        c_in = configs["encoder"]["input_size"]
        context_window = configs["common"]["sequence_length"]
        target_window = configs["common"]["predict_length"]

        n_layers = configs["encoder"]["n_layers"]
        n_heads = configs["common"]["n_heads"]
        d_model = configs["common"]["d_model"]
        d_ff = configs["common"]["fcn_dim"]
        dropout = configs["common"]["dropout"]
        fc_dropout = configs["common"]["fc_dropout"]
        head_dropout = configs["common"]["head_dropout"]
        act = configs["common"]["activation"]

        individual = configs["individual"]
        stride = configs["common"]["stride"]
        padding_patch = configs["padding_patch"]
        max_seq_len = configs["max_seq_len"]
        d_k = configs["d_k"]
        d_v = configs["d_v"]
        norm = configs["norm"]
        attn_dropout = configs["attn_dropout"]
        key_padding_mask = configs["key_padding_mask"]
        padding_var = configs["padding_var"]
        attn_mask = configs["attn_mask"]
        res_attention = configs["res_attention"]
        pre_norm = configs["pre_norm"]
        store_attn = configs["store_attn"]
        pe = configs["pe"]
        learn_pe = configs["learn_pe"]
        pretrain_head = configs["pretrain_head"]
        head_type = configs["head_type"]
        verbose = configs["verbose"]

        patch_len = ((context_window - target_window) / stride) + 2

        revin = configs["revin"]
        affine = configs["affine"]
        subtract_last = configs["subtract_last"]

        decomposition = configs["decomposition"]
        kernel_size = configs["common"]["kernel_size"]

        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(
                c_in=c_in,
                context_window=context_window,
                target_window=target_window,
                patch_len=patch_len,
                stride=stride,
                max_seq_len=max_seq_len,
                n_layers=n_layers,
                d_model=d_model,
                n_heads=n_heads,
                d_k=d_k,
                d_v=d_v,
                d_ff=d_ff,
                norm=norm,
                attn_dropout=attn_dropout,
                dropout=dropout,
                act=act,
                key_padding_mask=key_padding_mask,
                padding_var=padding_var,
                attn_mask=attn_mask,
                res_attention=res_attention,
                pre_norm=pre_norm,
                store_attn=store_attn,
                pe=pe,
                learn_pe=learn_pe,
                fc_dropout=fc_dropout,
                head_dropout=head_dropout,
                padding_patch=padding_patch,
                pretrain_head=pretrain_head,
                head_type=head_type,
                individual=individual,
                revin=revin,
                affine=affine,
                subtract_last=subtract_last,
                verbose=verbose,
            )
            self.model_res = PatchTST_backbone(
                c_in=c_in,
                context_window=context_window,
                target_window=target_window,
                patch_len=patch_len,
                stride=stride,
                max_seq_len=max_seq_len,
                n_layers=n_layers,
                d_model=d_model,
                n_heads=n_heads,
                d_k=d_k,
                d_v=d_v,
                d_ff=d_ff,
                norm=norm,
                attn_dropout=attn_dropout,
                dropout=dropout,
                act=act,
                key_padding_mask=key_padding_mask,
                padding_var=padding_var,
                attn_mask=attn_mask,
                res_attention=res_attention,
                pre_norm=pre_norm,
                store_attn=store_attn,
                pe=pe,
                learn_pe=learn_pe,
                fc_dropout=fc_dropout,
                head_dropout=head_dropout,
                padding_patch=padding_patch,
                pretrain_head=pretrain_head,
                head_type=head_type,
                individual=individual,
                revin=revin,
                affine=affine,
                subtract_last=subtract_last,
                verbose=verbose,
            )
        else:
            self.model = PatchTST_backbone(
                c_in=c_in,
                context_window=context_window,
                target_window=target_window,
                patch_len=patch_len,
                stride=stride,
                max_seq_len=max_seq_len,
                n_layers=n_layers,
                d_model=d_model,
                n_heads=n_heads,
                d_k=d_k,
                d_v=d_v,
                d_ff=d_ff,
                norm=norm,
                attn_dropout=attn_dropout,
                dropout=dropout,
                act=act,
                key_padding_mask=key_padding_mask,
                padding_var=padding_var,
                attn_mask=attn_mask,
                res_attention=res_attention,
                pre_norm=pre_norm,
                store_attn=store_attn,
                pe=pe,
                learn_pe=learn_pe,
                fc_dropout=fc_dropout,
                head_dropout=head_dropout,
                padding_patch=padding_patch,
                pretrain_head=pretrain_head,
                head_type=head_type,
                individual=individual,
                revin=revin,
                affine=affine,
                subtract_last=subtract_last,
                verbose=verbose,
            )

    def forward(self, x):  # x: [Batch, Input length, Channel]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(
                0, 2, 1
            )  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0, 2, 1)  # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0, 2, 1)  # x: [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0, 2, 1)  # x: [Batch, Input length, Channel]
        return x


class PatchTST(pl.LightningModule):
    def __init__(self, config, is_fit=True) -> None:
        super().__init__()
        print(self.trainer.state)

        self.features = config["data"]["features"]
        self.targets = config["data"]["targets"]

        self.input_length = config["data"]["series_config"]["input_length"]
        self.predict_length = config["data"]["series_config"]["output_length"]

        self.hparam = config["model"]["hparam"]

        model_struct = config["model"]["structure"]
        
        if is_fit:
            self.configure_criterion()

        pre_config_hparam = dict(
            individual=0,  # boolean
            padding_patch="end",
            revin=1,  # boolean
            affine=0,  # boolean
            subtract_last=0,  # boolean
            decomposition=0,  # boolean
            max_seq_len=1024,
            d_k=None,
            d_v=None,
            norm="BatchNorm",
            attn_dropout=0.0,
            key_padding_mask="auto",
            padding_var=None,
            attn_mask=None,
            res_attention=True,
            pre_norm=False,
            store_attn=False,
            pe="zeros",
            learn_pe=True,
            pretrain_head=False,
            head_type="flatten",
            verbose=False,
        )

        pre_config_hparam = pre_config_hparam | config["model"]["extra"]

        model_config = model_struct | pre_config_hparam

        self.model = PatchTSTModel(model_config)

    def configure_criterion(self) -> None:

        match self.hparam:
            case "L1Loss":
                self.criterion = nn.L1Loss()
            case "MSELoss":
                self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def adjust_result_size(self, truth, pred):
        f_dim = 0 if len(self.targets) > 1 else -1

        pred = pred[:, -self.predict_length :, f_dim:]
        truth = truth[:, -self.predict_length :, f_dim:].to(truth.device)

        return truth, pred

    def training_step(self, batch, batch_idx):
        seq_x, seq_y, _, _, _ = batch

        outputs = self(seq_x)

        truth, pred = self.adjust_result_size(seq_y, outputs)

        loss = self.criterion(pred, truth)

        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        seq_x, seq_y, _, _, _ = batch

        outputs = self(seq_x)

        truth, pred = self.adjust_result_size(seq_y, outputs)

        loss = (
            self.criterion(pred, truth)
            if self.val_criterion is None
            else self.val_criterion(pred, truth)
        )

        self.log("val_loss", loss, on_epoch=True, on_step=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        seq_x, seq_y, _, _, _ = batch

        outputs = self(seq_x)

        truth, pred = self.adjust_result_size(seq_y, outputs)

        loss = self.criterion(pred, truth)

        self.log("test_loss", loss, on_epoch=True, on_step=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        seq_x, seq_y, _, _, ts = batch

        origin_seq_x = seq_x.clone()[:, :, -1:]

        outputs = self(seq_x)

        truth, pred = self.adjust_result_size(seq_y, outputs)

        return pred, truth, origin_seq_x, ts
