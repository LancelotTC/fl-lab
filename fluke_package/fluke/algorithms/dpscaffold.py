"""Implementation of a Differentially Private SCAFFOLD algorithm.

This module combines SCAFFOLD control variates with Opacus-based DP-SGD,
following the same DP integration approach used by :mod:`fluke.algorithms.dpfedavg`.
"""

from __future__ import annotations

import sys
from copy import deepcopy
from typing import Sequence

import torch
from opacus import PrivacyEngine
from torch.nn import Module

sys.path.append(".")
sys.path.append("..")

from ..client import Client  # NOQA
from ..config import OptimizerConfigurator  # NOQA
from ..data import FastDataLoader  # NOQA
from ..server import Server  # NOQA
from ..utils.model import safe_load_state_dict, state_dict_zero_like  # NOQA
from .scaffold import SCAFFOLD, SCAFFOLDClient, SCAFFOLDServer  # NOQA

__all__ = ["DPSCAFFOLD", "DPSCAFFOLDClient", "DPSCAFFOLDServer"]


class _OpacusModelAdapter(Module):
    """Adapt a model to be compatible with Opacus wrappers and state dict keys."""

    def __init__(self, model: Module):
        super().__init__()
        self._module = model

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self._module(*args, **kwargs)


class DPSCAFFOLDClient(SCAFFOLDClient):
    def __init__(
        self,
        index: int,
        train_set: FastDataLoader,
        test_set: FastDataLoader,
        optimizer_cfg: OptimizerConfigurator,
        loss_fn: Module,
        local_epochs: int = 3,
        fine_tuning_epochs: int = 0,
        clipping: float = 0,
        noise_mul: float = 1.1,
        max_grad_norm: float = 1.0,
        target_epsilon: float | None = None,
        target_delta: float | None = None,
        dp_total_epochs: int | None = None,
        **kwargs,
    ):
        super().__init__(
            index=index,
            train_set=train_set,
            test_set=test_set,
            optimizer_cfg=optimizer_cfg,
            loss_fn=loss_fn,
            local_epochs=local_epochs,
            fine_tuning_epochs=fine_tuning_epochs,
            clipping=clipping,
            **kwargs,
        )
        self.hyper_params.update(noise_mul=noise_mul, max_grad_norm=max_grad_norm)
        if target_epsilon is not None:
            self.hyper_params.target_epsilon = target_epsilon
        if target_delta is not None:
            self.hyper_params.target_delta = target_delta
        if dp_total_epochs is not None:
            self.hyper_params.dp_total_epochs = dp_total_epochs

    def _init_private_engine(self) -> None:
        if self.model is None:
            return

        data_loader = (
            self.train_set.as_dataloader()
            if isinstance(self.train_set, FastDataLoader)
            else self.train_set
        )
        self.privacy_engine = PrivacyEngine()
        self.model.train()

        base_model = self.model._module if hasattr(self.model, "_module") else self.model
        target_epsilon = (
            self.hyper_params.target_epsilon if "target_epsilon" in self.hyper_params else None
        )
        if target_epsilon is not None:
            if not hasattr(self.privacy_engine, "make_private_with_epsilon"):
                raise RuntimeError(
                    "Installed Opacus version does not support make_private_with_epsilon(). "
                    "Use noise_mul or upgrade Opacus."
                )

            target_delta = self.hyper_params.target_delta if "target_delta" in self.hyper_params else 1e-5
            dp_total_epochs = (
                self.hyper_params.dp_total_epochs
                if "dp_total_epochs" in self.hyper_params
                else self.hyper_params.local_epochs
            )
            self.model, self.optimizer, self.train_set = self.privacy_engine.make_private_with_epsilon(
                module=base_model,
                optimizer=self.optimizer,
                data_loader=data_loader,
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                epochs=dp_total_epochs,
                max_grad_norm=self.hyper_params.max_grad_norm,
            )
            if hasattr(self.optimizer, "noise_multiplier"):
                self.hyper_params.noise_mul = float(self.optimizer.noise_multiplier)
        else:
            self.model, self.optimizer, self.train_set = self.privacy_engine.make_private(
                module=base_model,
                optimizer=self.optimizer,
                data_loader=data_loader,
                noise_multiplier=self.hyper_params.noise_mul,
                max_grad_norm=self.hyper_params.max_grad_norm,
            )

    def receive_model(self) -> None:
        model = self.channel.receive(self.index, "server", msg_type="model").payload
        self.server_control = self.channel.receive(self.index, "server", msg_type="control").payload
        if self.model is None:
            self.model = model
            if self.control is None:
                self.control = state_dict_zero_like(model.state_dict())
            self.server_model = deepcopy(model.state_dict())
            self.optimizer, self.scheduler = self._optimizer_cfg(self.model)
            self._init_private_engine()
        else:
            safe_load_state_dict(self.model, model.state_dict())
            self.server_model = deepcopy(model.state_dict())

    def current_epsilon(self) -> float | None:
        if not hasattr(self, "privacy_engine"):
            return None
        if "target_delta" not in self.hyper_params:
            return None
        try:
            return float(self.privacy_engine.get_epsilon(delta=self.hyper_params.target_delta))
        except Exception:
            return None


class DPSCAFFOLDServer(SCAFFOLDServer):
    def __init__(
        self,
        model: Module,
        test_set: FastDataLoader,
        clients: Sequence[Client],
        weighted: bool = True,
        global_step: float = 1.0,
        lr: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            model=_OpacusModelAdapter(model),
            test_set=test_set,
            clients=clients,
            weighted=weighted,
            global_step=global_step,
            lr=lr,
            **kwargs,
        )


class DPSCAFFOLD(SCAFFOLD):
    def get_client_class(self) -> type[Client]:
        return DPSCAFFOLDClient

    def get_server_class(self) -> type[Server]:
        return DPSCAFFOLDServer
