# Copyright 2026
#
# Additive worker variants for disaggregated online RSFT. The default TextArena
# GRPO/RSFT paths keep using verl.workers.fsdp_workers.

from __future__ import annotations

import logging

import torch
from torch.distributed.tensor import DTensor

from verl.checkpoint_engine import CheckpointEngineRegistry
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_id, set_expandable_segments
from verl.utils.fsdp_utils import get_fsdp_full_state_dict
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.model import convert_weight_keys
from verl.utils.profiler import log_gpu_memory_usage
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

logger = logging.getLogger(__file__)
logger.setLevel(logging.getLogger("verl.workers.fsdp_workers").level)


class DisaggregatedAsyncActorWorker(AsyncActorRolloutRefWorker):
    """FSDP actor worker that sends weights to standalone rollout replicas.

    The legacy AsyncActorRolloutRefWorker assumes the rollout engine is hybrid
    and lives inside the same worker. In disaggregated RSFT the worker is actor
    only, so weight sync must go through verl's checkpoint engine.
    """

    def _build_zero1_optimizer_and_scheduler(self, actor_module, optim_config):
        import importlib

        from torch.distributed.optim import ZeroRedundancyOptimizer
        from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

        optimizer_args = {
            "lr": optim_config.lr,
            "weight_decay": optim_config.weight_decay,
        }
        optimizer_name_lower = optim_config.optimizer.lower()
        if "adam" in optimizer_name_lower or "ademamix" in optimizer_name_lower:
            optimizer_args["betas"] = optim_config.betas
        if optim_config.override_optimizer_config is not None:
            optimizer_args.update(optim_config.override_optimizer_config)

        module = importlib.import_module(optim_config.optimizer_impl)
        optimizer_cls = getattr(module, optim_config.optimizer)
        actor_optimizer = ZeroRedundancyOptimizer(
            actor_module.parameters(),
            optimizer_class=optimizer_cls,
            **optimizer_args,
        )

        total_steps = optim_config.get("total_training_steps", 0)
        num_warmup_steps = int(optim_config.get("lr_warmup_steps", -1))
        if num_warmup_steps < 0:
            num_warmup_steps_ratio = optim_config.get("lr_warmup_steps_ratio", 0.0)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        lr_scheduler_type = optim_config.get("lr_scheduler_type", "constant")
        if lr_scheduler_type == "constant":
            actor_lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=actor_optimizer,
                num_warmup_steps=num_warmup_steps,
            )
        elif lr_scheduler_type == "cosine":
            actor_lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=actor_optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
                min_lr_ratio=optim_config.get("min_lr_ratio", 0.0),
                num_cycles=optim_config.get("num_cycles", 0.5),
            )
        else:
            raise NotImplementedError(f"LR scheduler type {lr_scheduler_type} is not supported")

        if self.rank == 0:
            print("[DisaggregatedAsyncActorWorker] trainer optimizer=ZeroRedundancyOptimizer stage=1")
            print(f"[DisaggregatedAsyncActorWorker] total_steps={total_steps} warmup_steps={num_warmup_steps}")
        return actor_optimizer, actor_lr_scheduler

    def _build_model_optimizer(self, *args, **kwargs):
        role = kwargs.get("role", "actor")
        optim_config = args[2] if len(args) > 2 else kwargs.get("optim_config")
        if role != "actor" or optim_config is None:
            return super()._build_model_optimizer(*args, **kwargs)
        if self.config.actor.strategy != "fsdp":
            raise NotImplementedError("ZeRO-1 disaggregated RSFT currently supports actor.strategy=fsdp")

        from torch.distributed.fsdp import ShardingStrategy

        import verl.workers.fsdp_workers as fsdp_workers

        original_get_sharding_strategy = fsdp_workers.get_sharding_strategy
        fsdp_workers.get_sharding_strategy = lambda device_mesh, zero3_enable=True: ShardingStrategy.NO_SHARD
        try:
            actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config = super()._build_model_optimizer(
                *args, **kwargs
            )
        finally:
            fsdp_workers.get_sharding_strategy = original_get_sharding_strategy

        del actor_optimizer, actor_lr_scheduler
        actor_optimizer, actor_lr_scheduler = self._build_zero1_optimizer_and_scheduler(actor_module_fsdp, optim_config)
        aggressive_empty_cache(force_sync=True)
        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if not self._is_actor:
            return

        checkpoint_engine_config = omega_conf_to_dataclass(self.config.rollout.checkpoint_engine)
        backend = checkpoint_engine_config.backend
        if backend == "naive":
            raise ValueError("DisaggregatedAsyncActorWorker requires a non-naive checkpoint engine backend")

        bucket_size = checkpoint_engine_config.update_weights_bucket_megabytes << 20
        engine_kwargs = checkpoint_engine_config.engine_kwargs.get(backend, {})
        self.checkpoint_engine = CheckpointEngineRegistry.new(
            backend,
            is_master=(torch.distributed.get_rank() == 0),
            bucket_size=bucket_size,
            **engine_kwargs,
        )

    def _iter_full_state_dict_for_rollout(self):
        state_dict = get_fsdp_full_state_dict(self.actor_module_fsdp, offload_to_cpu=False, rank0_only=True)
        module = getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        state_dict = convert_weight_keys(state_dict, module)
        device = get_device_id()

        try:
            for name, param in state_dict.items():
                if isinstance(param, DTensor):
                    param = param.to(device, non_blocking=True).full_tensor()
                elif isinstance(param, torch.Tensor) and not param.is_cuda:
                    param = param.to(device, non_blocking=True)
                yield name, param
        finally:
            del state_dict

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None):
        assert self._is_actor
        if not hasattr(self, "checkpoint_engine"):
            raise RuntimeError("checkpoint_engine is not initialized; did init_model() run?")

        if self._is_offload_param:
            from verl.utils.fsdp_utils import load_fsdp_model_to_gpu

            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        set_expandable_segments(False)
        log_gpu_memory_usage("Before disaggregated FSDP weight sync", logger=logger)
        weights = self._iter_full_state_dict_for_rollout()
        try:
            await self.checkpoint_engine.send_weights(weights)
        finally:
            del weights
            aggressive_empty_cache(force_sync=True)
            set_expandable_segments(True)

        if self._is_offload_param:
            from verl.utils.fsdp_utils import offload_fsdp_model_to_cpu

            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        log_gpu_memory_usage("After disaggregated FSDP weight sync", logger=logger)
        return True

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def execute_checkpoint_engine(self, method: str, *args, **kwargs):
        return getattr(self.checkpoint_engine, method)(*args, **kwargs)
