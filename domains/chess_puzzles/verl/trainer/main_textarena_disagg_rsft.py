# Copyright 2026
#
# TextArena/chess-puzzle disaggregated trainer/rollout entrypoint.
# This file is intentionally additive so the canonical GRPO/RSFT/MH scripts
# keep their current behavior.

from __future__ import annotations

import os
import signal
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.single_controller.ray import RayClassWithInitArgs, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.utils import Role, need_critic, need_reference_policy, need_reward_model
from verl.utils.config import omega_conf_to_dataclass, validate_config
from verl.utils.device import auto_set_device, is_cuda_available
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import load_class_from_fqn


class DisaggregatedRayTrainer(RayPPOTrainer):
    """RayPPOTrainer with actor-only FSDP workers and standalone vLLM rollout."""

    def init_workers(self):
        if self.config.actor_rollout_ref.rollout.checkpoint_engine.backend == "naive":
            raise ValueError("Disaggregated trainer requires actor_rollout_ref.rollout.checkpoint_engine.backend!=naive")
        if self.use_critic:
            raise ValueError("Disaggregated trainer entrypoint does not support critic workers")

        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        actor_role = Role.ActorRollout
        actor_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
        actor_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[actor_role],
            config=self.config.actor_rollout_ref,
            role=str(Role.Actor),
        )
        self.resource_pool_to_cls[actor_resource_pool][str(actor_role)] = actor_cls

        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            ref_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[ref_resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        all_wg = {}
        wg_kwargs = {"device_name": self.device_name}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            all_wg.update(wg_dict.spawn(prefix_set=class_dict.keys()))

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        from verl.experimental.reward_loop import RewardLoopManager

        rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
        self.reward_loop_manager = RewardLoopManager(config=self.config, rm_resource_pool=rm_resource_pool)

        self.async_rollout_mode = True
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            agent_loop_manager_cls = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager as agent_loop_manager_cls

        enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool
        reward_loop_worker_handles = (
            self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None
        )
        self.async_rollout_manager = agent_loop_manager_cls.create(
            config=self.config,
            worker_group=None,
            rollout_resource_pool=None,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            trainer=self.actor_rollout_wg,
            replicas=self.async_rollout_manager.rollout_replicas,
        )


class DisaggregatedTaskRunner:
    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def ping(self):
        return {"hostname": socket.gethostname(), "pid": os.getpid()}

    def add_actor_worker(self):
        from verl.workers.disagg_fsdp_workers import DisaggregatedAsyncActorWorker

        if self.config.actor_rollout_ref.actor.strategy not in {"fsdp", "fsdp2"}:
            raise NotImplementedError("Disaggregated RSFT currently supports FSDP/FSDP2 actors")
        self.role_worker_mapping[Role.ActorRollout] = ray.remote(DisaggregatedAsyncActorWorker)
        self.mapping[Role.ActorRollout] = "trainer_pool"
        return DisaggregatedAsyncActorWorker

    def add_ref_policy_worker(self, actor_cls):
        if need_reference_policy(self.config):
            self.role_worker_mapping[Role.RefPolicy] = ray.remote(actor_cls)
            self.mapping[Role.RefPolicy] = "trainer_pool"

    def init_resource_pool_mgr(self):
        trainer_pool = [self.config.trainer.n_gpus_per_node] * self.config.trainer.nnodes
        resource_pool_spec = {"trainer_pool": trainer_pool}

        if self.config.reward.reward_model.enable_resource_pool:
            resource_pool_spec["reward_pool"] = [
                self.config.reward.reward_model.n_gpus_per_node
            ] * self.config.reward.reward_model.nnodes
        else:
            self.config.reward.reward_model.nnodes = self.config.trainer.nnodes
            self.config.reward.reward_model.n_gpus_per_node = self.config.trainer.n_gpus_per_node

        return ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)

    def run(self, config):
        from pprint import pprint

        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn

        self.config = config
        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}", flush=True)
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_cls = self.add_actor_worker()
        self.add_ref_policy_worker(actor_cls)

        if need_reward_model(config):
            self.mapping[Role.RewardModel] = (
                "reward_pool" if config.reward.reward_model.enable_resource_pool else "trainer_pool"
            )

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        resource_pool_manager = self.init_resource_pool_mgr()

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = DisaggregatedRayTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    run_disaggregated_rsft(config)


def run_disaggregated_rsft(config) -> None:
    def log(message: str) -> None:
        print(f"[textarena_disagg_rsft] {message}", flush=True)

    def arm_startup_timeout():
        raw = os.environ.get("VERL_TASK_RUNNER_START_TIMEOUT_S", "0")
        try:
            timeout_s = int(raw)
        except ValueError:
            timeout_s = 0
        if timeout_s <= 0 or not hasattr(signal, "SIGALRM"):
            return None

        old_handler = signal.getsignal(signal.SIGALRM)

        def handler(signum, frame):
            raise TimeoutError(f"Timed out after {timeout_s}s while starting DisaggregatedTaskRunner")

        signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout_s)
        return old_handler

    def clear_startup_timeout(old_handler) -> None:
        if old_handler is None or not hasattr(signal, "SIGALRM"):
            return
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        log(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))
        log("ray.init complete")

    startup_timeout_handler = arm_startup_timeout()
    task_runner_class = ray.remote(num_cpus=1)(DisaggregatedTaskRunner)
    try:
        if (
            is_cuda_available
            and config.global_profiler.tool == "nsys"
            and config.global_profiler.get("steps") is not None
            and len(config.global_profiler.get("steps", [])) > 0
        ):
            nsight_options = OmegaConf.to_container(
                config.global_profiler.global_tool_config.nsys.controller_nsight_options
            )
            log("creating DisaggregatedTaskRunner actor with nsys runtime")
            runner = task_runner_class.options(runtime_env={"nsight": nsight_options}).remote()
        else:
            log("creating DisaggregatedTaskRunner actor")
            runner = task_runner_class.remote()

        log("waiting for DisaggregatedTaskRunner actor readiness")
        ready_ref = runner.ping.remote()
        raw_timeout = os.environ.get("VERL_TASK_RUNNER_START_TIMEOUT_S", "0")
        try:
            startup_timeout_s = int(raw_timeout)
        except ValueError:
            startup_timeout_s = 0
        if startup_timeout_s > 0:
            ready = ray.get(ready_ref, timeout=startup_timeout_s)
        else:
            ready = ray.get(ready_ref)
        log(f"DisaggregatedTaskRunner ready: {ready}")

        log("submitting DisaggregatedTaskRunner.run")
        run_ref = runner.run.remote(config)
        log("DisaggregatedTaskRunner.run submitted")
    finally:
        clear_startup_timeout(startup_timeout_handler)

    ray.get(run_ref)

    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


if __name__ == "__main__":
    main()
