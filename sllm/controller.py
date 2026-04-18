# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2024                                       #
#                                                                              #
#  licensed under the apache license, version 2.0 (the "license");             #
#  you may not use this file except in compliance with the license.            #
#                                                                              #
#  you may obtain a copy of the license at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/license-2.0                  #
#                                                                              #
#  unless required by applicable law or agreed to in writing, software         #
#  distributed under the license is distributed on an "as is" basis,           #
#  without warranties or conditions of any kind, either express or implied.    #
#  see the license for the specific language governing permissions and         #
#  limitations under the license.                                              #
# ---------------------------------------------------------------------------- #
import asyncio
import os
import uuid
from typing import Any, Dict, List, Mapping, Optional
import time

import ray

from sllm.logger import init_logger
from sllm.loading_perf_profile import (
    load_loading_perf_profile,
    solve_lazy_load_method,
)
from sllm.routers import MigrationRouter, RoundRobinRouter
from sllm.schedulers import FcfsScheduler
from sllm.utils import TokenizerWrapper

logger = init_logger(__name__)


class SllmController:
    def __init__(self, config: Optional[Mapping] = None):
        self.config = dict(config or {})
        self.loading_perf_profile = load_loading_perf_profile()

        self.running_lock = asyncio.Lock()
        self.running = False

        self.metadata_lock = asyncio.Lock()
        self.gpu_request_routers = {}
        self.cpu_request_routers = {} # 每个model在CPU的router中只有一个instance
        # Register model info
        self.registered_models = {}
        self.tokenizers: Dict[str, TokenizerWrapper] = {}

    def _control_node_resources(self) -> Dict[str, float]:
        control_node_fraction = float(
            self.config.get("control_node_fraction", 0.1)
        )
        return {"control_node": control_node_fraction}

    def _generate_lazy_load_method(
        self, model_name: str, input_length: int
    ) -> List[Any]:
        model_config = self.registered_models.get(model_name, {})
        backend_config = dict(model_config.get("backend_config", {}))
        solved = solve_lazy_load_method(
            self.loading_perf_profile,
            model_name,
            input_length,
            backend_config,
            model_config,
        )
        return solved

    async def start(self):
        async with self.running_lock:
            if self.running:
                logger.error("Controller already started")
                raise RuntimeError("Controller already started")
            self.running = True

        logger.info("Starting scheduler")
        ray_scheduler_cls = ray.remote(FcfsScheduler)

        enable_migration = self.config.get("enable_migration", False)
        if enable_migration:
            self.router_cls = ray.remote(MigrationRouter)
        else:
            self.router_cls = ray.remote(RoundRobinRouter)
        self.cpu_router_cls = ray.remote(RoundRobinRouter) 

        self.scheduler = ray_scheduler_cls.options(
            name="model_loading_scheduler",
            resources=self._control_node_resources(),
            max_concurrency=100,
        ).remote(self.config.get("scheduler_config", {}))
        self.scheduler.start.remote()

    async def register(self, model_config):
        if not self.running:
            logger.error("Controller not running")
            return
        model_name = model_config.get("model")
        backend = model_config.get("backend", None)
        if backend is None:
            logger.error(f"Backend not specified for model {model_name}")
            return
        backend_config = model_config.get("backend_config", {})
        auto_scaling_config = model_config.get("auto_scaling_config", None)
        async with self.metadata_lock:
            if model_name in self.registered_models:
                logger.info(f"Model {model_name} already registered")
                return

        logger.info(f"Registering new model {model_name}")

        self.tokenizers[model_name] = TokenizerWrapper(model_name, trust_remote_code=True)

        # TODO: put resource requirements in model_config
        cpu_resource_requirements = {
            "num_cpus": 1,
            "num_gpus": 0,
        }
        cpu_request_router = self.cpu_router_cls.options(   #register时直接创建cpu_router和一个未加载权重的gpu_router
            name=model_name,
            namespace="cpu_models",
            num_cpus=1,
            resources={"control_node": 0.1},
        ).remote(
            model_name,
            cpu_resource_requirements,
            backend,
            backend_config,
            "cpu"
        )

        backend_config["lazy_load"] = True
        gpu_resource_requirements = {
            "num_cpus": 1,
            "num_gpus": model_config.get("num_gpus", 0),
        }
        gpu_request_router = self.router_cls.options(   #register时直接创建cpu_router和一个未加载权重的gpu_router
            name=model_name,
            namespace="gpu_models",
            num_cpus=1,
            resources={"control_node": 0.1},
        ).remote(
            model_name,
            gpu_resource_requirements,
            backend,
            backend_config,
            "gpu"
        )

        async with self.metadata_lock:
            if model_name in self.cpu_request_routers:
                logger.error(f"Model {model_name} already registered")
                return

        cpu_request_router.start.remote(auto_scaling_config=None)
        gpu_request_router.start.remote(auto_scaling_config)

        logger.info(f"Model {model_name} registered")

        # Mark model as registered only after model registered successfully
        async with self.metadata_lock:
            self.registered_models[model_name] = model_config
            self.cpu_request_routers[model_name] = cpu_request_router
            self.gpu_request_routers[model_name] = gpu_request_router

    async def generate_stream(self, model_name: str, request_data: Dict[str, Any]):
        logger.info(f"Received request for model {model_name}")

        async with self.metadata_lock:
            if model_name not in self.cpu_request_routers or model_name not in self.gpu_request_routers:
                raise ValueError(f"Model {model_name} not found")
            cpu_router = self.cpu_request_routers[model_name]
            gpu_router = self.gpu_request_routers[model_name]

        logger.info(f"Got request router for {model_name}")

        model_cfg = self.registered_models.get(model_name, {})
        pp_size = int(
            model_cfg.get(
                "pipeline_parallel_size",
                model_cfg.get("backend_config", {}).get("pipeline_parallel_size", 1),
            )
            or 1
        )
        is_pp_model = pp_size > 1

        input_length = self.tokenizers[model_name].get_prompt_len(request_data["prompt"])
        request_data["input_length"] = input_length

        # ---- Hot path: route to loaded GPU instance when capacity is available ---- #
        force_amx_assisted_cold_start = False
        pool_status = await gpu_router.get_instance_pool_status.remote()
        loaded_ready = int(pool_status.get("loaded_ready", 0))
        loaded_available = int(pool_status.get("loaded_available", 0))
        empty_ready = int(pool_status.get("empty_ready", 0))
        if loaded_ready > 0:
            if loaded_available > 0:
                result_ref = gpu_router.inference.remote(
                    request_data=request_data, action="generate"
                )
                return await result_ref
            if empty_ready > 0:
                force_amx_assisted_cold_start = True
            else:
                cpu_result = await cpu_router.inference.remote(
                    request_data=request_data, action="generate"
                )
                await gpu_router.ensure_one_instance.remote()
                return cpu_result

        # ---- Resolve the node hosting this model's GPU instance ---- #
        node_id = await self.scheduler.get_node_for_model.remote(model_name)
        if node_id is None:
            return {"error": "No GPU node allocated for this model"}

        # ---- Check whether a cold start for THIS model is already running ---- #
        is_cold_starting, cold_start_method = (
            await self.scheduler.get_cold_start_status.remote(node_id, model_name)
        )

        if is_cold_starting:
            # Another request already triggered a cold start – piggyback
            # instead of starting a second cold start.
            logger.info(
                f"Cold start already in progress for {model_name} on node "
                f"{node_id} (method={cold_start_method}), piggybacking request"
            )
            if cold_start_method == "tokenwise" or (
                cold_start_method == "layerwise" and is_pp_model
            ):
                result = await gpu_router.inference.remote(
                    request_data=request_data, action="generate"
                )
                return result
            else:
                cpu_router.inference.remote(
                    request_data=request_data, action="generate"
                )
                result = await gpu_router.inference.remote(
                    request_data=request_data, action="generate"
                )
                return result

        # ---- No cold start running for this model – wait for any prior
        #      cold start on the same node (possibly another model) to
        #      release shared resources before starting a new one. ---- #
        await self.scheduler.wait_cold_start_ready.remote(node_id)

        # Cold start logic
        start_time = time.perf_counter()

        # Determine cold-start strategy
        methods = self._generate_lazy_load_method(model_name, input_length)
        cpu_instance = await cpu_router.get_instance.remote()
        if cpu_instance is None or cpu_instance.backend_instance is None:
            return {"error": "CPU instance not available"}
        cpu_backend = cpu_instance.backend_instance

        if methods[0] == "layerwise":
            # Mark cold start in progress (layerwise) at scheduler (node) level
            await self.scheduler.start_cold_start.remote(
                node_id, model_name, "layerwise"
            )
            logger.info(f"Starting cold start for model {model_name} (layerwise)")

            try:
                # 1. Update CPU computing layers
                await cpu_backend.update_computing_layers.remote(
                    computing_layers=methods[1]
                )

                # 2. Start CPU generation (background)
                cpu_task = cpu_backend.generate.remote(request_data=request_data)

                # 3. Wait for GPU weights
                layer_idxes = methods[2]
                await gpu_router.lazy_load_weights.remote(
                    layer_idxes=layer_idxes,
                    load_method=methods[0],
                    request_id=request_data.get("request_id"),
                )

                # 4. Start GPU generation
                # NOTE: For layerwise, the first-token callback in vllm_backend
                # calls router.notify_first_token → scheduler.notify_first_token,
                # which sets the node's ready event so the next cold start on
                # the same node may begin.
                result = await gpu_router.inference.remote(
                    request_data=request_data, action="generate"
                )

                _, metrics = result
                logger.info(
                    f"Cold start finished. E2E: {metrics['e2e']:.4f}s, "
                    f"TTFT: {metrics.get('ttft', 'N/A')}, "
                    f"TPOT: {metrics.get('tpot', 'N/A')}"
                )
                return result
            finally:
                await self.scheduler.finish_cold_start.remote(
                    node_id, model_name
                )

        else:
            # Tokenwise cold start
            await self.scheduler.start_cold_start.remote(
                node_id, model_name, "tokenwise"
            )
            logger.info(f"Starting cold start for model {model_name} (tokenwise)")

            try:
                start_time = time.perf_counter()
                cpu_compute_tokens = methods[1]
                cpu_request_data = request_data.copy()
                cpu_request_data.setdefault("extra_args", {})
                cpu_request_data["extra_args"]["kv_transfer_params"] = {
                    "do_remote_decode": True,
                    "do_remote_prefill": False,
                    "max_num_prefill_compute_tokens": cpu_compute_tokens,
                }
                cpu_task = cpu_backend.generate.remote(request_data=cpu_request_data)

                layer_idxes = methods[2]
                gpu_load_task = gpu_router.lazy_load_weights.remote(
                    layer_idxes=layer_idxes,
                    load_method=methods[0],
                    request_id=request_data.get("request_id"),
                )

                cpu_result = await cpu_task
                # CPU computation done → tokenwise readiness criteria met.
                # Signal at the node level so that the next cold start may begin.
                await self.scheduler.signal_cold_start_ready.remote(node_id)

                await gpu_load_task

                gpu_request_data = request_data.copy()
                gpu_request_data.setdefault("extra_args", {})
                gpu_request_data["prompt"] += cpu_result[0]["choices"][0]["text"]
                gpu_request_data["extra_args"]["kv_transfer_params"] = {
                    "do_remote_decode": False,
                    "do_remote_prefill": True,
                    "num_computed_tokens": cpu_compute_tokens,
                }
                gpu_result = await gpu_router.inference.remote(
                    request_data=gpu_request_data, action="generate"
                )

                end_time = time.perf_counter()
                gpu_result[1]["e2e"] = end_time - start_time
                gpu_result[1]["tpot"] = (
                    (gpu_result[1]["e2e"] - gpu_result[1]["ttft"])
                    / (gpu_result[1]["output_length"] - 1)
                )
                if cpu_compute_tokens >= input_length:
                    gpu_result[1]["ttft"] = cpu_result[1]["ttft"]
                    gpu_result[1]["output_length"] += cpu_result[1]["output_length"]
                    gpu_result[1]["itls"] = (
                        cpu_result[1]["itls"] + gpu_result[1]["itls"]
                    )
                else:
                    gpu_result[1]["ttft"] = (
                        gpu_result[1]["first_token_time"] - start_time
                    )

                _, metrics = gpu_result
                logger.info(
                    f"Cold start finished. E2E: {metrics['e2e']:.4f}s, "
                    f"TTFT: {metrics.get('ttft', 'N/A')}, "
                    f"TPOT: {metrics.get('tpot', 'N/A')}"
                )
                return gpu_result
            finally:
                await self.scheduler.finish_cold_start.remote(
                    node_id, model_name
                )

    async def update(self, model_name: str, model_config: Mapping):
        async with self.metadata_lock:
            if (
                model_name not in self.registered_models
                or model_name not in self.gpu_request_routers
                or model_name not in self.cpu_request_routers
            ):
                logger.error(f"Model {model_name} not found")
                raise ValueError(
                    f"Model {model_name} not found, please register first"
                )

        # update auto-scaling config
        auto_scaling_config = model_config.get("auto_scaling_config", None)
        logger.info(f"Try to update the model {model_name} config")
        if auto_scaling_config is not None:
            async with self.metadata_lock:
                self.registered_models[model_name]["auto_scaling_config"] = (
                    auto_scaling_config
                )
                request_router = self.gpu_request_routers[model_name]
            await request_router.update.remote(auto_scaling_config)
        # TODO: update other config (if possible)

    async def exists(self, model_name: str):
        async with self.metadata_lock:
            return model_name in self.registered_models

    async def delete(self, model_name: str):
        async with self.metadata_lock:
            if model_name not in self.gpu_request_routers or \
                    model_name not in self.cpu_request_routers:
                logger.error(f"Model {model_name} not found")
                return
            if model_name in self.cpu_request_routers:
                cpu_router = self.cpu_request_routers.pop(model_name)
                await cpu_router.shutdown.remote()
                del cpu_router
            if model_name in self.gpu_request_routers:
                gpu_router = self.gpu_request_routers.pop(model_name)
                await gpu_router.shutdown.remote()
                del gpu_router
            self.registered_models.pop(model_name, None)
            self.tokenizers.pop(model_name, None)
        logger.info(f"Model {model_name} deleted")

    async def get_models(self):
        async with self.metadata_lock:
            return self.registered_models

    async def status(self):
        """
        Returns the status of all registered models in OpenAI-compliant format.
        """
        async with self.metadata_lock:
            models = []
            model_folder = os.getenv("MODEL_FOLDER")
            for model_name, config in self.registered_models.items():
                # Extract or calculate relevant fields
                model_path = config.get("_name_or_path", None)
                created_time = (
                    next(
                        (
                            int(os.path.getctime(os.path.abspath(dirpath)))
                            for dirpath, _, _ in os.walk(model_folder)
                            if dirpath.endswith(model_path)
                        ),
                        None,
                    )
                    if model_path
                    else None
                )

                created_time = config.get("created", None)
                allow_create_engine = config.get("allow_create_engine", None)
                allow_sampling = config.get("allow_sampling", None)
                allow_logprobs = config.get("allow_logprobs", None)
                allow_search_indices = config.get("allow_search_indices", None)
                allow_view = config.get("allow_view", None)
                organization = config.get("organization", "*")
                group = config.get("group", None)
                is_blocking = config.get("is_blocking", None)

                max_model_len = config.get("max_position_embeddings", None)

                model_permission_id = f"modelperm-{model_name}"
                permission = [
                    {
                        "id": model_permission_id,
                        "object": "model_permission",
                        "created": created_time
                        if created_time is not None
                        else None,
                        "allow_create_engine": allow_create_engine
                        if allow_create_engine is not None
                        else None,
                        "allow_sampling": allow_sampling
                        if allow_sampling is not None
                        else None,
                        "allow_logprobs": allow_logprobs
                        if allow_logprobs is not None
                        else None,
                        "allow_search_indices": allow_search_indices
                        if allow_search_indices is not None
                        else None,
                        "allow_view": allow_view
                        if allow_view is not None
                        else None,
                        "organization": organization
                        if organization is not None
                        else None,
                        "group": group if group is not None else None,
                        "is_blocking": is_blocking
                        if is_blocking is not None
                        else None,
                    }
                ]

                # Build the model metadata entry
                model_metadata = {
                    "id": model_name,
                    "object": "model",
                    "created": created_time
                    if created_time is not None
                    else None,
                    "owned_by": "sllm",
                    "root": model_name,
                    "parent": None,
                    "max_model_len": max_model_len
                    if max_model_len is not None
                    else None,
                    "permission": permission,
                }
                models.append(model_metadata)

            return {"object": "list", "models": models}

    async def shutdown(self):
        # stop the control loop
        async with self.running_lock:
            if not self.running:
                logger.error("Controller not running")
                raise RuntimeError("Controller not running")
            self.running = False

        async with self.metadata_lock:
            model_names = list(self.cpu_request_routers.keys())
        await asyncio.gather(*[self.delete(m) for m in model_names])
