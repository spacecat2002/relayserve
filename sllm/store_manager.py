# ---------------------------------------------------------------------------- #
#  ServerlessLLM                                                               #
#  Copyright (c) ServerlessLLM Team 2024                                       #
#                                                                              #
#  Licensed under the Apache License, Version 2.0 (the "License");             #
#  you may not use this file except in compliance with the License.            #
#                                                                              #
#  You may obtain a copy of the License at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/LICENSE-2.0                  #
#                                                                              #
#  Unless required by applicable law or agreed to in writing, software         #
#  distributed under the License is distributed on an "AS IS" BASIS,           #
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
#  See the License for the specific language governing permissions and         #
#  limitations under the License.                                              #
# ---------------------------------------------------------------------------- #

import asyncio
import os
import time
from typing import List, Mapping, Optional

import ray

from sllm.logger import init_logger
from sllm.model_downloader import (
    VllmModelDownloader,
)
from sllm.utils import get_worker_nodes

logger = init_logger(__name__)


class NodeManager:
    def __init__(
        self,
        node_id: str,
    ):
        self.node_id = node_id
        self.registered_models = {}
        self.lock = asyncio.Lock()

        logger.info(
            f"Initialized local model info for worker node {self.node_id}"
        )

    async def get_model_shard_size(self, model_rank_path: str):
        data_idx = 0
        shard_size = 0
        while True:
            shard_path = os.path.join(model_rank_path, f"tensor.data_{data_idx}")
            if not os.path.exists(shard_path):
                break
            shard_size += os.path.getsize(shard_path)
            data_idx += 1
        return shard_size

    async def register_model(
        self, model_name: str, backend: str, backend_config: dict
    ):
        async with self.lock:
            if model_name in self.registered_models:
                logger.error(f"{model_name} already registered")
                return self.registered_models[model_name]
            model_path = model_name
            if backend == "vllm":
                tensor_parallel_size = backend_config.get(
                    "tensor_parallel_size", 1
                )
                model_size = 0
                for rank in range(tensor_parallel_size):
                    model_rank_path = os.path.join(model_path, f"rank_{rank}")
                    model_size += await self.get_model_shard_size(model_rank_path)
                self.registered_models[model_name] = model_size
            logger.info(f"{model_name} registered, {self.registered_models}")
            return model_size


# @ray.remote(num_cpus=1, resources={"control_node": 0.1})
class StoreManager:
    def __init__(self):
        logger.info("Initializing store manager")
        self.metadata_lock = asyncio.Lock()
        # Storage info
        self.round_robin_index = 0
        self.local_servers = {}
        self.model_info = {}
        self.model_storage_info = {}

    async def initialize_cluster(self) -> bool:
        logger.info("Initializing cluster and collecting hardware info")

        # Get worker nodes
        worker_node_info = get_worker_nodes()
        if not worker_node_info:
            logger.error("No worker nodes found")
            return False

        uninitialized_nodes = list(worker_node_info.keys())

        while len(uninitialized_nodes) > 0:
            for node_id in uninitialized_nodes:
                self.local_servers[node_id] = NodeManager(node_id)
                uninitialized_nodes.remove(node_id)
                logger.info(f"Node {node_id} initialized")
                break

        return True

    async def get_model_info(self, model_name: Optional[str] = None):
        logger.info(f"Getting info for {model_name}")
        async with self.metadata_lock:
            if model_name is not None:
                return self.model_info.get(model_name, {})
            else:
                return self.model_info

    async def register(self, model_config):
        model_name = model_config.get("model")
        backend = model_config.get("backend", None)
        if backend is None:
            logger.error(f"Backend not specified for {model_name}")
            return
        backend_config = model_config.get("backend_config", {})
        if model_name not in self.model_info:
            self.model_storage_info[model_name] = {}
            logger.info(f"Registering new {model_name}")

            backend = model_config.get("backend", None)
            pretrained_model_name_or_path = backend_config.get(
                "pretrained_model_name_or_path", None
            )
            # 1. download this model to one worker using round-robin
            worker_node_info = get_worker_nodes()

            n_nodes = len(worker_node_info)
            assert n_nodes > 0, "No worker nodes found"

            for node_id in worker_node_info.keys():
                if node_id not in self.local_servers:
                    if self.local_servers:
                        self.local_servers[node_id] = NodeManager(node_id)
                    else:
                        logger.error(f"Node {node_id} not found")
                        raise ValueError(f"Node {node_id} not found")


            node_id = list(worker_node_info.keys())[
                self.round_robin_index % n_nodes
            ]
            self.round_robin_index += 1

            logger.info(
                f"Registering model {pretrained_model_name_or_path} to nodes {node_id}"  # noqa: E501
            )
            if backend == "vllm":
                await self.download_vllm_model(
                    model_name,
                    pretrained_model_name_or_path,
                    node_id,
                    model_config.get("num_gpus", 1),
                    backend_config.get("tensor_parallel_size", 1),
                    backend_config.get("torch_dtype", "bfloat16"),
                )
                local_server = self.local_servers[node_id]
                model_size = await local_server.register_model(
                    model_name, backend, backend_config
                )
                # record the storage info
                self.model_storage_info[model_name][node_id] = True
                logger.info(f"{model_name} registered to node {node_id}")
                self.model_info[model_name] = model_size
                logger.info(f"{model_name} registered")
            else:
                logger.error(f"Backend {backend} not supported")

    async def download_vllm_model(
        self,
        model_name,
        pretrained_model_name_or_path,
        node_id,
        num_gpus,
        tensor_parallel_size,
        torch_dtype,
    ):
        logger.info(
            f"Downloading {pretrained_model_name_or_path} to node {node_id}"
        )
        vllm_backend_downloader = (
            ray.remote(VllmModelDownloader)
            .options(
                num_gpus=num_gpus,
                resources={"worker_node": 0.1, f"worker_id_{node_id}": 0.1},
            )
            .remote()
        )
        return await vllm_backend_downloader.download_vllm_model.remote(
            model_name,
            pretrained_model_name_or_path,
            torch_dtype,
            tensor_parallel_size,
        )
