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
from abc import ABC, abstractmethod
from typing import Dict, Mapping, Optional
import os
import json
import subprocess

from sllm.logger import init_logger

logger = init_logger(__name__)


class SllmScheduler(ABC):
    @abstractmethod
    def __init__(self, scheduler_config: Optional[Mapping] = None):
        super().__init__()

    @abstractmethod
    async def start(self) -> None:
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        pass

    @abstractmethod
    async def allocate_resource(
        self, model_name: str, instance_id: str, resources: Mapping
    ):
        pass

    @abstractmethod
    async def deallocate_resource(
        self, model_name: str, instance_id: str, resources: Mapping
    ):
        pass


# ---------------------------------------------------------------------------
# Low-level detection helpers
# ---------------------------------------------------------------------------

def detect_gpu_numa_affinity() -> Dict[int, int]:
    """Auto-detect GPU-to-NUMA affinity from sysfs.

    Scans ``/sys/bus/pci/devices`` for NVIDIA GPUs (vendor ``0x10de``,
    class ``0x0300xx`` or ``0x0302xx``) and reads the ``numa_node`` file.

    Returns:
        Mapping ``{gpu_id: numa_node_id}``.  Empty dict on failure.
    """
    affinity = _detect_via_sysfs()
    if not affinity:
        affinity = _detect_via_nvidia_smi()
    if affinity:
        logger.info(f"Auto-detected GPU NUMA affinity: {affinity}")
    else:
        logger.warning(
            "Could not auto-detect GPU NUMA affinity. "
            "Set SLLM_GPU_NUMA_AFFINITY or scheduler_config['gpu_numa_affinity'] explicitly."
        )
    return affinity


def _detect_via_sysfs() -> Dict[int, int]:
    """Read NUMA affinity from ``/sys/bus/pci/devices/*/numa_node``."""
    affinity: Dict[int, int] = {}
    pci_path = "/sys/bus/pci/devices"
    if not os.path.exists(pci_path):
        return affinity
    try:
        gpu_devices = []
        for device in sorted(os.listdir(pci_path)):
            device_path = os.path.join(pci_path, device)
            class_path = os.path.join(device_path, "class")
            if not os.path.exists(class_path):
                continue
            with open(class_path, "r") as f:
                device_class = f.read().strip()
            # 0x0300xx = VGA controller, 0x0302xx = 3D controller
            if not (device_class.startswith("0x0302") or device_class.startswith("0x0300")):
                continue
            vendor_path = os.path.join(device_path, "vendor")
            if not os.path.exists(vendor_path):
                continue
            with open(vendor_path, "r") as f:
                vendor = f.read().strip()
            if vendor != "0x10de":  # NVIDIA
                continue
            gpu_devices.append(device_path)

        for gpu_id, device_path in enumerate(gpu_devices):
            numa_path = os.path.join(device_path, "numa_node")
            if os.path.exists(numa_path):
                with open(numa_path, "r") as f:
                    numa_node = int(f.read().strip())
                affinity[gpu_id] = max(0, numa_node)  # -1 → default 0
            else:
                affinity[gpu_id] = 0
    except Exception as e:
        logger.warning(f"sysfs NUMA detection failed: {e}")
    return affinity


def _detect_via_nvidia_smi() -> Dict[int, int]:
    """Fallback: detect GPU NUMA affinity via ``nvidia-smi``."""
    affinity: Dict[int, int] = {}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,gpu_bus_id",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return affinity
        for line in result.stdout.strip().split("\n"):
            parts = line.strip().split(",")
            if len(parts) != 2:
                continue
            gpu_idx = int(parts[0].strip())
            bus_id = parts[1].strip().lower()
            pci_numa = f"/sys/bus/pci/devices/{bus_id}/numa_node"
            if os.path.exists(pci_numa):
                with open(pci_numa, "r") as f:
                    numa_node = int(f.read().strip())
                affinity[gpu_idx] = max(0, numa_node)
    except Exception as e:
        logger.warning(f"nvidia-smi NUMA detection failed: {e}")
    return affinity


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------

def load_gpu_numa_affinity(
    config: Optional[Dict] = None,
    env_key: str = "SLLM_GPU_NUMA_AFFINITY",
    auto_detect: bool = True,
) -> Dict[str, Dict[int, int]]:
    """Load GPU-to-NUMA affinity mapping with fallback chain.

    Priority:
      1. Explicit config  (``config["gpu_numa_affinity"]``)
      2. Environment variable  (``SLLM_GPU_NUMA_AFFINITY``)
      3. Auto-detection from sysfs / nvidia-smi  (if *auto_detect* is True)

    For auto-detection the actual node_id is unknown (the scheduler runs on
    the control node, not the worker).  The special key ``"__auto__"`` is
    used as placeholder; the scheduler should match it to real node IDs when
    only one worker exists or broadcast the affinity query to workers.

    Returns:
        ``{node_id_str: {gpu_id_int: numa_node_int}}``
    """
    affinity_raw: Dict = {}

    # 1. config
    if config:
        affinity_raw = config.get("gpu_numa_affinity", {})

    # 2. env var
    if not affinity_raw:
        env_value = os.getenv(env_key, "")
        if env_value:
            try:
                affinity_raw = json.loads(env_value)
            except json.JSONDecodeError:
                logger.error(f"Invalid {env_key} JSON format")

    # 3. auto-detect
    if not affinity_raw and auto_detect:
        detected = detect_gpu_numa_affinity()
        if detected:
            affinity_raw = {"__auto__": {str(k): v for k, v in detected.items()}}

    # normalise types
    parsed: Dict[str, Dict[int, int]] = {}
    for node_id, mapping in affinity_raw.items():
        parsed[str(node_id)] = {int(k): int(v) for k, v in mapping.items()}
    return parsed


def resolve_gpu_numa_affinity(
    worker_nodes: Mapping[str, Mapping],
    affinity: Dict[str, Dict[int, int]],
) -> Dict[str, Dict[int, int]]:
    """Resolve the special ``"__auto__"`` affinity entry to concrete workers.

    The scheduler runs on the control node, so automatic GPU NUMA detection may
    only discover the topology of the local machine and store it under the
    placeholder key ``"__auto__"``.  At runtime we map that template onto worker
    nodes that do not yet have an explicit entry.
    """
    if not affinity:
        return {}

    resolved = {
        str(node_id): {int(gpu_id): int(numa_id) for gpu_id, numa_id in mapping.items()}
        for node_id, mapping in affinity.items()
        if node_id != "__auto__"
    }
    auto_mapping = affinity.get("__auto__", {})
    if not auto_mapping:
        return resolved

    max_gpu_id = max(auto_mapping.keys(), default=-1)
    required_gpu_count = max_gpu_id + 1

    for node_id, node_info in worker_nodes.items():
        if node_id in resolved:
            continue
        total_gpus = int(node_info.get("total_gpu", 0))
        if total_gpus <= 0:
            continue
        if required_gpu_count > total_gpus:
            logger.warning(
                "Skip NUMA auto mapping for node %s: detected template needs %s GPUs, "
                "worker only has %s",
                node_id,
                required_gpu_count,
                total_gpus,
            )
            continue
        resolved[node_id] = {
            gpu_id: numa_id
            for gpu_id, numa_id in auto_mapping.items()
            if gpu_id < total_gpus
        }
    return resolved