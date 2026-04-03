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
"""
MigrationRouter — extends :class:`RoundRobinRouter` with NUMA-aware
instance migration so that empty (pre-warmed) instances can be relocated
to free up GPUs on other NUMA nodes, enabling cross-NUMA tensor-parallel
dispatch.

Migration strategy
------------------
When the scheduler detects that all free GPUs on a worker are concentrated
on a single NUMA node, it suggests migrating an *empty* instance away from
a non-dominant NUMA node **to** the dominant NUMA (which has spare GPUs).
This frees a GPU slot on the other NUMA, making the overall free-GPU
distribution span multiple NUMA nodes — a prerequisite for balanced TP
placement.
"""
import asyncio
from typing import Dict, List, Optional

import ray

from sllm.logger import init_logger

from .roundrobin_router import RoundRobinRouter

logger = init_logger(__name__)


class MigrationRouter(RoundRobinRouter):
    """Router with NUMA-aware instance migration support.

    Overrides the no-op migration hooks in :class:`RoundRobinRouter` with
    real logic that queries the scheduler for a migration plan and executes
    it by tearing down the old actor and recreating it on the target GPUs.
    """

    # ------------------------------------------------------------------
    # Override: rebalance hook
    # ------------------------------------------------------------------

    async def _try_rebalance_for_tp(self) -> None:
        """Ask the scheduler whether an empty instance should be migrated
        so that free GPUs span more NUMA nodes, then execute the migration.
        """
        if self.device != "gpu":
            return

        tp_size = int(self.resource_requirements.get("num_gpus", 1))
        if tp_size <= 1:
            return

        plan = await self.model_loading_scheduler.suggest_instance_migration.remote(
            self.model_name, tp_size
        )

        if not isinstance(plan, dict) or not plan:
            return

        instance_id: Optional[str] = plan.get("instance_id")
        target_gpu_ids: List[int] = plan.get("target_gpu_ids", [])
        if not instance_id or not target_gpu_ids:
            return

        success = await self._migrate_empty_instance(instance_id, target_gpu_ids)
        if success:
            logger.info(
                f"Rebalanced instance {instance_id} to GPUs {target_gpu_ids} "
                f"for TP={tp_size} NUMA balance"
            )

    # ------------------------------------------------------------------
    # Override: actual migration
    # ------------------------------------------------------------------

    async def _migrate_empty_instance(
        self, instance_id: str, target_gpu_ids: List[int]
    ) -> bool:
        """Migrate an empty (pre-warmed, no weights loaded) instance to
        *target_gpu_ids* by destroying the old actor and creating a new one.

        Returns ``True`` on success.
        """
        # ---- 1. Validate & remove from ready pool -----------------------
        async with self.instance_management_lock:
            instance = self.ready_inference_instances.get(instance_id)
            if instance is None:
                logger.warning(f"Migration skipped: instance {instance_id} not found")
                return False
            if not instance.empty_instance:
                logger.warning(f"Migration skipped: instance {instance_id} is not empty")
                return False
            if self.instance_to_load_status.get(instance_id, False):
                logger.warning(
                    f"Migration skipped: instance {instance_id} already loaded weights"
                )
                return False
            if instance.concurrency > 0:
                logger.warning(
                    f"Migration skipped: instance {instance_id} has active requests"
                )
                return False
            # Remove from bookkeeping
            self.ready_inference_instances.pop(instance_id, None)
            self.instance_to_load_status.pop(instance_id, None)

        # ---- 2. Release old GPU lock (if any) ---------------------------
        if (
            self.device == "gpu"
            and instance.gpu_locked
            and instance.gpu_group
            and instance.node_id
        ):
            try:
                await self.model_loading_scheduler.release_gpu_lock.remote(
                    instance.node_id, instance.gpu_group
                )
            except Exception as e:
                logger.error(
                    f"Failed to release GPU lock for migrated instance "
                    f"{instance_id}: {e}"
                )
            instance.gpu_locked = False

        # ---- 3. Tear down old actor -------------------------------------
        try:
            await instance.backend_instance.stop.remote()
            ray.kill(instance.backend_instance)
        except Exception as e:
            logger.error(f"Error tearing down instance {instance_id}: {e}")

        # ---- 4. Deallocate old resources --------------------------------
        if self.device == "gpu":
            try:
                await self.model_loading_scheduler.deallocate_resource.remote(
                    self.model_name,
                    instance_id,
                    self._build_deallocate_resources(instance),
                )
            except Exception as e:
                logger.error(
                    f"Failed to deallocate resources for instance {instance_id}: {e}"
                )

        # ---- 5. Create new instance on target GPUs ----------------------
        await self._create_instance(
            empty_instance=True, preferred_gpu_ids=target_gpu_ids
        )
        logger.info(
            f"Migrated empty instance {instance_id} → preferred GPUs {target_gpu_ids}"
        )
        return True
