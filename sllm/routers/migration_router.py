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
MigrationRouter — NUMA-aware migration for **active** (capacity-committed) instances.

Pure prewarm empties are excluded from NUMA rebalance plans: they do not relieve
uneven **active** load across NUMA that blocks tensor-parallel placement.

The scheduler prefers migrating loaded / cold-started instances onto the NUMA
domain where exclusive free GPUs are concentrated so a new TP group can span
multiple NUMAs. New placements use NUMA-fair GPU ordering in :class:`FcfsScheduler`
to reduce how often migration is needed.
"""
import asyncio
from typing import Dict, List, Optional

from sllm.logger import init_logger

from .roundrobin_router import RoundRobinRouter

logger = init_logger(__name__)


class MigrationRouter(RoundRobinRouter):
    """Router with NUMA-aware migration for non-prewarm instances."""

    # ------------------------------------------------------------------
    # Override: rebalance hook
    # ------------------------------------------------------------------

    async def _try_rebalance_for_tp(self) -> None:
        """Run scheduler NUMA migration plan (active instances only), if any."""
        pp_size = int(self.resource_requirements.get("pp_size", 1))
        if pp_size > 1:
            return

        tp_size = int(
            self.resource_requirements.get(
                "tp_size", self.resource_requirements.get("num_gpus", 1)
            )
        )
        if tp_size <= 1:
            return

        plan = await self.model_loading_scheduler.suggest_instance_migration.remote(
            self.model_name, tp_size
        )

        if not isinstance(plan, dict) or not plan:
            return

        success = await self._migrate_instance_for_numa(plan)
        if success:
            logger.info(
                "NUMA rebalance migrated instance %s → GPUs %s (TP=%s)",
                plan.get("instance_id"),
                plan.get("target_gpu_ids"),
                tp_size,
            )

    # ------------------------------------------------------------------
    # Override: actual migration
    # ------------------------------------------------------------------

    async def _migrate_empty_instance(
        self, instance_id: str, target_gpu_ids: List[int]
    ) -> bool:
        """Backward-compatible wrapper; prefer :meth:`_migrate_instance_for_numa`."""
        return await self._migrate_instance_for_numa(
            {
                "instance_id": instance_id,
                "target_gpu_ids": target_gpu_ids,
                "node_id": None,
                "target_node_id": None,
            }
        )

    async def _migrate_instance_for_numa(self, plan: Dict) -> bool:
        """Tear down *instance_id* and recreate on *target_gpu_ids* (optional cross-node).

        Requires zero in-flight requests. Works for loaded instances (full weight reload).
        """
        instance_id: Optional[str] = plan.get("instance_id")
        target_gpu_ids: List[int] = list(plan.get("target_gpu_ids", []))
        source_node: Optional[str] = (
            str(plan["node_id"]) if plan.get("node_id") is not None else None
        )
        target_node: Optional[str] = plan.get("target_node_id")
        if target_node is not None:
            target_node = str(target_node)
        if source_node and target_node is None:
            target_node = source_node
        if not instance_id or not target_gpu_ids:
            return False

        async with self.instance_management_lock:
            instance = self.ready_inference_instances.get(instance_id)
            if instance is None:
                logger.warning(
                    "NUMA migration skipped: instance %s not in ready pool",
                    instance_id,
                )
                return False
            if instance.concurrency > 0:
                logger.warning(
                    "NUMA migration skipped: instance %s has active requests",
                    instance_id,
                )
                return False
            self.ready_inference_instances.pop(instance_id, None)
            self.instance_to_load_status.pop(instance_id, None)

        kv_resume: List[List[int]] = []
        try:
            if instance.backend_instance is not None:
                kv_resume = await instance.backend_instance.get_current_tokens.remote()
        except Exception as exc:
            logger.warning(
                "Could not snapshot token state for KV resume after migration: %s",
                exc,
            )

        await self._destroy_backend(instance, instance_id)

        cross = (
            source_node is not None
            and target_node is not None
            and target_node != source_node
        )
        await self._create_instance(
            empty_instance=False,
            preferred_gpu_ids=target_gpu_ids,
            preferred_pp0_node_id=target_node if cross else None,
            kv_resume_after_init=kv_resume or None,
            force_eager_weight_load=True,
        )
        logger.info(
            "Migrated instance %s → GPUs %s (node=%s)",
            instance_id,
            target_gpu_ids,
            target_node,
        )
        return True
