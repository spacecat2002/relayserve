# ---------------------------------------------------------------------------- #
#  CGroup Utilities for CPU Affinity Management                                #
#  Provides functions to set CPU affinity using cgroup v2 cpuset controller   #
# ---------------------------------------------------------------------------- #
import logging
import os
import subprocess
from pathlib import Path
from typing import Set, Optional

logger = logging.getLogger(__name__)


def setup_cgroup_cpuset(
    cgroup_name: str,
    cpus: Set[int],
    cgroup_root: str = "/sys/fs/cgroup",
    parent_cgroup: Optional[str] = None,
) -> Optional[str]:
    """
    使用 cgroup v2 的 cpuset 控制器设置 CPU 亲和性。
    
    Args:
        cgroup_name: cgroup 的名称
        cpus: 要绑定的 CPU 核心集合，例如 {62} 或 {0, 1, 2}
        cgroup_root: cgroup 根目录，默认为 /sys/fs/cgroup
        parent_cgroup: 父 cgroup 路径（可选），例如 "user.slice" 或 "ray.slice"
    
    Returns:
        创建的 cgroup 路径，如果失败则返回 None
    
    Example:
        # 绑定到 CPU 62
        cgroup_path = setup_cgroup_cpuset("gpu_backend", {62})
        
        # 绑定到多个 CPU
        cgroup_path = setup_cgroup_cpuset("cpu_backend", {0, 1, 2, 32, 33, 34})
    """
    try:
        # 确定 cgroup 路径
        if parent_cgroup:
            cgroup_path = Path(cgroup_root) / parent_cgroup / cgroup_name
        else:
            cgroup_path = Path(cgroup_root) / cgroup_name
        
        # 创建 cgroup 目录
        cgroup_path.mkdir(parents=True, exist_ok=True)
        
        # 检查 cpuset 控制器是否可用
        cpuset_path = cgroup_path / "cpuset.cpus"
        if not cpuset_path.exists():
            # 尝试使用 unified cgroup v2 路径
            unified_path = Path(cgroup_root) / "unified" / cgroup_name
            if parent_cgroup:
                unified_path = Path(cgroup_root) / "unified" / parent_cgroup / cgroup_name
            unified_path.mkdir(parents=True, exist_ok=True)
            cpuset_path = unified_path / "cpuset.cpus"
            
            if not cpuset_path.exists():
                logger.warning(
                    f"cpuset controller not available at {cgroup_path} or {unified_path}. "
                    "Falling back to os.sched_setaffinity"
                )
                return None
        
        # 将 CPU 集合转换为 cgroup 格式（例如：0-2,32-34 或 62）
        cpu_list = sorted(cpus)
        cpu_str = format_cpu_list(cpu_list)
        
        # 写入 cpuset.cpus
        with open(cpuset_path, "w") as f:
            f.write(cpu_str)
        
        # 设置 cpuset.mems（通常设置为 0 表示所有内存节点）
        mems_path = cgroup_path.parent if "unified" in str(cpuset_path) else cgroup_path
        mems_path = mems_path / "cpuset.mems"
        if mems_path.exists():
            with open(mems_path, "w") as f:
                f.write("0")
        
        logger.info(
            f"Created cgroup {cgroup_path} with CPUs: {cpu_str}"
        )
        
        print(f"Created cgroup {cgroup_path} with CPUs: {cpu_str}")
        
        return str(cgroup_path)
        
    except PermissionError:
        logger.error(
            f"Permission denied when creating cgroup. "
            "You may need to run with sudo or have appropriate permissions."
        )
        return None
    except Exception as e:
        logger.error(f"Failed to setup cgroup: {e}")
        return None


def format_cpu_list(cpu_list: list) -> str:
    """
    将 CPU 列表格式化为 cgroup 格式。
    
    Args:
        cpu_list: 排序后的 CPU 列表，例如 [0, 1, 2, 32, 33, 34]
    
    Returns:
        格式化的字符串，例如 "0-2,32-34"
    """
    if not cpu_list:
        return ""
    
    ranges = []
    start = cpu_list[0]
    end = cpu_list[0]
    
    for cpu in cpu_list[1:]:
        if cpu == end + 1:
            end = cpu
        else:
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            start = cpu
            end = cpu
    
    # 添加最后一个范围
    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")
    
    return ",".join(ranges)


def move_process_to_cgroup(pid: int, cgroup_path: str) -> bool:
    """
    将进程移动到指定的 cgroup。
    
    Args:
        pid: 进程 ID（0 表示当前进程）
        cgroup_path: cgroup 路径
    
    Returns:
        成功返回 True，失败返回 False
    """
    try:
        cgroup_procs_path = Path(cgroup_path) / "cgroup.procs"
        
        # 如果不存在，尝试 unified 路径
        if not cgroup_procs_path.exists():
            # 检查是否是 unified cgroup
            if "unified" not in cgroup_path:
                unified_path = Path(cgroup_path).parent.parent / "unified" / Path(cgroup_path).name
                cgroup_procs_path = unified_path / "cgroup.procs"
            else:
                cgroup_procs_path = Path(cgroup_path) / "cgroup.procs"
        
        if not cgroup_procs_path.exists():
            logger.error(f"cgroup.procs not found at {cgroup_procs_path}")
            return False
        
        with open(cgroup_procs_path, "w") as f:
            f.write(str(pid))
        
        logger.info(f"Moved process {pid} to cgroup {cgroup_path}")
        return True
        
    except PermissionError:
        logger.error(
            f"Permission denied when moving process to cgroup. "
            "You may need to run with sudo or have appropriate permissions."
        )
        return False
    except Exception as e:
        logger.error(f"Failed to move process to cgroup: {e}")
        return False


def set_cpu_affinity_with_cgroup(
    cpus: Set[int],
    cgroup_name: Optional[str] = None,
    parent_cgroup: Optional[str] = None,
    fallback_to_sched: bool = True,
) -> bool:
    """
    使用 cgroup 设置当前进程的 CPU 亲和性。
    如果 cgroup 不可用，可以回退到 os.sched_setaffinity。
    
    Args:
        cpus: 要绑定的 CPU 核心集合
        cgroup_name: cgroup 名称（如果为 None，则使用进程名）
        parent_cgroup: 父 cgroup 路径（可选）
        fallback_to_sched: 如果 cgroup 失败，是否回退到 os.sched_setaffinity
    
    Returns:
        成功返回 True，失败返回 False
    """
    import os
    
    # 如果没有指定 cgroup 名称，使用进程名
    if cgroup_name is None:
        cgroup_name = f"process_{os.getpid()}"
    
    # 尝试使用 cgroup
    cgroup_path = setup_cgroup_cpuset(
        cgroup_name=cgroup_name,
        cpus=cpus,
        parent_cgroup=parent_cgroup,
    )
    
    if cgroup_path:
        # 将当前进程移动到 cgroup
        success = move_process_to_cgroup(0, cgroup_path)
        if success:
            logger.info(f"Successfully set CPU affinity using cgroup: {cpus}")
            return True
    
    # 如果 cgroup 失败且允许回退，使用 os.sched_setaffinity
    if fallback_to_sched:
        try:
            os.sched_setaffinity(0, cpus)
            logger.info(
                f"Cgroup not available, using os.sched_setaffinity: {cpus}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to set CPU affinity: {e}")
            return False
    
    return False


def cleanup_cgroup(cgroup_path: str) -> bool:
    """
    清理（删除）cgroup。
    
    Args:
        cgroup_path: cgroup 路径
    
    Returns:
        成功返回 True，失败返回 False
    """
    try:
        import shutil
        path = Path(cgroup_path)
        
        # 确保 cgroup 为空（没有进程）
        procs_path = path / "cgroup.procs"
        if procs_path.exists():
            with open(procs_path, "r") as f:
                procs = f.read().strip().split("\n")
                procs = [p for p in procs if p.strip()]
                if procs:
                    logger.warning(
                        f"Cgroup {cgroup_path} still contains processes: {procs}"
                    )
                    return False
        
        # 删除 cgroup 目录
        if path.exists():
            shutil.rmtree(path)
            logger.info(f"Cleaned up cgroup: {cgroup_path}")
            return True
        
        return False
        
    except PermissionError:
        logger.error(
            f"Permission denied when cleaning up cgroup. "
            "You may need to run with sudo or have appropriate permissions."
        )
        return False
    except Exception as e:
        logger.error(f"Failed to cleanup cgroup: {e}")
        return False

