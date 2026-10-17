"""Optional Linux cgroup-v2 study memory limit. No provisioning or fallback."""

import re
from pathlib import Path


def require_cgroup_v2(root):
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        mount, filesystem = line.split(" - ", 1)
        if filesystem.split()[0] != "cgroup2":
            continue
        point = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), mount.split()[4])
        if root.is_relative_to(Path(point)):
            return
    raise ValueError("Memory enforcement requires a real cgroup-v2 filesystem")


def prepare_group(root, name, memory_mib):
    if not name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Invalid cgroup study name")
    root = Path(root).resolve()
    require_cgroup_v2(root)
    if "memory" not in (root / "cgroup.subtree_control").read_text().split():
        raise ValueError(
            "A delegated cgroup-v2 root with memory controller enabled is required"
        )
    group = root / name
    group.mkdir()  # Never reuse another study's group.
    try:
        (group / "memory.max").write_text(str(memory_mib * 1024 * 1024))
        (group / "memory.swap.max").write_text("0")
        (group / "memory.oom.group").write_text("1")
    except OSError:
        group.rmdir()
        raise
    return group


def enter_group(group, memory_mib):
    import os

    group = Path(group).resolve()
    require_cgroup_v2(group)
    if int((group / "memory.max").read_text()) != memory_mib * 1024 * 1024:
        raise ValueError("Study cgroup memory limit differs from requested value")
    if (group / "memory.swap.max").read_text().strip() != "0":
        raise ValueError("Study cgroup must disable swap")
    (group / "cgroup.procs").write_text(str(os.getpid()))
    if str(os.getpid()) not in (group / "cgroup.procs").read_text().split():
        raise ValueError("Study process did not enter its memory cgroup")


def memory_events(group):
    try:
        return {
            k: int(v)
            for k, v in (
                line.split()
                for line in (Path(group) / "memory.events").read_text().splitlines()
            )
        }
    except OSError:
        return {}
