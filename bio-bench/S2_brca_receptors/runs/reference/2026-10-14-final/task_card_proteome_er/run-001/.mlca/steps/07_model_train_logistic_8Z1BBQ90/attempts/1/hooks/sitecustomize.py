import json, os, pathlib, sys
rules = json.loads(os.environ["MLCA_AUDIT_RULES"])
read_roots = [pathlib.Path(p).resolve() for p in rules["read_roots"]]
write_roots = [pathlib.Path(p).resolve() for p in rules["write_roots"]]
def inside(path, roots):
    # normcase preserves POSIX case and folds Windows drive/path case.
    path = os.path.normcase(str(path))
    for root in roots:
        root = os.path.normcase(str(root))
        try:
            if os.path.commonpath([path, root]) == root:
                return True
        except ValueError:
            pass  # Different Windows drives cannot share a root.
    return False
def guard(event, args):
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        path = pathlib.Path(os.fsdecode(args[0])).resolve()
        mode = args[1] or ""
        flags = args[2] if len(args)>2 else 0
        write = (isinstance(mode,str) and any(c in mode for c in "wax+")) or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if not inside(path, write_roots if write else read_roots):
            raise PermissionError("MLCA blocked an undeclared file access")
    elif event in ("os.system", "subprocess.Popen", "os.posix_spawn"):
        raise PermissionError("MLCA blocked child process creation")
    elif event.startswith("socket.") and not rules["network"]:
        raise PermissionError("MLCA blocked network access")
sys.addaudithook(guard)
