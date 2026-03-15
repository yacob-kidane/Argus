"""
Static safety validator for agent-generated code.
Rejects unsafe constructs before any sbatch submission.
See AGENTS.md Safety Principles.
"""
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Modules that are never allowed in generated code
FORBIDDEN_IMPORTS = {
    "subprocess", "socket", "requests", "urllib",
    "shutil", "paramiko", "ftplib", "httplib",
    "smtplib", "telnetlib", "xmlrpc",
}

# os.* calls that are never allowed
FORBIDDEN_OS_ATTRS = {
    "system", "popen", "execv", "execve", "execvp",
    "execvpe", "spawnl", "spawnle", "spawnlp", "spawnlpe",
    "spawnv", "spawnve", "spawnvp", "spawnvpe",
    "fork", "forkpty",
}

# Top-level function calls never allowed
FORBIDDEN_CALLS = {
    "__import__",
}


def is_safe(code: str) -> tuple[bool, str]:
    """
    Validate agent-generated Python code.
    Returns (True, 'ok') if safe.
    Returns (False, reason) if unsafe.
    """
    # Must parse cleanly
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax error: {e}"

    for node in ast.walk(tree):

        # Forbidden imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for name in names:
                root = name.split(".")[0]
                if root in FORBIDDEN_IMPORTS:
                    return False, f"forbidden import: {name}"

        # Forbidden calls: __import__, eval, exec
        if isinstance(node, ast.Call):
            # Direct name calls: __import__("x"), eval(...), exec(...)
            if isinstance(node.func, ast.Name):
                if node.func.id in FORBIDDEN_CALLS:
                    return False, f"forbidden call: {node.func.id}()"
                # eval/exec only forbidden on non-literal args
                if node.func.id in ("eval", "exec"):
                    if node.args and not isinstance(node.args[0], ast.Constant):
                        return False, f"forbidden dynamic {node.func.id}()"

            # os.system(), os.popen(), os.exec*() etc.
            if isinstance(node.func, ast.Attribute):
                if (isinstance(node.func.value, ast.Name) and
                        node.func.value.id == "os" and
                        node.func.attr in FORBIDDEN_OS_ATTRS):
                    return False, f"forbidden call: os.{node.func.attr}()"

        # open() writing outside allowed paths
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "open":
                if len(node.args) >= 2:
                    mode_node = node.args[1]
                    if isinstance(mode_node, ast.Constant):
                        mode = str(mode_node.value)
                        if any(c in mode for c in ("w", "a", "x")):
                            # Check path arg
                            path_node = node.args[0]
                            if isinstance(path_node, ast.Constant):
                                path = str(path_node.value)
                                allowed = ("/tmp", config.JOBS_DIR)
                                if not any(path.startswith(a) for a in allowed):
                                    return False, (
                                        f"forbidden write outside allowed paths: {path}"
                                    )

    return True, "ok"


def check_file(path: str) -> tuple[bool, str]:
    """Convenience wrapper to check a file on disk."""
    with open(path, "r") as f:
        code = f.read()
    return is_safe(code)


def log_rejection(fingerprint: str, reason: str) -> None:
    """Log rejected code to the AST rejection log."""
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    log_path = os.path.join(config.LOGS_DIR, "ast_rejections.log")
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).isoformat()
    with open(log_path, "a") as f:
        f.write(f"{ts} | {fingerprint} | {reason}\n")


if __name__ == "__main__":
    # Safe code — should pass
    safe_code = """
MATRIX_LR = 0.02
DEPTH = 8
WINDOW_PATTERN = "SSSL"
"""
    ok, reason = is_safe(safe_code)
    assert ok, f"safe code rejected: {reason}"
    print("[ast_check] safe code ok")

    # Forbidden import
    bad_import = "import subprocess\nsubprocess.run(['ls'])"
    ok, reason = is_safe(bad_import)
    assert not ok
    assert "subprocess" in reason
    print(f"[ast_check] forbidden import caught: {reason}")

    # os.system
    bad_os = "import os\nos.system('rm -rf /')"
    ok, reason = is_safe(bad_os)
    assert not ok
    assert "os.system" in reason
    print(f"[ast_check] os.system caught: {reason}")

    # __import__
    bad_dyn = "__import__('socket')"
    ok, reason = is_safe(bad_dyn)
    assert not ok
    print(f"[ast_check] __import__ caught: {reason}")

    # Dynamic eval
    bad_eval = "eval(user_input)"
    ok, reason = is_safe(bad_eval)
    assert not ok
    print(f"[ast_check] dynamic eval caught: {reason}")

    # Literal eval is ok
    ok_eval = "eval('1 + 1')"
    ok, reason = is_safe(ok_eval)
    assert ok, f"literal eval should be allowed: {reason}"
    print("[ast_check] literal eval allowed ok")

    # Syntax error
    ok, reason = is_safe("def broken(")
    assert not ok
    assert "syntax" in reason
    print(f"[ast_check] syntax error caught: {reason}")

    # Check the real train.py passes
    ok, reason = check_file(config.TRAIN_PY_SOURCE)
    assert ok, f"train.py failed safety check: {reason}"
    print(f"[ast_check] train.py passes safety check")

    print("[ast_check] ALL TESTS PASSED")
