#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fail-fast Triton JIT launcher/compiler validation for DS4 GB10 serving.

DSV4 native startup now reaches the Torch/Triton profile-compile path. Triton
builds a small C launcher module at runtime with the active host compiler. This
preflight checks that the exact serving environment can compile/import a Python
extension, link against libcuda, and JIT-launch a tiny Triton CUDA kernel before
vLLM spends minutes loading DeepSeek-V4-Flash.
"""

from __future__ import annotations

import argparse
import ctypes.util
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import textwrap
from pathlib import Path
from typing import Iterable


def emit(label: str, value: object) -> None:
    print(f"{label}: {value}", flush=True)


def bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def executable_from_env(name: str, fallback_names: Iterable[str]) -> str | None:
    configured = os.environ.get(name)
    if configured:
        if os.path.isabs(configured) and os.access(configured, os.X_OK):
            return configured
        resolved = shutil.which(configured)
        return resolved
    for fallback in fallback_names:
        resolved = shutil.which(fallback)
        if resolved:
            return resolved
    return None


def existing_path(path: str | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path).expanduser()
    return candidate if candidate.exists() else None


def python_include_dirs() -> list[Path]:
    paths: list[Path] = []
    for key in ("include", "platinclude"):
        value = sysconfig.get_paths().get(key)
        path = existing_path(value)
        if path is not None:
            paths.append(path)
    include_py = existing_path(sysconfig.get_config_var("INCLUDEPY"))
    if include_py is not None:
        paths.append(include_py)
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def ldconfig_libcuda_paths() -> list[Path]:
    try:
        output = subprocess.check_output(
            ["/sbin/ldconfig", "-p"], text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        return []
    paths: list[Path] = []
    for line in output.splitlines():
        if "libcuda.so" not in line:
            continue
        _, _, path_text = line.rpartition("=>")
        path = existing_path(path_text.strip())
        if path is not None:
            paths.append(path)
    return paths


def libcuda_candidate_paths() -> list[Path]:
    candidates: list[Path] = []

    for env_name in ("TRITON_LIBCUDA_PATH", "LD_LIBRARY_PATH", "LIBRARY_PATH"):
        for item in os.environ.get(env_name, "").split(os.pathsep):
            item = item.strip()
            if not item:
                continue
            directory = existing_path(item)
            if directory is None or not directory.is_dir():
                continue
            for filename in ("libcuda.so", "libcuda.so.1"):
                path = existing_path(str(directory / filename))
                if path is not None:
                    candidates.append(path)

    for root in (
        os.environ.get("CUDA_HOME"),
        os.environ.get("CUDA_PATH"),
        "/usr/local/cuda",
    ):
        root_path = existing_path(root)
        if root_path is None:
            continue
        for relative in (
            "lib64/libcuda.so",
            "lib64/libcuda.so.1",
            "lib64/stubs/libcuda.so",
            "compat/libcuda.so",
            "compat/libcuda.so.1",
            "compat/lib.real/libcuda.so",
            "compat/lib.real/libcuda.so.1",
        ):
            path = existing_path(str(root_path / relative))
            if path is not None:
                candidates.append(path)

    for glob_root in (
        "/usr/lib/aarch64-linux-gnu",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib64",
        "/usr/lib",
        "/lib/aarch64-linux-gnu",
        "/lib/x86_64-linux-gnu",
        "/run/opengl-driver/lib",
    ):
        root_path = existing_path(glob_root)
        if root_path is None:
            continue
        for pattern in ("libcuda.so", "libcuda.so.1", "libcuda.so.*"):
            candidates.extend(path for path in root_path.glob(pattern) if path.exists())

    candidates.extend(ldconfig_libcuda_paths())

    find_library_result = ctypes.util.find_library("cuda")
    path = existing_path(find_library_result)
    if path is not None:
        candidates.append(path)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            deduped.append(resolved)
            seen.add(resolved)
    return deduped


def compile_shared_module(
    cc: str,
    source: str,
    output: Path,
    include_dirs: list[Path],
    extra_args: list[str] | None = None,
) -> tuple[bool, str]:
    source_path = output.with_suffix(".c")
    source_path.write_text(source)
    command = [
        cc,
        "-shared",
        "-fPIC",
        "-O2",
        str(source_path),
        "-o",
        str(output),
    ]
    for include_dir in include_dirs:
        command.extend(["-I", str(include_dir)])
    if extra_args:
        command.extend(extra_args)
    emit("compile_command", " ".join(command))
    proc = subprocess.run(command, text=True, capture_output=True)
    if proc.returncode != 0:
        return False, (proc.stdout + proc.stderr).strip()
    return True, str(output)


def check_compiler() -> tuple[bool, str | None]:
    cc = executable_from_env("CC", ("gcc", "cc"))
    cxx = executable_from_env("CXX", ("g++", "c++"))
    emit("cc", cc or "no")
    emit("cxx", cxx or "no")
    ok = cc is not None
    if cc is None:
        emit("compiler", "no: set CC or install gcc")
        return False, None
    try:
        version = subprocess.check_output([cc, "--version"], text=True).splitlines()[0]
        emit("cc_version", version)
    except Exception as exc:
        emit("cc_version", f"no: {exc}")
        ok = False
    return ok, cc


def check_python_headers() -> tuple[bool, list[Path]]:
    include_dirs = python_include_dirs()
    emit("python_executable", sys.executable)
    emit("python_version", sys.version.split()[0])
    emit("python_include_dirs", os.pathsep.join(str(path) for path in include_dirs))
    python_h = next((path / "Python.h" for path in include_dirs if (path / "Python.h").exists()), None)
    emit("python_h", python_h or "no")
    return python_h is not None, include_dirs


def check_cache_paths() -> bool:
    ok = True
    for env_name in (
        "TMPDIR",
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "TORCH_EXTENSIONS_DIR",
        "VLLM_CACHE_ROOT",
    ):
        value = os.environ.get(env_name)
        if not value:
            emit(env_name.lower(), "unset")
            continue
        path = Path(value).expanduser()
        try:
            path.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path, delete=True) as handle:
                handle.write(b"ds4")
                handle.flush()
            emit(env_name.lower(), f"yes:{path}")
        except Exception as exc:
            emit(env_name.lower(), f"no:{path}:{exc}")
            ok = False
    return ok


def check_python_extension_compile(cc: str, include_dirs: list[Path]) -> bool:
    with tempfile.TemporaryDirectory(prefix="ds4_pyext_", dir=os.environ.get("TMPDIR")) as tmp:
        module_path = Path(tmp) / "ds4_pyext_probe.so"
        source = textwrap.dedent(
            """
            #include <Python.h>
            static PyObject* ds4_ok(PyObject* self, PyObject* args) {
                Py_RETURN_TRUE;
            }
            static PyMethodDef Methods[] = {
                {"ok", ds4_ok, METH_NOARGS, "ok"},
                {NULL, NULL, 0, NULL}
            };
            static struct PyModuleDef Module = {
                PyModuleDef_HEAD_INIT, "ds4_pyext_probe", NULL, -1, Methods
            };
            PyMODINIT_FUNC PyInit_ds4_pyext_probe(void) {
                return PyModule_Create(&Module);
            }
            """
        ).strip()
        ok, message = compile_shared_module(cc, source, module_path, include_dirs)
        emit("python_extension_compile", f"yes:{message}" if ok else f"no:{message}")
        return ok


def check_libcuda_compile(cc: str, include_dirs: list[Path]) -> bool:
    candidates = libcuda_candidate_paths()
    emit("libcuda_candidates", os.pathsep.join(str(path) for path in candidates) or "no")
    if not candidates:
        emit("libcuda_compile", "no: no libcuda.so or libcuda.so.1 candidate found")
        return False

    link_args_sets: list[list[str]] = []
    for candidate in candidates:
        if candidate.name == "libcuda.so":
            link_args_sets.append(["-L", str(candidate.parent), "-lcuda"])
        else:
            # Direct linking to the resolved .so.1 catches missing path issues even
            # when the container lacks the unversioned development symlink.
            link_args_sets.append([str(candidate)])
    for candidate in candidates:
        if candidate.parent.exists():
            link_args_sets.append(["-L", str(candidate.parent), "-lcuda"])

    source = textwrap.dedent(
        """
        #include <Python.h>
        extern int cuDriverGetVersion(int *driverVersion);
        static PyObject* ds4_cuda_version(PyObject* self, PyObject* args) {
            int version = 0;
            int status = cuDriverGetVersion(&version);
            if (status != 0) {
                return PyLong_FromLong(-status);
            }
            return PyLong_FromLong(version);
        }
        static PyMethodDef Methods[] = {
            {"driver_version", ds4_cuda_version, METH_NOARGS, "driver version"},
            {NULL, NULL, 0, NULL}
        };
        static struct PyModuleDef Module = {
            PyModuleDef_HEAD_INIT, "ds4_libcuda_probe", NULL, -1, Methods
        };
        PyMODINIT_FUNC PyInit_ds4_libcuda_probe(void) {
            return PyModule_Create(&Module);
        }
        """
    ).strip()

    last_message = ""
    with tempfile.TemporaryDirectory(prefix="ds4_libcuda_", dir=os.environ.get("TMPDIR")) as tmp:
        for index, extra_args in enumerate(link_args_sets):
            module_path = Path(tmp) / f"ds4_libcuda_probe_{index}.so"
            ok, message = compile_shared_module(
                cc, source, module_path, include_dirs, extra_args=extra_args
            )
            if ok:
                emit("libcuda_compile", f"yes:{message}:{' '.join(extra_args)}")
                return True
            last_message = message
    emit("libcuda_compile", f"no:{last_message}")
    return False


def check_triton_import() -> bool:
    try:
        import triton  # noqa: F401
    except BaseException as exc:
        emit("triton_import", f"no: {exc}")
        return False
    emit("triton_version", getattr(triton, "__version__", "unknown"))
    emit("triton_file", getattr(triton, "__file__", "unknown"))
    return True


def check_triton_active_jit() -> bool:
    try:
        import torch
        import triton
        import triton.language as tl
    except BaseException as exc:
        emit("triton_active_import", f"no: {exc}")
        return False

    if not torch.cuda.is_available():
        emit("triton_active_cuda", "no")
        return False

    @triton.jit
    def _ds4_triton_launcher_probe(output):
        tl.store(output, tl.full((), 7, tl.int32))

    try:
        output = torch.empty((1,), device="cuda", dtype=torch.int32)
        _ds4_triton_launcher_probe[(1,)](output)
        torch.cuda.synchronize()
        value = int(output.cpu()[0])
        if value != 7:
            emit("triton_active_jit", f"no:bad_value:{value}")
            return False
        emit("triton_active_jit", "yes")
        return True
    except BaseException as exc:
        emit("triton_active_jit", f"no: {type(exc).__name__}: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-active-jit-probe",
        action="store_true",
        help="Skip the tiny Triton CUDA kernel launch.",
    )
    parser.add_argument(
        "--skip-libcuda-link-probe",
        action="store_true",
        help="Skip the host compiler libcuda link probe.",
    )
    args = parser.parse_args()

    ok = True
    compiler_ok, cc = check_compiler()
    ok = compiler_ok and ok
    headers_ok, include_dirs = check_python_headers()
    ok = headers_ok and ok
    ok = check_cache_paths() and ok

    if cc and include_dirs:
        ok = check_python_extension_compile(cc, include_dirs) and ok
        if not args.skip_libcuda_link_probe:
            ok = check_libcuda_compile(cc, include_dirs) and ok

    ok = check_triton_import() and ok
    if not args.skip_active_jit_probe:
        ok = check_triton_active_jit() and ok

    emit("ds4_triton_jit_preflight", "pass" if ok else "fail")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
