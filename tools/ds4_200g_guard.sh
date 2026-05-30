#!/usr/bin/env bash

ds4_200g_print_diag()
{
  {
    echo "hostname: $(hostname)"
    echo "NODE_RANK: ${NODE_RANK:-<unset>}"
    echo "HEAD_ADDR: ${HEAD_ADDR:-<unset>}"
    echo "DS4_200G_IFNAME: ${DS4_200G_IFNAME:-<unset>}"
    echo "DS4_CONTROL_IFNAME: ${DS4_CONTROL_IFNAME:-<unset>}"
    echo "DS4_200G_ADVERTISE_LOOPBACK: ${DS4_200G_ADVERTISE_LOOPBACK:-<unset>}"
    echo "DS4_NODE_LOOPBACK: ${DS4_NODE_LOOPBACK:-<unset>}"
    echo "DS4_200G_NCCL_TRANSPORT: ${DS4_200G_NCCL_TRANSPORT:-<unset>}"
    echo "NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME:-<unset>}"
    echo "GLOO_SOCKET_IFNAME: ${GLOO_SOCKET_IFNAME:-<unset>}"
    echo "TP_SOCKET_IFNAME: ${TP_SOCKET_IFNAME:-<unset>}"
    echo "NCCL_NET: ${NCCL_NET:-<unset>}"
    echo "NCCL_IB_DISABLE: ${NCCL_IB_DISABLE:-<unset>}"
    echo "VLLM_HOST_IP: ${VLLM_HOST_IP:-<unset>}"
    echo "interfaces:"
    ip -o -4 addr show 2>/dev/null || true
    if [[ -n "${HEAD_ADDR:-}" ]]; then
      echo "route to HEAD_ADDR:"
      ip route get "$HEAD_ADDR" 2>/dev/null || true
    fi
    if [[ -n "${DS4_200G_IFNAME:-}" ]]; then
      echo "selected interface(s):"
      local diag_ifname
      IFS=',' read -r -a diag_ifnames <<< "$DS4_200G_IFNAME"
      for diag_ifname in "${diag_ifnames[@]}"; do
        [[ -n "$diag_ifname" ]] || continue
        ip -o -4 addr show dev "$diag_ifname" 2>/dev/null || true
        if [[ -r "/sys/class/net/$diag_ifname/speed" ]]; then
          echo "$diag_ifname speed: $(cat "/sys/class/net/$diag_ifname/speed" 2>/dev/null)Mb/s"
        fi
        if [[ -r "/sys/class/net/$diag_ifname/carrier" ]]; then
          echo "$diag_ifname carrier: $(cat "/sys/class/net/$diag_ifname/carrier" 2>/dev/null)"
        fi
      done
    fi
  } >&2
}

ds4_200g_die()
{
  echo "DS4 200G fabric guard failed: $*" >&2
  ds4_200g_print_diag
  exit 64
}

ds4_200g_if_ip()
{
  ip -o -4 addr show dev "$1" scope global 2>/dev/null | awk 'NR == 1 { split($4,a,"/"); print a[1] }'
}

ds4_200g_route_dev()
{
  ip route get "$1" 2>/dev/null | awk 'found == 0 { for (i=1; i<=NF; i++) if ($i == "dev") { print $(i+1); found=1 } }'
}

ds4_200g_bound_dev()
{
  ip -o -4 addr show 2>/dev/null | awk -v ip="$1" 'found == 0 && $4 ~ "^" ip "/" { print $2; found=1 }'
}

ds4_200g_hca_for_if()
{
  if ! command -v ibdev2netdev >/dev/null 2>&1; then
    return
  fi
  ibdev2netdev 2>/dev/null | awk -v ifname="$1" 'found == 0 && $0 ~ "==> " ifname " " { print $1; found=1 }'
}

ds4_200g_check_or_export()
{
  local name="$1"
  local value="$2"
  local current="${!name:-}"
  if [[ -n "$current" && "$current" != "$value" ]]; then
    ds4_200g_die "$name is '$current' but must be '$value' for the selected 200G fabric"
  fi
  export "$name=$value"
}

ds4_200g_csv_contains()
{
  local csv="$1"
  local needle="$2"
  local item
  IFS=',' read -r -a items <<< "$csv"
  for item in "${items[@]}"; do
    [[ "$item" == "$needle" ]] && return 0
  done
  return 1
}

ds4_200g_rank_loopback()
{
  if [[ -n "${DS4_NODE_LOOPBACK:-}" ]]; then
    echo "$DS4_NODE_LOOPBACK"
    return 0
  fi
  if [[ "$NODE_RANK" =~ ^[0-9]+$ ]]; then
    echo "10.10.100.$((10 + NODE_RANK))"
    return 0
  fi
  return 1
}

ds4_require_200g_fabric()
{
  local ifname ifnames_csv control_ifname socket_ifname speed carrier local_ip bound_dev route_dev hca hcas_csv advertise_ip advertise_bound nccl_transport
  : "${NODE_RANK:?set NODE_RANK before ds4_require_200g_fabric}"
  : "${HEAD_ADDR:?set HEAD_ADDR before ds4_require_200g_fabric}"
  if [[ -z "${DS4_200G_IFNAME:-}" ]]; then
    ds4_200g_die "DS4_200G_IFNAME is unset; set it to this rank's 200GbE/RoCE interface list"
  fi
  ifnames_csv="$DS4_200G_IFNAME"
  hcas_csv=""
  local_ip=""
  IFS=',' read -r -a ifnames <<< "$ifnames_csv"
  for ifname in "${ifnames[@]}"; do
    [[ -n "$ifname" ]] || ds4_200g_die "DS4_200G_IFNAME contains an empty interface"
    [[ -d "/sys/class/net/$ifname" ]] || ds4_200g_die "interface '$ifname' does not exist"
    [[ -r "/sys/class/net/$ifname/speed" ]] || ds4_200g_die "interface '$ifname' does not expose link speed"
    speed="$(cat "/sys/class/net/$ifname/speed" 2>/dev/null || true)"
    [[ "$speed" == "200000" ]] || ds4_200g_die "interface '$ifname' speed is ${speed:-unknown}Mb/s, expected 200000Mb/s"
    carrier="$(cat "/sys/class/net/$ifname/carrier" 2>/dev/null || true)"
    [[ "$carrier" == "1" ]] || ds4_200g_die "interface '$ifname' carrier is ${carrier:-unknown}, expected 1"
    if [[ -z "$local_ip" ]]; then
      local_ip="$(ds4_200g_if_ip "$ifname")"
    fi
    hca="$(ds4_200g_hca_for_if "$ifname")"
    [[ -n "$hca" ]] || ds4_200g_die "no RoCE HCA maps to interface '$ifname'"
    if [[ -n "$hcas_csv" ]]; then
      hcas_csv="$hcas_csv,$hca"
    else
      hcas_csv="$hca"
    fi
  done
  [[ -n "$local_ip" ]] || ds4_200g_die "selected interface list has no IPv4 fabric address"
  control_ifname="${DS4_CONTROL_IFNAME:-$ifnames_csv}"
  IFS=',' read -r -a control_ifnames <<< "$control_ifname"
  for ifname in "${control_ifnames[@]}"; do
    [[ -n "$ifname" ]] || ds4_200g_die "DS4_CONTROL_IFNAME contains an empty interface"
    [[ -d "/sys/class/net/$ifname" ]] || ds4_200g_die "control interface '$ifname' does not exist"
  done
  ds4_200g_check_or_export GLOO_SOCKET_IFNAME "$control_ifname"
  nccl_transport="${DS4_200G_NCCL_TRANSPORT:-}"
  if [[ -z "$nccl_transport" ]]; then
    if [[ "${DS4_200G_ADVERTISE_LOOPBACK:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
      nccl_transport="socket"
    else
      nccl_transport="ib"
    fi
  fi
  case "$nccl_transport" in
    socket)
      export DS4_200G_NCCL_TRANSPORT="socket"
      socket_ifname="${DS4_200G_SOCKET_IFNAME:-$control_ifname}"
      IFS=',' read -r -a socket_ifnames <<< "$socket_ifname"
      for ifname in "${socket_ifnames[@]}"; do
        [[ -n "$ifname" ]] || ds4_200g_die "DS4_200G_SOCKET_IFNAME contains an empty interface"
        [[ -d "/sys/class/net/$ifname" ]] || ds4_200g_die "socket interface '$ifname' does not exist"
      done
      ds4_200g_check_or_export NCCL_SOCKET_IFNAME "$socket_ifname"
      ds4_200g_check_or_export TP_SOCKET_IFNAME "$socket_ifname"
      ds4_200g_check_or_export NCCL_IB_DISABLE "1"
      ds4_200g_check_or_export NCCL_NET "Socket"
      if [[ -n "${NCCL_IB_HCA:-}" ]]; then
        ds4_200g_die "NCCL_IB_HCA is set while DS4_200G_NCCL_TRANSPORT=socket; unset it or use DS4_200G_NCCL_TRANSPORT=ib"
      fi
      ;;
    ib)
      export DS4_200G_NCCL_TRANSPORT="ib"
      ds4_200g_check_or_export NCCL_SOCKET_IFNAME "$ifnames_csv"
      ds4_200g_check_or_export TP_SOCKET_IFNAME "$ifnames_csv"
      ds4_200g_check_or_export NCCL_IB_DISABLE "0"
      ds4_200g_check_or_export NCCL_NET "IB"
      ds4_200g_check_or_export NCCL_IB_HCA "$hcas_csv"
      ;;
    *)
      ds4_200g_die "unsupported DS4_200G_NCCL_TRANSPORT=$nccl_transport; expected socket or ib"
      ;;
  esac
  bound_dev="$(ds4_200g_bound_dev "$HEAD_ADDR")"
  if [[ "$NODE_RANK" == "0" ]]; then
    if ds4_200g_csv_contains "$control_ifname" "$bound_dev"; then
      ds4_200g_check_or_export VLLM_HOST_IP "$HEAD_ADDR"
      return
    fi
    if ds4_200g_csv_contains "$ifnames_csv" "$bound_dev"; then
      ds4_200g_check_or_export VLLM_HOST_IP "$local_ip"
      return
    fi
    if [[ "$bound_dev" == "lo" && "${DS4_200G_ALLOW_LOOPBACK_HEAD:-0}" == "1" && "$HEAD_ADDR" == 10.10.* ]]; then
      ds4_200g_check_or_export VLLM_HOST_IP "$HEAD_ADDR"
      return
    fi
    ds4_200g_die "rank 0 HEAD_ADDR '$HEAD_ADDR' is bound to '${bound_dev:-no local device}', not 200G interface list '$ifnames_csv'"
  fi
  route_dev="$(ds4_200g_route_dev "$HEAD_ADDR")"
  ds4_200g_csv_contains "$ifnames_csv" "$route_dev" || ds4_200g_die "route to HEAD_ADDR '$HEAD_ADDR' uses '${route_dev:-no route}', not 200G interface list '$ifnames_csv'"
  if [[ "${DS4_200G_ADVERTISE_LOOPBACK:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    advertise_ip="$(ds4_200g_rank_loopback)" || ds4_200g_die "could not derive routed loopback for NODE_RANK=$NODE_RANK"
    advertise_bound="$(ds4_200g_bound_dev "$advertise_ip")"
    if [[ "$advertise_bound" != "lo" ]] && ! ds4_200g_csv_contains "$control_ifname" "$advertise_bound"; then
      ds4_200g_die "advertised loopback '$advertise_ip' is bound to '${advertise_bound:-no local device}', expected lo or control interface '$control_ifname'"
    fi
    ds4_200g_check_or_export VLLM_HOST_IP "$advertise_ip"
    return
  fi
  ds4_200g_check_or_export VLLM_HOST_IP "$local_ip"
}

ds4_run_nccl_preflight()
{
  local world_size="$1"
  local preflight_port="${DS4_NCCL_PREFLIGHT_PORT:-$((MASTER_PORT + 1000))}"
  local mode="${DS4_NCCL_PREFLIGHT_MODE:-nccl}"
  local timeout_s="${DS4_NCCL_PREFLIGHT_TIMEOUT:-90}"
  echo "DS4 200G NCCL preflight: rank=$NODE_RANK/$world_size addr=$HEAD_ADDR port=$preflight_port if=$DS4_200G_IFNAME transport=${DS4_200G_NCCL_TRANSPORT:-<unset>} hca=${NCCL_IB_HCA:-<unset>} host_ip=$VLLM_HOST_IP" >&2
  case "$mode" in
    gloo)
      if command -v timeout >/dev/null 2>&1; then
        RANK="$NODE_RANK" WORLD_SIZE="$world_size" MASTER_ADDR="$HEAD_ADDR" MASTER_PORT="$preflight_port" DS4_NCCL_PREFLIGHT_BACKEND=gloo timeout --kill-after=5s "$((timeout_s + 15))s" "$RUNTIME_PYTHON" "$SCRIPT_DIR/ds4_nccl_preflight.py"
      else
        RANK="$NODE_RANK" WORLD_SIZE="$world_size" MASTER_ADDR="$HEAD_ADDR" MASTER_PORT="$preflight_port" DS4_NCCL_PREFLIGHT_BACKEND=gloo "$RUNTIME_PYTHON" "$SCRIPT_DIR/ds4_nccl_preflight.py"
      fi
      ;;
    nccl)
      if command -v timeout >/dev/null 2>&1; then
        RANK="$NODE_RANK" WORLD_SIZE="$world_size" MASTER_ADDR="$HEAD_ADDR" MASTER_PORT="$preflight_port" timeout --kill-after=5s "$((timeout_s + 15))s" "$RUNTIME_PYTHON" "$SCRIPT_DIR/ds4_nccl_preflight.py"
      else
        RANK="$NODE_RANK" WORLD_SIZE="$world_size" MASTER_ADDR="$HEAD_ADDR" MASTER_PORT="$preflight_port" "$RUNTIME_PYTHON" "$SCRIPT_DIR/ds4_nccl_preflight.py"
      fi
      ;;
    store)
      echo "DS4 store preflight: rank=$NODE_RANK/$world_size addr=$HEAD_ADDR port=$preflight_port" >&2
      RANK="$NODE_RANK" WORLD_SIZE="$world_size" MASTER_ADDR="$HEAD_ADDR" MASTER_PORT="$preflight_port" DS4_STORE_PREFLIGHT_TIMEOUT="$timeout_s" "$RUNTIME_PYTHON" - <<'PY'
import os
import socket
import sys
import time

rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
addr = os.environ["MASTER_ADDR"]
port = int(os.environ["MASTER_PORT"])
timeout = float(os.environ.get("DS4_STORE_PREFLIGHT_TIMEOUT", "90"))

if rank == 0:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((addr, port))
    sock.listen(world)
    sock.settimeout(timeout)
    seen = {0}
    deadline = time.monotonic() + timeout
    while len(seen) < world:
        remaining = max(0.1, deadline - time.monotonic())
        sock.settimeout(remaining)
        conn, peer = sock.accept()
        with conn:
            data = conn.recv(32).decode("ascii", "replace").strip()
            seen.add(int(data))
            conn.sendall(b"ok\n")
    print(f"DS4 store preflight passed on rank 0: {sorted(seen)}", file=sys.stderr)
else:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((addr, port), timeout=3.0) as conn:
                conn.sendall(f"{rank}\n".encode("ascii"))
                data = conn.recv(32)
                if data.strip() == b"ok":
                    print(f"DS4 store preflight passed on rank {rank}", file=sys.stderr)
                    break
        except OSError as exc:
            last = exc
            time.sleep(0.5)
    else:
        raise SystemExit(f"DS4 store preflight failed on rank {rank}: {last}")
PY
      ;;
    skip)
      echo "DS4 NCCL preflight skipped by DS4_NCCL_PREFLIGHT_MODE=skip" >&2
      ;;
    *)
      ds4_200g_die "unsupported DS4_NCCL_PREFLIGHT_MODE=$mode; expected gloo, nccl, store, or skip"
      ;;
  esac
}


ds4_env_path_contains()
{
  local env_name="$1"
  local entry="$2"
  local current="${!env_name:-}"
  local part
  IFS=':' read -r -a parts <<< "$current"
  for part in "${parts[@]}"; do
    [[ "$part" == "$entry" ]] && return 0
  done
  return 1
}

ds4_prepend_env_path()
{
  local env_name="$1"
  local entry="$2"
  [[ -n "$entry" && -d "$entry" ]] || return 0
  if ds4_env_path_contains "$env_name" "$entry"; then
    return 0
  fi
  if [[ -n "${!env_name:-}" ]]; then
    export "$env_name=$entry:${!env_name}"
  else
    export "$env_name=$entry"
  fi
}

ds4_find_executable()
{
  local configured="$1"
  shift
  if [[ -n "$configured" ]]; then
    if [[ -x "$configured" ]]; then
      echo "$configured"
      return 0
    fi
    if command -v "$configured" >/dev/null 2>&1; then
      command -v "$configured"
      return 0
    fi
    return 1
  fi
  local candidate
  for candidate in "$@"; do
    if command -v "$candidate" >/dev/null 2>&1; then
      command -v "$candidate"
      return 0
    fi
  done
  return 1
}

ds4_find_libcuda_path()
{
  local dir candidate
  local search_dirs=()
  if [[ -n "${TRITON_LIBCUDA_PATH:-}" ]]; then
    IFS=':' read -r -a search_dirs <<< "$TRITON_LIBCUDA_PATH"
  fi
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    local ld_dirs=()
    IFS=':' read -r -a ld_dirs <<< "$LD_LIBRARY_PATH"
    search_dirs+=("${ld_dirs[@]}")
  fi
  if [[ -n "${LIBRARY_PATH:-}" ]]; then
    local library_dirs=()
    IFS=':' read -r -a library_dirs <<< "$LIBRARY_PATH"
    search_dirs+=("${library_dirs[@]}")
  fi
  for dir in \
    "${CUDA_HOME:-}/lib64" \
    "${CUDA_HOME:-}/lib64/stubs" \
    "${CUDA_HOME:-}/compat" \
    "${CUDA_HOME:-}/compat/lib" \
    "${CUDA_HOME:-}/compat/lib.real" \
    "${CUDA_PATH:-}/lib64" \
    /usr/local/cuda/lib64 \
    /usr/local/cuda/lib64/stubs \
    /usr/local/cuda/compat \
    /usr/local/cuda/compat/lib \
    /usr/local/cuda/compat/lib.real \
    /usr/lib/aarch64-linux-gnu \
    /usr/lib/x86_64-linux-gnu \
    /usr/lib64 \
    /usr/lib \
    /lib/aarch64-linux-gnu \
    /lib/x86_64-linux-gnu \
    /run/opengl-driver/lib
  do
    [[ -n "$dir" ]] && search_dirs+=("$dir")
  done
  for dir in "${search_dirs[@]}"; do
    [[ -n "$dir" && -d "$dir" ]] || continue
    for candidate in "$dir/libcuda.so" "$dir/libcuda.so.1" "$dir"/libcuda.so.*; do
      [[ -e "$candidate" ]] || continue
      echo "$candidate"
      return 0
    done
  done
  if command -v ldconfig >/dev/null 2>&1; then
    ldconfig -p 2>/dev/null | awk '/libcuda\.so/{print $NF; exit}'
  fi
}

ds4_find_libcudart_path()
{
  local dir candidate
  local search_dirs=()
  if [[ -n "${VLLM_CUDART_SO_PATH:-}" ]]; then
    [[ -e "$VLLM_CUDART_SO_PATH" ]] && echo "$VLLM_CUDART_SO_PATH" && return 0
  fi
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    local ld_dirs=()
    IFS=':' read -r -a ld_dirs <<< "$LD_LIBRARY_PATH"
    search_dirs+=("${ld_dirs[@]}")
  fi
  for dir in \
    "$("$RUNTIME_PYTHON" - <<'PY' 2>/dev/null
import os
try:
    import torch
    root = os.path.abspath(os.path.join(os.path.dirname(torch.__file__), "..", "nvidia"))
    for name in ("cu13", "cu12"):
        path = os.path.join(root, name, "lib")
        if os.path.isdir(path):
            print(path)
except Exception:
    pass
PY
    )" \
    "${CUDA_HOME:-}/targets/sbsa-linux/lib" \
    "${CUDA_HOME:-}/targets/x86_64-linux/lib" \
    "${CUDA_HOME:-}/lib64" \
    "${CUDA_PATH:-}/targets/sbsa-linux/lib" \
    "${CUDA_PATH:-}/targets/x86_64-linux/lib" \
    "${CUDA_PATH:-}/lib64" \
    /usr/local/cuda/targets/sbsa-linux/lib \
    /usr/local/cuda/targets/x86_64-linux/lib \
    /usr/local/cuda/lib64 \
    /usr/lib/aarch64-linux-gnu \
    /usr/lib/x86_64-linux-gnu \
    /usr/lib64 \
    /usr/lib \
    /lib/aarch64-linux-gnu \
    /lib/x86_64-linux-gnu
  do
    [[ -n "$dir" ]] && search_dirs+=("$dir")
  done
  for dir in "${search_dirs[@]}"; do
    [[ -n "$dir" && -d "$dir" ]] || continue
    for candidate in "$dir/libcudart.so" "$dir"/libcudart.so.*; do
      [[ -e "$candidate" ]] || continue
      [[ "$candidate" == *"/stubs/"* || "$(basename "$candidate")" == *stub* ]] && continue
      echo "$candidate"
      return 0
    done
  done
  if command -v ldconfig >/dev/null 2>&1; then
    ldconfig -p 2>/dev/null | awk '/libcudart\.so/ && $NF !~ /\/stubs\// && $NF !~ /stub/ {print $NF; exit}'
  fi
}

ds4_prepare_python_include_environment()
{
  local py_tag include_dir
  local include_dirs=()
  py_tag="$("$RUNTIME_PYTHON" -c 'import sys; print("python%d.%d" % sys.version_info[:2])')" || ds4_200g_die "cannot determine Python include tag from '$RUNTIME_PYTHON'"

  if [[ -n "${DS4_PYTHON_INCLUDE_DIR:-}" ]]; then
    include_dirs+=("$DS4_PYTHON_INCLUDE_DIR")
  fi
  if [[ -n "${DS4_PYTHON_DEV_ROOT:-}" ]]; then
    include_dirs+=(
      "$DS4_PYTHON_DEV_ROOT/usr/include/$py_tag"
      "$DS4_PYTHON_DEV_ROOT/usr/include"
      "$DS4_PYTHON_DEV_ROOT/usr/include/aarch64-linux-gnu/$py_tag"
      "$DS4_PYTHON_DEV_ROOT/usr/include/x86_64-linux-gnu/$py_tag"
    )
  fi
  include_dirs+=(
    "$HOME/ds4_deps/python3.12-dev/usr/include/$py_tag"
    "$HOME/ds4_deps/python3.12-dev/usr/include"
    "$HOME/ds4_deps/python3.12-dev/usr/include/aarch64-linux-gnu/$py_tag"
    "$HOME/ds4_deps/python3.12-dev/usr/include/x86_64-linux-gnu/$py_tag"
    "$HOME/standard-runtimes/python3.12-dev-extract/usr/include/$py_tag"
    "$HOME/standard-runtimes/python3.12-dev-extract/usr/include"
    "$HOME/standard-runtimes/python3.12-dev-extract/usr/include/aarch64-linux-gnu/$py_tag"
    "$HOME/standard-runtimes/python3.12-dev-extract/usr/include/x86_64-linux-gnu/$py_tag"
    "$HOME/.cache/ds4-python312-dev/root/usr/include/$py_tag"
    "$HOME/.cache/ds4-python312-dev/root/usr/include"
    "$HOME/.cache/ds4-python312-dev/root/usr/include/aarch64-linux-gnu/$py_tag"
    "$HOME/.cache/ds4-python312-dev/root/usr/include/x86_64-linux-gnu/$py_tag"
  )

  for include_dir in "${include_dirs[@]}"; do
    [[ -n "$include_dir" && -d "$include_dir" ]] || continue
    ds4_prepend_env_path CPATH "$include_dir"
    ds4_prepend_env_path C_INCLUDE_PATH "$include_dir"
    ds4_prepend_env_path CPLUS_INCLUDE_PATH "$include_dir"
    ds4_prepend_env_path DS4_PYTHON_INCLUDE_DIRS "$include_dir"
  done
  if [[ -z "${DS4_PYTHON_INCLUDE_DIRS:-}" ]]; then
    ds4_200g_die "Triton JIT requires Python.h; install ${py_tag}-dev or set DS4_PYTHON_DEV_ROOT/DS4_PYTHON_INCLUDE_DIR"
  fi
}

ds4_prepare_triton_jit_environment()
{
  local service_name="${1:-ds4}"
  local default_work_root
  local default_ipc_tmp
  default_work_root="$HOME/ds4_triton/$service_name"
  default_ipc_tmp="/tmp/d4i/${MASTER_PORT:-0}_${NODE_RANK:-x}"
  local work_root="${DS4_TRITON_WORK_ROOT:-$default_work_root}"
  local ipc_tmp="${DS4_IPC_TMPDIR:-$default_ipc_tmp}"
  local cc cxx libcuda_path libcuda_dir libcudart_path libcudart_dir symlink_dir

  cc="$(ds4_find_executable "${CC:-${DS4_CC:-}}" gcc cc || true)"
  [[ -n "$cc" ]] || ds4_200g_die "Triton JIT requires gcc/cc; set CC or DS4_CC"
  export CC="$cc"

  cxx="$(ds4_find_executable "${CXX:-${DS4_CXX:-}}" g++ c++ || true)"
  [[ -n "$cxx" ]] || ds4_200g_die "Triton JIT requires g++/c++; set CXX or DS4_CXX"
  export CXX="$cxx"

  if [[ -z "${CUDA_HOME:-}" && -d /usr/local/cuda ]]; then
    export CUDA_HOME=/usr/local/cuda
  fi
  ds4_prepare_python_include_environment

  mkdir -p \
    "$ipc_tmp" \
    "$work_root/tmp" \
    "$work_root/cache" \
    "$work_root/inductor_cache" \
    "$work_root/torch_extensions" \
    "$work_root/vllm_cache" \
    "$work_root/libcuda" || ds4_200g_die "cannot create writable Triton JIT work root '$work_root'; set DS4_TRITON_WORK_ROOT"

  export TMPDIR="${TMPDIR:-$ipc_tmp}"
  if [[ "${#TMPDIR}" -gt 29 ]]; then
    ds4_200g_die "TMPDIR '$TMPDIR' is ${#TMPDIR} chars; LMCache ZMQ IPC sockets require TMPDIR <= 29 chars. Set DS4_IPC_TMPDIR to a short path like /tmp/d4i/${MASTER_PORT:-0}_${NODE_RANK:-x}, or unset TMPDIR."
  fi
  mkdir -p "$TMPDIR" || ds4_200g_die "cannot create short TMPDIR '$TMPDIR'"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$work_root/cache}"
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$work_root/inductor_cache}"
  export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$work_root/torch_extensions}"
  export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$work_root/vllm_cache}"

  libcuda_path="$(ds4_find_libcuda_path || true)"
  [[ -n "$libcuda_path" ]] || ds4_200g_die "Triton JIT requires libcuda.so/libcuda.so.1 visible to the container"
  libcuda_dir="$(dirname "$libcuda_path")"
  ds4_prepend_env_path LD_LIBRARY_PATH "$libcuda_dir"
  ds4_prepend_env_path LIBRARY_PATH "$libcuda_dir"

  libcudart_path="$(ds4_find_libcudart_path || true)"
  [[ -n "$libcudart_path" ]] || ds4_200g_die "DS4 native runtime requires the real libcudart.so; CUDA stubs are not acceptable"
  if [[ "$libcudart_path" == *"/stubs/"* || "$(basename "$libcudart_path")" == *stub* ]]; then
    ds4_200g_die "resolved libcudart to CUDA stub '$libcudart_path'; set VLLM_CUDART_SO_PATH to the real libcudart.so"
  fi
  libcudart_dir="$(dirname "$libcudart_path")"
  ds4_prepend_env_path LD_LIBRARY_PATH "$libcudart_dir"
  export VLLM_CUDART_SO_PATH="$libcudart_path"

  if [[ -e "$libcuda_dir/libcuda.so" ]]; then
    export TRITON_LIBCUDA_PATH="${TRITON_LIBCUDA_PATH:-$libcuda_dir}"
  else
    symlink_dir="$work_root/libcuda"
    ln -sfn "$libcuda_path" "$symlink_dir/libcuda.so"
    export TRITON_LIBCUDA_PATH="$symlink_dir"
    ds4_prepend_env_path LD_LIBRARY_PATH "$symlink_dir"
    ds4_prepend_env_path LIBRARY_PATH "$symlink_dir"
  fi

  echo "DS4 Triton JIT env: service=$service_name CC=$CC CXX=$CXX TRITON_CACHE_DIR=$TRITON_CACHE_DIR TRITON_LIBCUDA_PATH=$TRITON_LIBCUDA_PATH VLLM_CUDART_SO_PATH=$VLLM_CUDART_SO_PATH TMPDIR=$TMPDIR DS4_PYTHON_INCLUDE_DIRS=${DS4_PYTHON_INCLUDE_DIRS:-<unset>}" >&2
}

ds4_prepare_flashinfer_jit_environment()
{
  local max_jobs="${DS4_FLASHINFER_JIT_MAX_JOBS:-1}"
  [[ "$max_jobs" =~ ^[1-9][0-9]*$ ]] || ds4_200g_die "DS4_FLASHINFER_JIT_MAX_JOBS must be a positive integer, got '$max_jobs'"
  if [[ -n "${MAX_JOBS:-}" && ! "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]]; then
    ds4_200g_die "MAX_JOBS must be a positive integer for FlashInfer runtime JIT, got '$MAX_JOBS'"
  fi
  export MAX_JOBS="${MAX_JOBS:-$max_jobs}"
  echo "DS4 FlashInfer JIT guard: MAX_JOBS=$MAX_JOBS DS4_FLASHINFER_JIT_MAX_JOBS=$max_jobs" >&2
}

ds4_set_flashinfer_autotune_args()
{
  local env_name="${1:-DS4_ENABLE_FLASHINFER_AUTOTUNE}"
  local value="${!env_name:-0}"
  FLASHINFER_AUTOTUNE_ARGS=(--no-enable-flashinfer-autotune)
  if [[ "$value" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    if [[ "${DS4_FLASHINFER_AUTOTUNE_TUNING_JOB:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
      FLASHINFER_AUTOTUNE_ARGS=(--enable-flashinfer-autotune)
      echo "DS4 FlashInfer autotune: enabled for explicit tuning job" >&2
      return
    fi
    ds4_200g_die "$env_name=$value requested FlashInfer autotune from a production/validation launcher; use a dedicated tuning job with DS4_FLASHINFER_AUTOTUNE_TUNING_JOB=1"
  fi
  echo "DS4 FlashInfer autotune: disabled for production/validation startup" >&2
}

ds4_run_triton_jit_preflight()
{
  local args=()
  echo "DS4 Triton JIT preflight: CC=${CC:-<unset>} TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-<unset>} TRITON_LIBCUDA_PATH=${TRITON_LIBCUDA_PATH:-<unset>}" >&2
  if [[ "${DS4_TRITON_ACTIVE_JIT_PREFLIGHT:-1}" != "1" ]]; then
    args+=(--skip-active-jit-probe)
  fi
  if [[ "${DS4_TRITON_LIBCUDA_LINK_PREFLIGHT:-1}" != "1" ]]; then
    args+=(--skip-libcuda-link-probe)
  fi
  "$RUNTIME_PYTHON" "$SCRIPT_DIR/ds4_triton_jit_preflight.py" "${args[@]}"
}

ds4_run_native_blackwell_preflight()
{
  echo "DS4 native Blackwell preflight: strict=$VLLM_DS4_STRICT_NATIVE_FP4 deep_gemm=$VLLM_USE_DEEP_GEMM e8m0=$VLLM_USE_DEEP_GEMM_E8M0" >&2
  "$RUNTIME_PYTHON" "$SCRIPT_DIR/ds4_native_blackwell_probe.py" --strict-dsv4
}

ds4_run_dsv4_native_preflight()
{
  local args=()
  echo "DS4 DSV4 native package preflight: active_probe=${DS4_NATIVE_PREFLIGHT_ACTIVE:-0}" >&2
  if [[ "${DS4_NATIVE_PREFLIGHT_ACTIVE:-0}" == "1" ]]; then
    args+=(--active-kernel-probe)
  fi
  "$RUNTIME_PYTHON" "$SCRIPT_DIR/ds4_dsv4_native_preflight.py" "${args[@]}"
}
