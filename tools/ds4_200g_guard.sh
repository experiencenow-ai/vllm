#!/usr/bin/env bash

ds4_200g_print_diag()
{
  {
    echo "hostname: $(hostname)"
    echo "NODE_RANK: ${NODE_RANK:-<unset>}"
    echo "HEAD_ADDR: ${HEAD_ADDR:-<unset>}"
    echo "DS4_200G_IFNAME: ${DS4_200G_IFNAME:-<unset>}"
    echo "NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME:-<unset>}"
    echo "GLOO_SOCKET_IFNAME: ${GLOO_SOCKET_IFNAME:-<unset>}"
    echo "TP_SOCKET_IFNAME: ${TP_SOCKET_IFNAME:-<unset>}"
    echo "VLLM_HOST_IP: ${VLLM_HOST_IP:-<unset>}"
    echo "interfaces:"
    ip -o -4 addr show 2>/dev/null || true
    if [[ -n "${HEAD_ADDR:-}" ]]; then
      echo "route to HEAD_ADDR:"
      ip route get "$HEAD_ADDR" 2>/dev/null || true
    fi
    if [[ -n "${DS4_200G_IFNAME:-}" && -d "/sys/class/net/$DS4_200G_IFNAME" ]]; then
      echo "selected interface:"
      ip -o -4 addr show dev "$DS4_200G_IFNAME" 2>/dev/null || true
      if [[ -r "/sys/class/net/$DS4_200G_IFNAME/speed" ]]; then
        echo "speed: $(cat "/sys/class/net/$DS4_200G_IFNAME/speed" 2>/dev/null)Mb/s"
      fi
      if [[ -r "/sys/class/net/$DS4_200G_IFNAME/carrier" ]]; then
        echo "carrier: $(cat "/sys/class/net/$DS4_200G_IFNAME/carrier" 2>/dev/null)"
      fi
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

ds4_require_200g_fabric()
{
  local ifname speed carrier local_ip bound_dev route_dev hca
  : "${NODE_RANK:?set NODE_RANK before ds4_require_200g_fabric}"
  : "${HEAD_ADDR:?set HEAD_ADDR before ds4_require_200g_fabric}"
  if [[ -z "${DS4_200G_IFNAME:-}" ]]; then
    ds4_200g_die "DS4_200G_IFNAME is unset; set it to this rank's 200GbE/RoCE interface"
  fi
  ifname="$DS4_200G_IFNAME"
  [[ -d "/sys/class/net/$ifname" ]] || ds4_200g_die "interface '$ifname' does not exist"
  [[ -r "/sys/class/net/$ifname/speed" ]] || ds4_200g_die "interface '$ifname' does not expose link speed"
  speed="$(cat "/sys/class/net/$ifname/speed" 2>/dev/null || true)"
  [[ "$speed" == "200000" ]] || ds4_200g_die "interface '$ifname' speed is ${speed:-unknown}Mb/s, expected 200000Mb/s"
  carrier="$(cat "/sys/class/net/$ifname/carrier" 2>/dev/null || true)"
  [[ "$carrier" == "1" ]] || ds4_200g_die "interface '$ifname' carrier is ${carrier:-unknown}, expected 1"
  local_ip="$(ds4_200g_if_ip "$ifname")"
  [[ -n "$local_ip" ]] || ds4_200g_die "interface '$ifname' has no IPv4 fabric address"
  ds4_200g_check_or_export NCCL_SOCKET_IFNAME "$ifname"
  ds4_200g_check_or_export GLOO_SOCKET_IFNAME "$ifname"
  ds4_200g_check_or_export TP_SOCKET_IFNAME "$ifname"
  ds4_200g_check_or_export VLLM_HOST_IP "$local_ip"
  hca="$(ds4_200g_hca_for_if "$ifname")"
  [[ -n "$hca" ]] || ds4_200g_die "no RoCE HCA maps to interface '$ifname'"
  ds4_200g_check_or_export NCCL_IB_DISABLE "0"
  ds4_200g_check_or_export NCCL_NET "IB"
  ds4_200g_check_or_export NCCL_IB_HCA "$hca"
  bound_dev="$(ds4_200g_bound_dev "$HEAD_ADDR")"
  if [[ "$NODE_RANK" == "0" ]]; then
    if [[ "$bound_dev" == "$ifname" ]]; then
      return
    fi
    if [[ "$bound_dev" == "lo" && "${DS4_200G_ALLOW_LOOPBACK_HEAD:-0}" == "1" && "$HEAD_ADDR" == 10.10.* ]]; then
      return
    fi
    ds4_200g_die "rank 0 HEAD_ADDR '$HEAD_ADDR' is bound to '${bound_dev:-no local device}', not 200G interface '$ifname'"
  fi
  route_dev="$(ds4_200g_route_dev "$HEAD_ADDR")"
  [[ "$route_dev" == "$ifname" ]] || ds4_200g_die "route to HEAD_ADDR '$HEAD_ADDR' uses '${route_dev:-no route}', not 200G interface '$ifname'"
}

ds4_run_nccl_preflight()
{
  local world_size="$1"
  local preflight_port="${DS4_NCCL_PREFLIGHT_PORT:-$((MASTER_PORT + 1000))}"
  echo "DS4 200G NCCL preflight: rank=$NODE_RANK/$world_size addr=$HEAD_ADDR port=$preflight_port if=$DS4_200G_IFNAME hca=$NCCL_IB_HCA host_ip=$VLLM_HOST_IP" >&2
  RANK="$NODE_RANK" WORLD_SIZE="$world_size" MASTER_ADDR="$HEAD_ADDR" MASTER_PORT="$preflight_port" "$RUNTIME_PYTHON" "$SCRIPT_DIR/ds4_nccl_preflight.py"
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
