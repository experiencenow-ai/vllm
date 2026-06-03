#!/usr/bin/env bash
set -euo pipefail

DEFAULT_NODES="spark0,spark1,spark2,spark3,spark4,spark5,spark6,spark7"
MODE="local"
NODES="${DS4_SPARK_NODES:-$DEFAULT_NODES}"
SOURCE_ROOT="${DS4_VLLM_SOURCE_ROOT:-\$HOME/src/vllm}"
REMOTE="${DS4_VLLM_REMOTE:-origin}"
BRANCH="${DS4_VLLM_BRANCH:-main}"
HELPER_PATH="${DS4_STATIC_FABRIC_ROOT_HELPER:-/usr/local/sbin/ds4-static-fabric-root}"
INSTALL_USER="${DS4_STATIC_FABRIC_USER:-}"
DRY_RUN=0
DS4_INSTALL_TMPDIR=""

cleanup_tmpdir()
{
	if [[ -n "${DS4_INSTALL_TMPDIR:-}" ]]; then
		rm -rf "$DS4_INSTALL_TMPDIR"
	fi
}

usage()
{
	cat <<USAGE
usage: $0 [--local|--fleet] [options]

Install the constrained DS4 static-fabric root helper and sudoers rule.

Modes:
  --local                  install on the current Spark (default)
  --fleet                  ssh to every Spark and run the local installer

Options:
  --nodes CSV              fleet nodes (default: $DEFAULT_NODES)
  --source-root PATH       remote vLLM checkout (default: \$HOME/src/vllm)
  --remote NAME            git remote to pull in fleet mode (default: origin)
  --branch NAME            git branch to pull in fleet mode (default: main)
  --user USER              sudoers user; default is the current user on each Spark
  --helper-path PATH       installed root helper path
  --dry-run                print actions without installing

Typical Mac-side use after this PR is merged:

  bash tools/ds4_install_static_fabric_sudoers.sh --fleet

Then the normal post-power-cycle fabric restore can run without prompts:

  python tools/ds4_relaunch_spark_service.py --service dsv4-pp8 \\
    --static-fabric apply --static-fabric-edge-rail enp \\
    --static-fabric-route-scope all --setup-only
USAGE
}

shell_quote()
{
	printf "%q" "$1"
}

remote_cd_command()
{
	if [[ "$SOURCE_ROOT" == "~/"* ]]; then
		printf 'cd "$HOME/%s"' "${SOURCE_ROOT#~/}"
	elif [[ "$SOURCE_ROOT" == "\$HOME/"* ]]; then
		printf 'cd "$HOME/%s"' "${SOURCE_ROOT#\$HOME/}"
	else
		printf 'cd %s' "$(shell_quote "$SOURCE_ROOT")"
	fi
}

validate_user()
{
	local user="$1"
	if [[ ! "$user" =~ ^[A-Za-z_][A-Za-z0-9_-]*$ ]]; then
		echo "invalid sudoers user: $user" >&2
		exit 2
	fi
}

validate_helper_path()
{
	if [[ "$HELPER_PATH" != /* || "$HELPER_PATH" =~ [[:space:]] ]]; then
		echo "helper path must be an absolute path without whitespace: $HELPER_PATH" >&2
		exit 2
	fi
}

write_helper()
{
	local output="$1"
	cat > "$output" <<'HELPER'
#!/usr/bin/env bash
set -euo pipefail

RANK=""
NNODES="8"
ROUTE_SCOPE="all"
EDGE_RAIL="enp"
LOOPBACK_DEV="ds4ring0"

usage()
{
	cat <<USAGE
usage: $0 --rank N [--nnodes N] [--route-scope all|adjacent|head] [--edge-rail enp|enP2p|prev,next] [--loopback-dev DEV]
       $0 --check
USAGE
}

if [[ "${1:-}" == "--check" ]]; then
	exit 0
fi

while [[ $# -gt 0 ]]; do
	case "$1" in
		--rank)
			RANK="$2"
			shift 2
			;;
		--nnodes)
			NNODES="$2"
			shift 2
			;;
		--route-scope)
			ROUTE_SCOPE="$2"
			shift 2
			;;
		--edge-rail)
			EDGE_RAIL="$2"
			shift 2
			;;
		--loopback-dev)
			LOOPBACK_DEV="$2"
			shift 2
			;;
		-h|--help)
			usage
			exit 0
			;;
		*)
			echo "unknown argument: $1" >&2
			usage >&2
			exit 2
			;;
	esac
done

if [[ -z "$RANK" || ! "$RANK" =~ ^[0-9]+$ || ! "$NNODES" =~ ^[0-9]+$ ]]; then
	echo "--rank and --nnodes must be non-negative integers" >&2
	exit 2
fi
if (( RANK >= NNODES )); then
	echo "rank $RANK is outside nnodes=$NNODES" >&2
	exit 2
fi

case "$ROUTE_SCOPE" in
	all|adjacent|head)
		;;
	*)
		echo "invalid route scope: $ROUTE_SCOPE" >&2
		exit 2
		;;
esac

case "$EDGE_RAIL" in
	enp|rail0|lower)
		PREV_DEV="enp1s0f0np0"
		NEXT_DEV="enp1s0f1np1"
		;;
	enP2p|enp2p|rail1|upper)
		PREV_DEV="enP2p1s0f0np0"
		NEXT_DEV="enP2p1s0f1np1"
		;;
	*,*)
		PREV_DEV="${EDGE_RAIL%%,*}"
		NEXT_DEV="${EDGE_RAIL#*,}"
		;;
	*)
		echo "invalid edge rail: $EDGE_RAIL" >&2
		exit 2
		;;
esac

if [[ -z "$PREV_DEV" || -z "$NEXT_DEV" || "$PREV_DEV" =~ [[:space:]] || "$NEXT_DEV" =~ [[:space:]] ]]; then
	echo "invalid edge rail devices: $EDGE_RAIL" >&2
	exit 2
fi

IP_BIN="${IP_BIN:-/usr/sbin/ip}"
SYSCTL_BIN="${SYSCTL_BIN:-/usr/sbin/sysctl}"
if [[ ! -x "$IP_BIN" ]]; then
	IP_BIN="$(command -v ip)"
fi
if [[ ! -x "$SYSCTL_BIN" ]]; then
	SYSCTL_BIN="$(command -v sysctl)"
fi

fabric_ip()
{
	printf '10.10.100.%d' "$((10 + $1))"
}

route_allowed()
{
	local source_rank="$1"
	local target_rank="$2"
	local delta
	if [[ "$ROUTE_SCOPE" == "adjacent" ]]; then
		if (( target_rank > source_rank )); then
			delta=$((target_rank - source_rank))
		else
			delta=$((source_rank - target_rank))
		fi
		(( delta == 1 ))
		return
	fi
	if [[ "$ROUTE_SCOPE" == "head" ]]; then
		(( source_rank == 0 || target_rank == 0 ))
		return
	fi
	return 0
}

if ! "$IP_BIN" link show "$LOOPBACK_DEV" >/dev/null 2>&1; then
	"$IP_BIN" link add "$LOOPBACK_DEV" type dummy
fi
"$IP_BIN" link set "$LOOPBACK_DEV" up
"$IP_BIN" addr replace "$(fabric_ip "$RANK")/32" dev "$LOOPBACK_DEV"

if (( RANK > 0 )); then
	SUBNET=$((RANK * 2))
	"$IP_BIN" link set "$PREV_DEV" up
	"$IP_BIN" addr replace "10.10.${SUBNET}.2/30" dev "$PREV_DEV"
fi
if (( RANK < (NNODES - 1) )); then
	SUBNET=$(((RANK + 1) * 2))
	"$IP_BIN" link set "$NEXT_DEV" up
	"$IP_BIN" addr replace "10.10.${SUBNET}.1/30" dev "$NEXT_DEV"
fi

if [[ "$ROUTE_SCOPE" != "adjacent" ]]; then
	"$SYSCTL_BIN" -w net.ipv4.ip_forward=1 >/dev/null
fi

for ((target=0; target<NNODES; target++)); do
	if (( target == RANK )); then
		continue
	fi
	if ! route_allowed "$RANK" "$target"; then
		continue
	fi
	if (( target > RANK )); then
		SUBNET=$(((RANK + 1) * 2))
		VIA="10.10.${SUBNET}.2"
		DEV="$NEXT_DEV"
	else
		SUBNET=$((RANK * 2))
		VIA="10.10.${SUBNET}.1"
		DEV="$PREV_DEV"
	fi
	"$IP_BIN" route replace "$(fabric_ip "$target")" via "$VIA" dev "$DEV" src "$(fabric_ip "$RANK")"
done

echo "PASS ds4 static fabric root helper rank=${RANK} loopback=$(fabric_ip "$RANK") route_scope=${ROUTE_SCOPE} edge_rail=${EDGE_RAIL}"
HELPER
}

install_local()
{
	local install_user="$INSTALL_USER"
	local tmpdir helper_tmp sudoers_tmp sudoers_path
	validate_helper_path
	if [[ -z "$install_user" ]]; then
		install_user="$(id -un)"
	fi
	validate_user "$install_user"
	sudoers_path="/etc/sudoers.d/ds4-static-fabric"
	if (( DRY_RUN )); then
		echo "would install helper: $HELPER_PATH"
		echo "would install sudoers: $sudoers_path for user $install_user"
		return 0
	fi
	tmpdir="$(mktemp -d)"
	DS4_INSTALL_TMPDIR="$tmpdir"
	trap cleanup_tmpdir EXIT
	helper_tmp="$tmpdir/ds4-static-fabric-root"
	sudoers_tmp="$tmpdir/ds4-static-fabric-sudoers"
	write_helper "$helper_tmp"
	chmod 0755 "$helper_tmp"
	cat > "$sudoers_tmp" <<SUDOERS
# DS4 static fabric post-power-cycle helper.
# This grants only the root-owned helper, not broad passwordless ip/sysctl.
${install_user} ALL=(root) NOPASSWD: ${HELPER_PATH} *
SUDOERS
	echo "Installing DS4 static fabric helper for user ${install_user}"
	sudo -v
	sudo /usr/sbin/visudo -cf "$sudoers_tmp" >/dev/null
	sudo /usr/bin/install -o root -g root -m 0755 "$helper_tmp" "$HELPER_PATH"
	sudo /usr/bin/install -o root -g root -m 0440 "$sudoers_tmp" "$sudoers_path"
	sudo /usr/sbin/visudo -cf "$sudoers_path" >/dev/null
	sudo -k
	sudo -n "$HELPER_PATH" --check
	echo "PASS installed DS4 static fabric sudo helper: user=${install_user} helper=${HELPER_PATH}"
}

install_fleet()
{
	local node remote_cmd cd_cmd user_arg dry_arg helper_arg
	IFS=',' read -r -a node_array <<< "$NODES"
	for node in "${node_array[@]}"; do
		node="${node//[[:space:]]/}"
		if [[ -z "$node" ]]; then
			continue
		fi
		cd_cmd="$(remote_cd_command)"
		user_arg=""
		if [[ -n "$INSTALL_USER" ]]; then
			user_arg=" --user $(shell_quote "$INSTALL_USER")"
		fi
		dry_arg=""
		if (( DRY_RUN )); then
			dry_arg=" --dry-run"
		fi
		helper_arg=" --helper-path $(shell_quote "$HELPER_PATH")"
		remote_cmd="${cd_cmd} && git pull --ff-only $(shell_quote "$REMOTE") $(shell_quote "$BRANCH") && bash tools/ds4_install_static_fabric_sudoers.sh --local${helper_arg}${user_arg}${dry_arg}"
		echo "== ${node} == ${remote_cmd}"
		if (( DRY_RUN )); then
			continue
		fi
		ssh -tt "$node" "$remote_cmd"
	done
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--local)
			MODE="local"
			shift
			;;
		--fleet)
			MODE="fleet"
			shift
			;;
		--nodes)
			NODES="$2"
			shift 2
			;;
		--source-root)
			SOURCE_ROOT="$2"
			shift 2
			;;
		--remote)
			REMOTE="$2"
			shift 2
			;;
		--branch)
			BRANCH="$2"
			shift 2
			;;
		--user)
			INSTALL_USER="$2"
			shift 2
			;;
		--helper-path)
			HELPER_PATH="$2"
			shift 2
			;;
		--dry-run)
			DRY_RUN=1
			shift
			;;
		-h|--help)
			usage
			exit 0
			;;
		*)
			echo "unknown argument: $1" >&2
			usage >&2
			exit 2
			;;
	esac
done

case "$MODE" in
	local)
		install_local
		;;
	fleet)
		install_fleet
		;;
	*)
		echo "invalid mode: $MODE" >&2
		exit 2
		;;
esac
