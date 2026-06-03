# DS4 Static Spark Fabric

The Spark fabric is now treated as two separate things:

- Static OS state: `ds4ring0` loopback aliases, line-fabric host routes,
  forwarding, and fixed interface choices. Apply this once after a Spark power
  cycle.
- Runtime communicators: PyTorch/NCCL/Gloo/PyNCCL process groups. These are
  recreated by each vLLM launch, but they must use the fixed static profile.

Normal service relaunch verifies the static profile before stopping an existing
service. It does not repair routes unless explicitly requested.

## Post-Power-Cycle Bootstrap

Preferred: run from the Mac vLLM checkout after a Spark power cycle:

```bash
python tools/ds4_relaunch_spark_service.py \
    --service dsv4-pp8 \
    --static-fabric apply \
    --static-fabric-edge-rail enp \
    --static-fabric-route-scope all \
    --setup-only
```

That performs the zero-drift setup path:

```text
git pull on every Spark
build/static audits on every Spark
apply the fixed fabric profile once
exit without stopping or launching a model
```

Lower-level direct bootstrap, from the vLLM checkout on `spark0`:

```bash
python tools/ds4_static_fabric.py \
    --fleet \
    --apply \
    --edge-rail enp \
    --route-scope all
```

The script uses `sudo -n`; if passwordless sudo is not configured on a Spark,
it fails immediately instead of hanging.

Then verify:

```bash
python tools/ds4_static_fabric.py \
    --fleet \
    --verify \
    --edge-rail enp \
    --route-scope all
```

`--route-scope all` installs deterministic `/32` routes for every
`10.10.100.x` Spark fabric alias through the correct next hop on the open line.
The PP data plane still uses adjacent edges; the all-route profile also keeps
rank-0 rendezvous/control traffic off Wi-Fi and other accidental paths.

## Launch Behavior

`tools/ds4_relaunch_spark_service.py` defaults to:

```text
--static-fabric verify
--static-fabric-edge-rail enp
--static-fabric-route-scope all
```

It exports fixed launch bindings:

```text
DS4_200G_IFNAME=enp1s0f0np0,enp1s0f1np1
DS4_200G_SOCKET_IFNAME=enp1s0f0np0,enp1s0f1np1
DS4_200G_NCCL_IFNAME=enp1s0f0np0,enp1s0f1np1
DS4_CONTROL_IFNAME=ds4ring0
DS4_GLOO_SOCKET_IFNAME=enP7s7
DS4_200G_ADVERTISE_LOOPBACK=1
DS4_200G_NCCL_TRANSPORT=socket
VLLM_DS4_PP_EDGE_RAIL=enp
```

If verification fails, relaunch stops before killing the existing service and
prints the exact bootstrap command to run.

Use `--static-fabric off` only for deliberate diagnosis.
