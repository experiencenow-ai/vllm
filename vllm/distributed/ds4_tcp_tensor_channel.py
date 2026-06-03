# SPDX-License-Identifier: Apache-2.0
"""DS4 PP tensor payload transport over striped rail TCP sockets.

This is intentionally narrow: vLLM still sends tensor-dict metadata through the
normal PP CPU control group, while large CUDA payloads are staged to CPU and
moved across the already-validated DS4 rail routes with multiple TCP streams.
"""

from __future__ import annotations

import socket
import threading
from collections.abc import Callable
from typing import Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_MAGIC = "ds4_pp_tcp_tensor_v1"


class Ds4TcpTensorHandle:
    """Thread-backed handle matching the small vLLM PP handle protocol."""

    def __init__(self, threads: list[threading.Thread], tensors: list[torch.Tensor]):
        self._threads = threads
        self._tensors = tensors
        self._errors: list[BaseException] = []

    def _record_error(self, exc: BaseException) -> None:
        self._errors.append(exc)

    def is_completed(self) -> bool:
        return all(not thread.is_alive() for thread in self._threads)

    def wait(self) -> None:
        for thread in self._threads:
            thread.join()
        self._tensors = []
        if self._errors:
            raise RuntimeError("DS4 PP TCP tensor transfer failed") from self._errors[0]


class Ds4TcpTensorChannel:
    """Striped TCP channel for one PP process rank."""

    def __init__(
        self,
        *,
        rank: int,
        rank_in_group: int,
        send_control: Callable[[Any, int], None],
        recv_control: Callable[[int], Any],
    ) -> None:
        self.rank = rank
        self.rank_in_group = rank_in_group
        self._send_control = send_control
        self._recv_control = recv_control
        self._send_seq: dict[int, int] = {}
        self._recv_seq: dict[int, int] = {}
        self._seq_lock = threading.Lock()

    def can_handle(self, tensor: torch.Tensor) -> bool:
        if not envs.VLLM_DS4_PP_TCP_TENSOR_DICT:
            return False
        if tensor.numel() <= 0:
            return False
        byte_size = tensor.numel() * tensor.element_size()
        return byte_size >= int(envs.VLLM_DS4_PP_TCP_MIN_BYTES)

    def send(self, tensor: torch.Tensor, dst: int) -> Ds4TcpTensorHandle:
        cpu_tensor = self._cpu_contiguous(tensor)
        byte_view = self._byte_view(cpu_tensor)
        byte_count = len(byte_view)
        seq = self._next_send_seq(dst)
        ready = self._recv_control(dst)
        self._check_ready(ready, byte_count, seq, dst)
        ranges = self._stripe_ranges(byte_count, len(ready["ports"]))
        handle = Ds4TcpTensorHandle([], [cpu_tensor])
        for index, (start, end) in enumerate(ranges):
            thread = threading.Thread(
                target=self._send_stripe,
                args=(handle, ready["host"], int(ready["ports"][index]),
                      byte_view[start:end]),
                name=f"ds4-pp-tcp-send-{self.rank_in_group}-{dst}-{seq}-{index}",
                daemon=True,
            )
            handle._threads.append(thread)
            thread.start()
        return handle

    def recv(self, tensor: torch.Tensor, src: int) -> Ds4TcpTensorHandle:
        cpu_tensor = self._cpu_contiguous(tensor)
        byte_view = self._byte_view(cpu_tensor)
        byte_count = len(byte_view)
        seq = self._next_recv_seq(src)
        stripe_count = self._stripe_count(byte_count)
        listeners: list[socket.socket] = []
        ports: list[int] = []
        host = envs.VLLM_DS4_PP_TCP_BIND_HOST or envs.VLLM_HOST_IP or "0.0.0.0"
        for _ in range(stripe_count):
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((host, 0))
            listener.listen(1)
            listeners.append(listener)
            ports.append(int(listener.getsockname()[1]))
        ranges = self._stripe_ranges(byte_count, stripe_count)
        handle = Ds4TcpTensorHandle([], [cpu_tensor])
        for index, (start, end) in enumerate(ranges):
            thread = threading.Thread(
                target=self._recv_stripe,
                args=(handle, listeners[index], byte_view[start:end]),
                name=f"ds4-pp-tcp-recv-{src}-{self.rank_in_group}-{seq}-{index}",
                daemon=True,
            )
            handle._threads.append(thread)
            thread.start()
        advertised_host = envs.VLLM_DS4_PP_TCP_ADVERTISE_HOST or envs.VLLM_HOST_IP
        if not advertised_host:
            advertised_host = socket.gethostbyname(socket.gethostname())
        self._send_control(
            {
                "magic": _MAGIC,
                "seq": seq,
                "rank": self.rank,
                "rank_in_group": self.rank_in_group,
                "bytes": byte_count,
                "host": advertised_host,
                "ports": ports,
            },
            src,
        )
        return handle

    def _next_send_seq(self, dst: int) -> int:
        with self._seq_lock:
            seq = self._send_seq.get(dst, 0)
            self._send_seq[dst] = seq + 1
            return seq

    def _next_recv_seq(self, src: int) -> int:
        with self._seq_lock:
            seq = self._recv_seq.get(src, 0)
            self._recv_seq[src] = seq + 1
            return seq

    def _stripe_count(self, byte_count: int) -> int:
        stripes = max(1, int(envs.VLLM_DS4_PP_TCP_STRIPES))
        stripes = min(stripes, max(1, byte_count))
        return stripes

    def _stripe_ranges(self, byte_count: int, stripes: int) -> list[tuple[int, int]]:
        stripes = min(max(1, stripes), max(1, byte_count))
        base = byte_count // stripes
        rem = byte_count % stripes
        out: list[tuple[int, int]] = []
        offset = 0
        for index in range(stripes):
            length = base + (1 if index < rem else 0)
            out.append((offset, offset + length))
            offset += length
        return out

    def _cpu_contiguous(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_cuda:
            return tensor.detach().to("cpu").contiguous()
        if not tensor.is_contiguous():
            return tensor.contiguous()
        return tensor

    def _byte_view(self, tensor: torch.Tensor) -> memoryview:
        if tensor.device.type != "cpu":
            raise RuntimeError("DS4 PP TCP tensor channel requires CPU tensors")
        try:
            byte_tensor = tensor.view(torch.uint8).reshape(-1)
            return memoryview(byte_tensor.numpy())
        except Exception as exc:
            raise RuntimeError(
                f"DS4 PP TCP tensor channel cannot byte-view dtype={tensor.dtype} "
                f"shape={tuple(tensor.shape)}"
            ) from exc

    def _check_ready(
        self,
        ready: Any,
        byte_count: int,
        seq: int,
        dst: int,
    ) -> None:
        if not isinstance(ready, dict) or ready.get("magic") != _MAGIC:
            raise RuntimeError(
                f"DS4 PP TCP tensor channel expected ready dict from {dst}, got "
                f"{type(ready).__name__}"
            )
        if int(ready.get("seq", -1)) != seq:
            raise RuntimeError(
                f"DS4 PP TCP tensor send/recv sequence mismatch for dst={dst}: "
                f"receiver_ready_seq={ready.get('seq')} sender_seq={seq}. "
                "Sequence counters are per peer and per direction; this usually "
                "means one side posted a different PP tensor payload order."
            )
        if int(ready.get("bytes", -1)) != byte_count:
            raise RuntimeError(
                f"DS4 PP TCP tensor byte mismatch for dst={dst}: "
                f"{ready.get('bytes')} != {byte_count}"
            )
        ports = ready.get("ports")
        if not isinstance(ports, list) or not ports:
            raise RuntimeError(f"DS4 PP TCP tensor ready from {dst} has no ports")

    def _send_stripe(
        self,
        handle: Ds4TcpTensorHandle,
        host: str,
        port: int,
        payload: memoryview,
    ) -> None:
        try:
            timeout = float(envs.VLLM_DS4_PP_TCP_CONNECT_TIMEOUT_SECONDS)
            with socket.create_connection((host, port), timeout=timeout) as sock:
                if envs.VLLM_DS4_PP_TCP_NODELAY:
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.sendall(payload)
        except BaseException as exc:
            handle._record_error(exc)

    def _recv_stripe(
        self,
        handle: Ds4TcpTensorHandle,
        listener: socket.socket,
        target: memoryview,
    ) -> None:
        try:
            timeout = float(envs.VLLM_DS4_PP_TCP_CONNECT_TIMEOUT_SECONDS)
            listener.settimeout(timeout)
            try:
                conn, _ = listener.accept()
            finally:
                listener.close()
            with conn:
                if envs.VLLM_DS4_PP_TCP_NODELAY:
                    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                conn.settimeout(float(envs.VLLM_DS4_PP_TCP_READ_TIMEOUT_SECONDS))
                offset = 0
                while offset < len(target):
                    nread = conn.recv_into(target[offset:])
                    if nread == 0:
                        raise RuntimeError(
                            "DS4 PP TCP tensor channel saw EOF before full stripe"
                        )
                    offset += nread
        except BaseException as exc:
            handle._record_error(exc)


def build_ds4_pp_tcp_tensor_channel(
    *,
    group_name: str,
    rank: int,
    rank_in_group: int,
    send_control: Callable[[Any, int], None],
    recv_control: Callable[[int], Any],
) -> Ds4TcpTensorChannel | None:
    if "pp" not in group_name:
        return None
    if not envs.VLLM_DS4_PP_TCP_TENSOR_DICT:
        return None
    logger.info(
        "DS4 PP TCP tensor channel enabled: group=%s rank=%s pp_rank=%s "
        "stripes=%s min_bytes=%s bind_host=%s advertise_host=%s",
        group_name,
        rank,
        rank_in_group,
        envs.VLLM_DS4_PP_TCP_STRIPES,
        envs.VLLM_DS4_PP_TCP_MIN_BYTES,
        envs.VLLM_DS4_PP_TCP_BIND_HOST or envs.VLLM_HOST_IP or "0.0.0.0",
        envs.VLLM_DS4_PP_TCP_ADVERTISE_HOST or envs.VLLM_HOST_IP or "<auto>",
    )
    return Ds4TcpTensorChannel(
        rank=rank,
        rank_in_group=rank_in_group,
        send_control=send_control,
        recv_control=recv_control,
    )
