# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DS4 high-speed point-to-point channels.

The Spark 200G ring reaches its best host throughput by using several
parallel TCP flows. Some NCCL/PyNCCL P2P edges on GB10/SM12x show the same
single-flow ceiling, so the DS4 PP path can stripe one pipeline-boundary tensor
across several independent NCCL communicators.

This module intentionally keeps the abstraction small:

* CUDA tensors: striped NCCL communicators, used by vLLM PP send/recv.
* Byte/file transfers for DS4 tools live in the DS4 coordinator repo, but share
  the same striping contract: split a contiguous payload into deterministic
  ranges and move those ranges over independent lanes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.logger import init_logger
from vllm.utils.torch_utils import current_stream

logger = init_logger(__name__)


@dataclass(frozen=True)
class Ds4StripeRange:
    begin: int
    end: int

    @property
    def numel(self) -> int:
        return self.end - self.begin


class Ds4CudaEventHandle:
    """Async handle backed by CUDA events and tensor references."""

    def __init__(self, events: list[torch.cuda.Event], tensors: Iterable[torch.Tensor]):
        self._events = events
        self._tensors = list(tensors)

    def is_completed(self) -> bool:
        return all(event.query() for event in self._events)

    def wait(self) -> None:
        # vLLM's existing P2P handles block in wait().  Synchronizing the event
        # here preserves that contract and avoids returning a tensor whose NCCL
        # recv stream has not finished populating it yet.
        for event in self._events:
            event.synchronize()
        self._tensors.clear()


class Ds4StripedNcclTensorChannel:
    """Stripe one tensor over multiple NCCL communicators.

    A single NCCL communicator may open too few sockets on GB10/Spark ring
    links.  Creating several independent communicators lets the same tensor
    use multiple lanes without changing model math or PP metadata semantics.
    """

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device,
        stripe_count: int,
        min_striped_bytes: int,
        use_independent_streams: bool = True,
    ) -> None:
        self.device = device
        self.stripe_count = max(1, int(stripe_count))
        self.min_striped_bytes = max(0, int(min_striped_bytes))
        self.use_independent_streams = bool(use_independent_streams)
        self.communicators: list[PyNcclCommunicator] = []
        self.streams: list[torch.cuda.Stream] = []
        self.available = False

        if self.stripe_count <= 1:
            return
        if device.type != "cuda":
            return

        for stripe_index in range(self.stripe_count):
            communicator = PyNcclCommunicator(cpu_group, device)
            if communicator.disabled or not communicator.available:
                self.destroy()
                logger.warning(
                    "DS4 striped NCCL PP channel disabled: stripe %d could "
                    "not create a PyNCCL communicator.",
                    stripe_index,
                )
                return
            self.communicators.append(communicator)
            if self.use_independent_streams:
                with torch.cuda.device(device):
                    self.streams.append(torch.cuda.Stream(device=device))

        self.available = True
        logger.info(
            "DS4 striped NCCL PP tensor channel enabled: stripes=%d "
            "min_bytes=%d independent_streams=%s",
            self.stripe_count,
            self.min_striped_bytes,
            self.use_independent_streams,
        )

    def destroy(self) -> None:
        for communicator in self.communicators:
            communicator.destroy()
        self.communicators.clear()
        self.streams.clear()
        self.available = False

    def can_handle(self, tensor: torch.Tensor) -> bool:
        if not self.available:
            return False
        if not tensor.is_cuda:
            return False
        if tensor.device != self.device:
            return False
        if not tensor.is_contiguous():
            return False
        if tensor.numel() == 0:
            return False
        return tensor.numel() * tensor.element_size() >= self.min_striped_bytes

    def _stripe_ranges(self, numel: int) -> list[Ds4StripeRange]:
        stripe_count = min(self.stripe_count, numel)
        base = numel // stripe_count
        rem = numel % stripe_count
        ranges: list[Ds4StripeRange] = []
        begin = 0
        for stripe_index in range(stripe_count):
            stripe_numel = base + (1 if stripe_index < rem else 0)
            end = begin + stripe_numel
            ranges.append(Ds4StripeRange(begin=begin, end=end))
            begin = end
        return ranges

    def send(self, tensor: torch.Tensor, dst: int) -> Ds4CudaEventHandle:
        assert self.can_handle(tensor)
        flat = tensor.view(-1)
        events: list[torch.cuda.Event] = []
        tensors: list[torch.Tensor] = [tensor]
        producer_stream = current_stream()
        ranges = self._stripe_ranges(flat.numel())

        for stripe_index, stripe_range in enumerate(ranges):
            communicator = self.communicators[stripe_index]
            chunk = flat[stripe_range.begin : stripe_range.end]
            tensors.append(chunk)
            if self.use_independent_streams:
                stream = self.streams[stripe_index]
            else:
                stream = producer_stream
            with torch.cuda.stream(stream):
                if stream is not producer_stream:
                    stream.wait_stream(producer_stream)
                communicator.send(chunk, dst, stream=stream)
                event = torch.cuda.Event()
                event.record(stream)
                events.append(event)
            tensor.record_stream(stream)

        return Ds4CudaEventHandle(events, tensors)

    def recv(self, tensor: torch.Tensor, src: int) -> Ds4CudaEventHandle:
        assert self.can_handle(tensor)
        flat = tensor.view(-1)
        events: list[torch.cuda.Event] = []
        tensors: list[torch.Tensor] = [tensor]
        consumer_stream = current_stream()
        ranges = self._stripe_ranges(flat.numel())

        for stripe_index, stripe_range in enumerate(ranges):
            communicator = self.communicators[stripe_index]
            chunk = flat[stripe_range.begin : stripe_range.end]
            tensors.append(chunk)
            if self.use_independent_streams:
                stream = self.streams[stripe_index]
            else:
                stream = consumer_stream
            with torch.cuda.stream(stream):
                communicator.recv(chunk, src, stream=stream)
                event = torch.cuda.Event()
                event.record(stream)
                events.append(event)
            tensor.record_stream(stream)

        return Ds4CudaEventHandle(events, tensors)


def build_ds4_pp_striped_nccl_channel(
    *,
    group_name: str,
    cpu_group: ProcessGroup,
    device: torch.device,
) -> Ds4StripedNcclTensorChannel | None:
    if group_name != "pp":
        return None
    if not envs.VLLM_DS4_PP_STRIPED_NCCL_TENSOR_DICT:
        return None
    if envs.VLLM_DS4_PP_DISABLE_DEVICE_COMMUNICATOR:
        logger.warning(
            "VLLM_DS4_PP_STRIPED_NCCL_TENSOR_DICT=1 ignored because "
            "VLLM_DS4_PP_DISABLE_DEVICE_COMMUNICATOR=1."
        )
        return None
    stripe_count = max(1, envs.VLLM_DS4_PP_STRIPED_NCCL_STRIPES)
    if stripe_count <= 1:
        return None
    try:
        return Ds4StripedNcclTensorChannel(
            cpu_group=cpu_group,
            device=device,
            stripe_count=stripe_count,
            min_striped_bytes=envs.VLLM_DS4_PP_STRIPED_NCCL_MIN_BYTES,
            use_independent_streams=envs.VLLM_DS4_PP_STRIPED_NCCL_STREAMS,
        )
    except Exception:
        logger.exception("Failed to create DS4 striped NCCL PP channel.")
        if envs.VLLM_DS4_STRICT_NATIVE_FP4:
            raise
        return None
