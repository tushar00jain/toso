"""Live TorchStore example with one role per rank, for the Rust ``toso-tui``.

An async-RL post-training job has three kinds of participant. This example maps
each to its own ``torchrun`` rank so the topology looks like it would in practice
— separate processes, each hosting its own storage volume (``LocalRankStrategy``
places one volume per rank), talking only through the shared store:

  * ``Trainer`` — each step overwrites its policy tensors under ``trainer.<i>.*``,
    bumps step/metadata, and periodically rotates an optimizer key in/out. A pure
    producer: it only ``ts.put``/``ts.delete`` and never knows the aggregator or
    TUI exists. Its writes land on its own rank's volume.
  * ``Generator`` — pulls its source trainer's latest policy version, then writes
    fresh rollout tensors under ``generator.<j>.*``. Producer + a guarded reader.
  * ``Aggregator`` — the §5 query server (SPEC §5) the TUI connects to over a TCP
    socket. It writes nothing; it READS the whole store (every rank's volume) via
    the ``toso_store_reader`` builders and answers the one endpoint the TUI dials.

Ranks
-----
``SpmdExample`` assigns roles from ``WORLD_SIZE``:

  * ``world_size >= 2`` — **one role per rank.** The LAST rank is the aggregator;
    the rest are workers, split trainers-then-generators, one per rank. So
    ``--nproc-per-node=5`` gives ranks 0–1 = trainers, 2–3 = generators, 4 =
    aggregator, and the TUI sees 5 volumes with each worker's keys on its own
    volume (the aggregator's volume stays empty).
  * ``world_size == 1`` — every role collapses onto rank 0 (one process runs all
    trainers, all generators, and the aggregator). Handy for a quick local run.

Every rank hosts a storage volume regardless of role; "role" is only what a rank
*does* on top of the store it helps host.

Reads are point-in-time (SPEC §9): a concurrent workload write/delete can race a
key out from under a read, so counts may be approximate and racing keys are
skipped — a write in progress never crashes a read, and one bad request never
kills the connection or the server (each request is guarded → ``{"error": ...}``).

Run (one role per rank — 2 trainers + 2 generators + 1 aggregator)::

    cd toso
    uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=5 \\
        live_example.py --port 8099

or collapse every role onto a single rank::

    uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=1 \\
        live_example.py --port 8099

The aggregator (the last rank) logs ``aggregator listening on host:port``. Point
the TUI at it::

    cd tui
    cargo run --offline --bin toso-tui -- --aggregator 127.0.0.1:8099 --refresh 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
from typing import Any

import torch
import torch.distributed as dist
import torchstore as ts
from toso_store_reader import (
    build_expand_prefix,
    build_key,
    build_list_volumes,
    build_peek,
    build_search,
    build_summary,
)

logger: logging.Logger = logging.getLogger(__name__)

TRAINER_INTERVAL_S: float = 10
# Only used when a single rank collapses every role (world_size == 1).
NUM_TRAINERS: int = 2
NUM_GENERATORS: int = 2


# ---------------------------------------------------------------------------
# Role: STORE
# ---------------------------------------------------------------------------


class Store:
    """The shared TorchStore this rank helps host (one local volume per rank).

    Owns the store lifecycle — attach to the torchrun process group, hand the
    aggregator its read handles, and tear down. Per-key reads/writes are done by
    the other roles through the module-level ``ts.*`` API against this store.
    """

    async def initialize(self) -> None:
        """Attach TorchStore to the existing torchrun process group."""
        dist.init_process_group("gloo")
        await ts.initialize_spmd(strategy=ts.LocalRankStrategy())

    async def read_handles(self) -> tuple[Any, Any]:
        """The ``(controller, strategy)`` introspection handles the aggregator
        reads the store through (never used to mutate)."""
        client = await ts.client()
        return client._controller, client.strategy

    async def shutdown(self) -> None:
        await ts.shutdown()
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Role: WORKLOAD (trainers + generators)
# ---------------------------------------------------------------------------


class Trainer:
    """One trainer: every step overwrites its policy tensors, bumps metadata, and
    periodically rotates an optimizer key in/out so the keyspace grows/shrinks.

    A pure producer — it only ``ts.put``/``ts.delete`` and never knows the
    aggregator or TUI exists.
    """

    # Toggle the optimizer key in/out on this step cadence so the keyspace
    # visibly grows and shrinks while the TUI is connected.
    ROTATE_EVERY: int = 5

    def __init__(self, tid: int, interval: float) -> None:
        self.tid = tid
        self.interval = interval
        self.prefix = f"trainer.{tid}"
        self.label = self.prefix

    async def seed(self) -> None:
        """Write the step-0 baseline so the aggregator sees data before step 1."""
        await ts.put(f"{self.prefix}.step", torch.tensor([0], dtype=torch.int64))
        await ts.put(f"{self.prefix}.policy.layer0.weight", torch.randn(64, 64))
        await ts.put(f"{self.prefix}.policy.layer1.weight", torch.randn(32, 32))
        await ts.put(f"{self.prefix}.metadata", {"step": 0, "lr": 0.01})

    async def _rotate_key(self, key: str, present: bool) -> bool:
        """Toggle ``key`` in/out of the store; return its new presence state."""
        if present:
            await ts.delete(key)
        else:
            await ts.put(key, torch.randn(16, 16))
        return not present

    async def run(self) -> None:
        step = 0
        momentum_present = False
        while True:
            step += 1
            await ts.put(f"{self.prefix}.step", torch.tensor([step], dtype=torch.int64))
            await ts.put(f"{self.prefix}.policy.layer0.weight", torch.randn(64, 64))
            await ts.put(f"{self.prefix}.policy.layer1.weight", torch.randn(32, 32))
            await ts.put(
                f"{self.prefix}.metadata", {"step": step, "lr": 0.01, "loss": 1.0 / step}
            )

            if step % self.ROTATE_EVERY == 0:
                momentum_present = await self._rotate_key(
                    f"{self.prefix}.optimizer.momentum", momentum_present
                )

            logger.info(
                "trainer %d step %d (momentum %s)",
                self.tid,
                step,
                "present" if momentum_present else "absent",
            )
            await asyncio.sleep(self.interval)


class Generator:
    """One generator: syncs its source trainer's latest policy version, then
    writes fresh rollout tensors. Producer plus a guarded reader (SPEC §9)."""

    def __init__(self, gid: int, source_trainer: int, interval: float) -> None:
        self.gid = gid
        self.source_trainer = source_trainer
        self.interval = interval
        self.prefix = f"generator.{gid}"
        self.label = self.prefix

    async def seed(self) -> None:
        """Write the step-0 baseline so the aggregator sees data before step 1."""
        await ts.put(f"{self.prefix}.step", torch.tensor([0], dtype=torch.int64))
        await ts.put(f"{self.prefix}.rollout.tokens", torch.randint(0, 32000, (8, 128)))
        await ts.put(f"{self.prefix}.rollout.rewards", torch.randn(8))
        await ts.put(f"{self.prefix}.metadata", {"step": 0, "policy_version": 0})

    async def _read_policy_version(self) -> int:
        """Pull the source trainer's latest step (its "policy version").

        A guarded read (SPEC §9): the trainer may not have written yet, or a key
        can race out from under this read — return -1 rather than crashing.
        """
        try:
            version = await ts.get(f"trainer.{self.source_trainer}.step")
            return int(version.item())
        except Exception:
            logger.debug(
                "generator could not read trainer.%d.step yet", self.source_trainer
            )
            return -1

    async def run(self) -> None:
        step = 0
        # Offset generators half a step so their writes interleave with trainers'.
        await asyncio.sleep(self.interval / 2)
        while True:
            step += 1
            policy_version = await self._read_policy_version()
            await ts.put(f"{self.prefix}.step", torch.tensor([step], dtype=torch.int64))
            await ts.put(f"{self.prefix}.rollout.tokens", torch.randint(0, 32000, (8, 128)))
            await ts.put(f"{self.prefix}.rollout.rewards", torch.randn(8))
            await ts.put(
                f"{self.prefix}.metadata",
                {
                    "step": step,
                    "policy_version": policy_version,
                    "source": f"trainer.{self.source_trainer}",
                },
            )

            logger.info(
                "generator %d step %d (policy v%d from trainer %d)",
                self.gid,
                step,
                policy_version,
                self.source_trainer,
            )
            await asyncio.sleep(self.interval)


# ---------------------------------------------------------------------------
# Role: AGGREGATOR (§5 query server)
# ---------------------------------------------------------------------------


class Aggregator:
    """The §5 query server the TUI connects to.

    READ-ONLY over the store: each request is answered by the
    ``toso_store_reader`` builders against the live controller/strategy handles,
    reading keys physically held on every rank's volume. It never writes and
    never touches the workload — the TUI's single connection lands here, never on
    a trainer. One bad request never kills the connection or the server (SPEC §9):
    every failure maps to an ``{"error": ...}`` line.
    """

    def __init__(self, controller: Any, strategy: Any) -> None:
        self.controller = controller
        self.strategy = strategy
        self._server: asyncio.AbstractServer | None = None

    async def _dispatch(self, req: dict[str, Any]) -> dict[str, Any]:
        """Route one parsed §5 request to its builder against the live store."""
        op = req.get("op")
        if op == "summary":
            return await build_summary(self.controller, self.strategy)
        if op == "expand_prefix":
            return await build_expand_prefix(self.controller, self.strategy, req)
        if op == "list_volumes":
            return await build_list_volumes(self.controller, self.strategy, req)
        if op == "key":
            return await build_key(self.controller, self.strategy, req)
        if op == "search":
            return await build_search(self.controller, self.strategy, req)
        if op == "peek":
            return await build_peek(req)
        return {"error": f"unknown op: {op!r}"}

    async def _handle_request(self, text: str) -> dict[str, Any]:
        """Parse + answer one request line, mapping any failure to an error line."""
        try:
            req = json.loads(text)
        except json.JSONDecodeError as e:
            return {"error": f"invalid request json: {e}"}
        if not isinstance(req, dict):
            return {"error": "request must be a JSON object"}

        try:
            return await self._dispatch(req)
        except KeyError as e:
            # Missing/racing key — an expected, recoverable per-request error.
            return {"error": str(e).strip("'\"")}
        except Exception as e:
            # A guarded boundary (SPEC §9): never let one request kill the server.
            logger.warning("request %r failed", text, exc_info=True)
            return {"error": f"{type(e).__name__}: {e}"}

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        peer = writer.get_extra_info("peername")
        logger.info("client connected: %s", peer)
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                text = line.decode("utf-8", "replace").strip()
                if not text:
                    continue
                response = await self._handle_request(text)
                writer.write((json.dumps(response) + "\n").encode("utf-8"))
                await writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            logger.info("client %s disconnected", peer)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionResetError, BrokenPipeError):
                pass

    async def start(self, host: str, port: int) -> None:
        """Bind the socket (already serving) and log the endpoint for the TUI."""
        self._server = await asyncio.start_server(self._handle_client, host, port)
        bound_host, bound_port = self._server.sockets[0].getsockname()[:2]
        # The user / TUI reads this line to learn the ephemeral port.
        logger.info("aggregator listening on %s:%d", bound_host, bound_port)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None


# ---------------------------------------------------------------------------
# Orchestrator: one role per rank (SPMD)
# ---------------------------------------------------------------------------


class SpmdExample:
    """Assigns one role per rank and owns the per-rank lifecycle.

    See the module docstring for the rank→role mapping. Every rank initializes
    the store (contributing its volume), seeds any workers it owns, barriers so
    the store is consistent, then runs its role until a signal arrives.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.store = Store()

    @staticmethod
    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--host", default="127.0.0.1", help="bind host")
        parser.add_argument(
            "--port",
            type=int,
            default=0,
            help="aggregator bind port (0 picks a free ephemeral port)",
        )
        parser.add_argument(
            "--interval",
            type=float,
            default=TRAINER_INTERVAL_S,
            help="seconds between trainer steps",
        )
        return parser.parse_args()

    def _plan(self) -> tuple[list[Trainer | Generator], bool]:
        """Return ``(workers, run_aggregator)`` for THIS rank.

        ``world_size == 1`` collapses every role onto rank 0. Otherwise the last
        rank is the aggregator and the rest are one worker each — trainers first,
        then generators (each generator syncs from a trainer, round-robin).
        """
        interval = self.args.interval
        if self.world_size == 1:
            workers: list[Trainer | Generator] = [
                Trainer(i, interval) for i in range(NUM_TRAINERS)
            ]
            workers += [
                Generator(j, j % NUM_TRAINERS, interval) for j in range(NUM_GENERATORS)
            ]
            return workers, True

        num_workers = self.world_size - 1
        num_trainers = (num_workers + 1) // 2  # ceil half; at least one trainer
        if self.rank == self.world_size - 1:
            return [], True  # aggregator
        if self.rank < num_trainers:
            return [Trainer(self.rank, interval)], False
        gid = self.rank - num_trainers
        return [Generator(gid, gid % num_trainers, interval)], False

    def _install_signal_handlers(self, stop: asyncio.Event) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)

    async def _await_stop(
        self, stop: asyncio.Event, tasks: list[asyncio.Task[None]]
    ) -> None:
        """Block until the stop signal — or surface a role task that died early
        (workers/servers otherwise run forever, so completion means failure)."""
        stop_task = asyncio.create_task(stop.wait())
        done, _ = await asyncio.wait(
            {stop_task, *tasks}, return_when=asyncio.FIRST_COMPLETED
        )
        stop_task.cancel()
        for task in tasks:
            if task in done and not task.cancelled() and task.exception() is not None:
                raise task.exception()  # type: ignore[misc]

    async def run(self) -> None:
        # Role: STORE — attach TorchStore to the torchrun process group.
        await self.store.initialize()
        try:
            workers, run_aggregator = self._plan()
            role = ",".join([w.label for w in workers] + (["aggregator"] if run_aggregator else []))
            logger.info("rank %d/%d role=%s", self.rank, self.world_size, role or "idle")

            controller = strategy = None
            if run_aggregator:
                controller, strategy = await self.store.read_handles()

            # Each worker seeds its own keyspace before the barrier, so once every
            # rank passes it the aggregator can already read a consistent store.
            for worker in workers:
                await worker.seed()
            dist.barrier()

            stop = asyncio.Event()
            self._install_signal_handlers(stop)

            # Role: WORKLOAD — this rank's producers (empty on the aggregator rank).
            tasks = [asyncio.create_task(worker.run()) for worker in workers]

            # Role: AGGREGATOR — start the §5 server (only on the aggregator rank).
            aggregator = None
            if run_aggregator:
                aggregator = Aggregator(controller, strategy)
                await aggregator.start(self.args.host, self.args.port)

            try:
                await self._await_stop(stop, tasks)
                logger.info("shutdown signal received; tearing down")
            finally:
                for task in tasks:
                    task.cancel()
                for task in tasks:
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                if aggregator is not None:
                    await aggregator.stop()

            # Everyone stops before any rank tears down the shared store.
            dist.barrier()
        finally:
            await self.store.shutdown()
            logger.info("rank %d clean shutdown complete", self.rank)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    asyncio.run(SpmdExample(SpmdExample.parse_args()).run())
