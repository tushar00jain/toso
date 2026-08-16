"""Synthetic conversations (seeded, deterministic), and the block keys their turns
address prompts with.

Models the target workload shape: many requests that **share prefixes**, and share
them the way a chat product's traffic really does -- by being *turns of the same
conversation*. A conversation opens with

* a **system prompt** of ``system_blocks`` blocks common to *every* request (the
  hottest shared prefix), and
* a per-**conversation** context of ``conv_base_blocks`` blocks (a cached document,
  a persona, a tool manifest) shared by every dialogue of the same popularity rank;

and then it *grows*. Turn 1 is that opening plus a **user message** of
``query_blocks`` blocks. Turn 2 is all of turn 1, plus the blocks turn 1's model
output left behind, plus a new user message. Turn N+1 is turn N's whole sequence
plus turn N's output plus the new message::

    turn N+1 prompt = turn N prompt + turn N OUTPUT + new user message

That is the whole point of this module and it is what changed. It used to build
every request as ``[system][fixed conversation base][fresh query]``, so turn 1 and
turn 20 of one conversation had a byte-identical reusable prefix and differed only
in a tail nothing ever reused. Nothing grew, and nothing a model *generated* ever
appeared in a later prompt -- so the KV a decode host published under
:meth:`~kvcache_sim.control.request.Request.continuation_keys` was findable and
nobody ever looked for it. Both the payoff of prefix caching and the memory
pressure it creates were understated, and by the one term that dominates both.

A conversation is an object, and its turns are serial
-----------------------------------------------------
The unit of work is now a :class:`Conversation`, not a request: one
:class:`~realsim.runner.WorkItem` per conversation, released at the conversation's
first arrival, and the client walks its turns *one at a time*, awaiting each to
completion before submitting the next (:class:`kvcache_sim.workload._serving._Client`).
A user cannot reply before seeing the answer, so a conversation is a closed loop
and the concurrency in a run is concurrency **across** conversations. Nothing in
the harness had to change for that: :meth:`realsim.runner.Runner.run` gathers the
items, and an item is now a coroutine that lives for a whole dialogue.

That makes a turn's arrival **emergent** rather than scheduled: turn N+1 arrives
``think`` seconds after turn N finished, and when turn N finished depends on the
queueing the run produced. Only the *first* turn of a conversation has an arrival
this module can state. Every turn still carries an ``arrival`` field, holding the
instant it would arrive if the system answered instantly (the dialogue's arrival
plus the think time of every turn before it) -- a lower bound, useful for reading
the stream, and re-stamped by the client with the real one when the turn is
actually submitted. What is *not* emergent is which turns exist, what they
contain and how long each user pauses: that is fixed by the seed alone and is
identical across every configuration a scenario compares, which is what makes
"same workload, different wiring" still true.

Popularity became depth (and dialogue count)
--------------------------------------------
The Zipf draw is unchanged -- ``num_requests`` draws over ``num_conversations``
ranks, same rng calls in the same order -- but what it now decides is **how many
turns a rank contributes**, because a conversation is a first-class object and a
request is a turn of one. A few ranks account for most of the traffic, exactly as
before; the traffic is now deep conversations rather than repeated one-shots.

Those draws are cut into dialogues of at most ``max_turns`` turns. A conversation
that never ends is not a model of anything: every real dialogue is closed, and it
is the only thing that bounds a prompt -- the hottest rank here draws ~40% of the
stream, and left uncut its last turn would carry a six-figure token count and a
block chain longer than a whole instance's KV memory. So a rank's draws become
several dialogues, which is also what a heavy user is: many chats, each of them
finite. They share the rank's ``conv_base_blocks`` (same tenant, same attached
document) so the rank stays a hot *prefix* and not merely a busy one, and each has
its own growing history.

``num_requests`` is therefore still exactly the number of requests, so every
scenario offers the work it always did. What it cannot preserve is the *arrival
rate* of those requests, because turns are now paced by the system: this module
starts dialogues at ``arrival_rate`` scaled by the mean turns per dialogue it
actually produced, so the stream still delivers ``arrival_rate`` turns per second
in the limit where the system answers instantly, and less than that as it slows
down. That is a closed-loop workload's defining property, not a knob.

Determinism: a single ``random.Random(seed)`` drives conversation choice, dialogue
arrivals and think times, so the whole stream -- and every downstream metric -- is
byte-identical across runs of the same seed. Nothing here iterates a dict or a set
in a way that reaches the output; the one dict (``active``, rank -> the dialogue
currently accepting that rank's turns) is only ever looked up.

A prompt and its keys, generated together
-----------------------------------------
Each request leaves here carrying both a **prompt tensor** (``prompt_tokens``
zero-storage token ids -- see :func:`kvcache_sim.workload._accelerator.token_tensor`)
and the block-key chain that addresses it. The two are produced side by side and
that is the compromise, stated where it is made: a real chain is a hash of the
prompt's *content*, and a ``device="meta"`` prompt has none to hash. So the keys are
built from the segment ids this generator chose, which is the same sharing structure
a content hash would produce over prompts that really did share those segments, and
the prompt is the right shape with nothing in it. Hashing the prompt's shape instead
would make every same-length prompt a cache hit for every other, which is not a
weaker model but a wrong one.

The generated blocks in the middle of that chain are **not** built from a segment
id of this module's invention. They are
:meth:`~kvcache_sim.control.request.Request.continuation_keys` of the previous
turn -- literally the method the decode host calls to decide what to publish its
generated KV under. Re-deriving the same strings here from a parallel rule would
be two descriptions of one key, and the failure mode is silent in the worst way:
the workload would look up ``...|g1`` and the data plane would publish ``...|gen1``,
and the run would simply report that generated KV is never reused, which is what
it reported before this change for a completely different reason.

How many such blocks a turn contributes is the accelerator's answer for the same
reason (:meth:`~kvcache_sim.data._compute.Accelerator.blocks_for` over the
``output_tokens - 1`` positions decode generates -- the first token comes out of
prefill's last position and leaves no new block). This module builds one to ask,
rather than re-deriving the ceiling division, because it is the same object the
decode host will hand the store.

Whole blocks, everywhere, including the output
----------------------------------------------
A turn's output occupies a whole block even when it is 63 tokens long, and turn
N+1's prompt therefore grows by a whole block of "output" it did not really
generate. That is the same quantisation as everything else here -- a "user
message" is ``query_blocks`` whole blocks, a prefix match is priced in blocks, and
:meth:`~kvcache_sim.data._compute.Accelerator.generated_kv` charges the partial
trailing block whole because a paged cache really does hand out the whole physical
block. Modelling the output at token granularity in the prompt while the KV that
covers it is charged per block would make the two descriptions of one turn
disagree.

Deliberately absent
-------------------
**A varying output length.** ``output_tokens`` is fixed per request, so a
conversation's whole key chain is computable before the run starts, which is what
keeps the request stream a property of the seed rather than of the wiring. It
could vary and stay precomputable (the model has no stopping rule, so a request
produces exactly what it was asked for -- see
:attr:`kvcache_sim.report.metrics.RequestResult.output_tokens`), and the reason
not to is that ``output_tokens`` is also what control predicts decode occupancy
from and what every TBT target here is calibrated against: varying it would move
the decode columns for a reason that has nothing to do with multi-turn. Real
outputs vary; what that would buy this model is a distribution over a quantity
with no mechanism behind it.

**A retry, and an abandonment.** A turn refused at the door (overload, an SLO
gate) still counts as a turn, and the conversation continues to the next one as
though it had been served. Two alternatives were considered. Ending the
conversation at the first refusal is what a discouraged user does, but it makes
*which requests exist* depend on the selector under test, so a rejection count would
no longer be comparable between the two columns it is the whole point of. Re-deriving
the chain at run time to drop the refused turn's content would make the stream
depend on the wiring in the same way. What the simplification costs is one block
of query and one of output in turn N+1's prompt that a real transcript would not
carry -- an over-charge, never an invented reuse: the refused turn published
nothing, so the prefix run simply stops where it stopped.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from ..control.request import Request
from ._accelerator import SimulatedAccelerator, token_tensor

__all__ = ["Conversation", "make_workload", "Turn"]


@dataclass(frozen=True)
class Turn:
    """One request of a conversation, and the pause in front of it.

    ``think`` is the delay between the *previous* turn finishing and this one being
    submitted -- a user reading the answer and typing a reply. It is zero for the
    first turn of a conversation, whose arrival is the conversation's own and is
    the runner's release time.

    A pause belongs here rather than on the
    :class:`~kvcache_sim.control.request.Request` because it is not a property of
    the request: it is the gap in front of it, and
    a request that arrives is a request whose gap has already elapsed. Putting it on
    the request would also put a workload's user model into a dataclass the control
    plane routes on.
    """

    request: Request
    think: float = 0.0


@dataclass(frozen=True)
class Conversation:
    """One dialogue: a sequence of turns, run strictly one after another.

    The unit a run releases onto the clock. ``arrival`` is when the dialogue's
    *first* turn arrives -- the only arrival this module can state, since every
    later turn waits on the answer to the one before it.

    ``id`` is ``c<rank>.<n>``: the popularity rank the Zipf draw picked, and which
    of that rank's dialogues this is.

    What :attr:`kvcache_sim.control.request.Request.conversation` carries is the
    **rank**, not this id, and the difference is a routing decision.
    :func:`kvcache_sim.workload._serving._affinity` sends a conversation's requests
    to one host, and the useful unit for that is the tenant: a rank's dialogues all
    open with the same ``conv_base_blocks``, so keeping them together keeps that
    opening on one volume, and it keeps each dialogue's own growing history there
    too (a dialogue belongs to exactly one rank). Routing per dialogue instead would
    scatter a rank's shared opening across every instance and buy nothing back.
    """

    id: str
    arrival: float
    turns: Tuple[Turn, ...]

    @property
    def requests(self) -> Tuple[Request, ...]:
        """This dialogue's turns as plain requests, in order."""
        return tuple(turn.request for turn in self.turns)


def _extend(prefix: str, segments: Sequence[int]) -> Tuple[str, ...]:
    """Continue a prefix-hash chain from ``prefix`` by ``segments``.

    ``_extend("m0|3", [7, 2])`` -> ``("m0|3|7", "m0|3|7|2")``; from a model id,
    ``_extend("m0", [3, 7, 2])`` -> ``("m0|3", "m0|3|7", "m0|3|7|2")``, which is how a
    prompt's whole chain is built. The chain's whole property is that a key contains
    everything before it, so continuing one is appending to the accumulator -- which
    is also how a *later turn* is built out of an earlier one, and how
    :meth:`~kvcache_sim.control.request.Request.continuation_keys` builds the
    generated blocks in between. Sharing a leading run of segments yields identical
    leading keys, which is exactly what makes a shared prefix a single set of entries
    in the directory; rooting at the model id is what keeps two models from aliasing.

    The "hash" is modelled as the concatenation of the prompt's *segment ids* up to a
    block -- a deterministic, collision-free stand-in for a real content hash, so the
    sim never needs Python's (salted) ``hash``. Only a generator of prompts computes
    these; the planes are handed the finished chain on a
    :class:`~kvcache_sim.control.request.Request` and treat it as opaque keys.
    """
    keys: List[str] = []
    acc = prefix
    for seg in segments:
        acc = f"{acc}|{seg}"
        keys.append(acc)
    return tuple(keys)


def _zipf_weights(n: int, s: float) -> List[float]:
    """Normalized Zipf weights for ``n`` items with exponent ``s``."""
    raw = [1.0 / ((rank + 1) ** s) for rank in range(n)]
    total = sum(raw)
    return [w / total for w in raw]


class _Dialogue:
    """A conversation under construction: its keys so far, and its turns so far.

    Mutable scratch, private to :func:`make_workload`, because building turn N+1
    means reading turn N -- the chain, the generated keys it will leave behind, and
    the instant it would arrive if nothing queued. It is frozen into a
    :class:`Conversation` on the way out.
    """

    def __init__(self, id: str, rank: str, arrival: float) -> None:
        self.id = id
        self.rank = rank
        self.arrival = arrival
        #: The instant the next turn would arrive if every turn were served
        #: instantly: the dialogue's arrival plus the think time so far.
        self.planned = arrival
        self.turns: List[Turn] = []


def make_workload(
    num_requests: int,
    *,
    num_conversations: int = 8,
    system_blocks: int = 4,
    conv_base_blocks: int = 4,
    query_blocks: int = 2,
    zipf_s: float = 1.1,
    arrival_rate: float = 2.5,
    block_tokens: int = 512,
    output_tokens: int = 64,
    max_turns: int = 8,
    think_time: float = 1.0,
    model_id: str = "m0",
    seed: int = 0,
) -> List[Conversation]:
    """Generate a deterministic list of :class:`Conversation` sorted by arrival.

    - ``num_requests``: total **turns**, across every conversation. Unchanged in
      meaning: a scenario offers the same amount of work it always did.
    - ``system_blocks``: shared prefix present in every request (segments ``0..``).
    - ``conv_base_blocks``: per-rank shared context, opening every dialogue of that
      rank (distinct segment ranges per rank).
    - ``query_blocks``: the user's message on each turn (fresh, never shared).
    - ``zipf_s``: popularity skew over ranks (higher => hotter head). What it now
      decides is how many turns -- and hence how many dialogues -- a rank
      contributes.
    - ``arrival_rate``: turns per second the stream *offers*; dialogues are started
      at that rate divided by the mean turns per dialogue, so the two agree in the
      limit where service is free. See the module docstring: a closed loop cannot
      promise more.
    - ``max_turns``: the longest a dialogue runs before it ends and the rank's next
      one starts from the system prompt again.
    - ``think_time``: mean seconds a user spends reading an answer and typing the
      next message, drawn per turn from an exponential (a pause with no memory of
      how long it has already lasted). ``0`` submits the next turn the instant the
      last one lands, which is a load generator rather than a user.

    Why a mean of one second, when a human reading a paragraph takes ten. It has
    to be on the order of a turn's service time, and here that is seconds: make it
    ten times the service time and the cluster is idle nine tenths of the run, so
    every queue empties between turns and the queueing effects these scenarios
    exist to measure disappear -- not because the model got more honest but because
    the load did. This is the same compression every closed-loop benchmark applies,
    and it is a parameter so a scenario that wants to study the idle regime can ask
    for it.
    """
    rng = random.Random(seed)
    weights = _zipf_weights(num_conversations, zipf_s)
    # How many blocks of KV a turn's generation leaves behind, and therefore how
    # many keys the *next* turn's chain walks through in the middle. The
    # accelerator's answer, from the same object the decode host asks: the first
    # token comes out of prefill's last position, so only the ``output_tokens - 1``
    # positions decode produces are new KV.
    geometry = SimulatedAccelerator(block_tokens=block_tokens)
    generated_blocks = geometry.blocks_for(max(output_tokens - 1, 0))

    # Shared system prompt segments (identical across all requests -> shared keys).
    system = list(range(system_blocks))

    # Per-rank fixed context prefix, opening every dialogue that rank runs.
    fresh = 1000  # global counter for unique segment ids (namespaced above system)
    opening: List[List[int]] = []
    for _c in range(num_conversations):
        base = list(range(fresh, fresh + conv_base_blocks))
        fresh += conv_base_blocks
        opening.append(system + base)

    # Which rank each turn belongs to. Drawn first and in full, because the rate
    # dialogues start at depends on how many there will be, and that is only known
    # once every turn has been assigned. Same draws, same order, as when this
    # decided which conversation a one-shot request belonged to.
    draws = [_sample(rng, weights) for _ in range(num_requests)]
    counts = [0] * num_conversations
    for c in draws:
        counts[c] += 1
    dialogues = sum(-(-n // max_turns) for n in counts)
    # Dialogues per second such that turns per second is ``arrival_rate`` when
    # service is free. ``dialogues`` is at least 1 whenever there is any work.
    conv_rate = (
        arrival_rate * dialogues / num_requests if num_requests else arrival_rate
    )

    built: List[_Dialogue] = []
    active: Dict[int, _Dialogue] = {}  # rank -> the dialogue taking its next turn
    started = [0] * num_conversations  # how many dialogues each rank has started
    t = 0.0
    for i, c in enumerate(draws):
        dialogue = active.get(c)
        if dialogue is None or len(dialogue.turns) >= max_turns:
            t += rng.expovariate(conv_rate)
            dialogue = _Dialogue(f"c{c}.{started[c]}", f"c{c}", t)
            started[c] += 1
            active[c] = dialogue
            built.append(dialogue)

        # This turn's user message: fresh segments, shared with nothing.
        query = list(range(fresh, fresh + query_blocks))
        fresh += query_blocks
        if not dialogue.turns:
            keys = _extend(model_id, opening[c] + query)
            think = 0.0
        else:
            # ...and here is the whole of multi-turn: the previous turn's entire
            # sequence, then the blocks its output left behind (named by the
            # request itself, so these are the very keys the decode host publishes
            # under), then the new message continuing from the last of those.
            previous = dialogue.turns[-1].request
            history = previous.block_keys + previous.continuation_keys(
                generated_blocks
            )
            keys = history + _extend(history[-1], query)
            think = rng.expovariate(1.0 / think_time) if think_time > 0 else 0.0

        dialogue.planned += think
        prompt_tokens = len(keys) * block_tokens
        dialogue.turns.append(Turn(
            request=Request(
                id=f"r{i}",
                # The arrival this turn would have if the system answered
                # instantly; the client re-stamps it with the real one when it
                # actually submits (see the module docstring).
                arrival=dialogue.planned,
                block_keys=keys,
                conversation=dialogue.rank,
                prompt_tokens=prompt_tokens,
                # The prompt the client submits, one token id per prompt token.
                # Built here because a prompt is the caller's, not the cluster's --
                # and free, so the stream costs no more memory than the counts did.
                prompt=token_tensor(prompt_tokens),
                output_tokens=output_tokens,
            ),
            think=think,
        ))

    # Sorted, though the loop above already produces increasing arrivals: the
    # runner sorts by ``(release_time, id)`` anyway, and a stream that arrives
    # sorted is one less thing a reader has to take on trust.
    return sorted(
        (Conversation(d.id, d.arrival, tuple(d.turns)) for d in built),
        key=lambda c: (c.arrival, c.id),
    )


def _sample(rng: random.Random, weights: List[float]) -> int:
    """Sample an index from ``weights`` (a normalized distribution)."""
    x = rng.random()
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if x <= acc:
            return i
    return len(weights) - 1
