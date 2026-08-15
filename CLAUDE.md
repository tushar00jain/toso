# toso — project conventions

## Cross-cutting run knobs: use ambient config, don't thread arguments
New run-wide options (fidelity models, debug/output toggles, resource limits) go
through `sim_common/config.py` (`SimConfig`), read ambiently — NOT as scalar
parameters threaded down scenario → cluster/coordinator → transport call chains.
Threading a scalar through every layer is churn across many files; ambient config
is one edit.

To add a flag:
1. Add a field to `SimConfig` (+ parse its `TOSO_*` env var in `_from_env`).
2. Read it at the leaf that needs it via `config.current().<flag>` (or, for a
   shared object built once per run, a `from_config()` factory that reads the
   flag when its explicit arg is `None`).
3. Wire the CLI once at startup: `config.configure(<flag>=args.<flag>)`
   (`configure` ignores `None`, so an unset flag defers to env/default).
4. In tests, set it with the scoped `config.overrides(<flag>=...)` context
   manager (auto-restored) — not by passing an argument.

Do NOT add the flag as a parameter to intermediate functions. Flags to copy:
`trace`, `fingerprint`, `real_directory`, `contention`, `collapse_charges`.

Nuances:
- Shared OBJECTS (e.g. a per-run `ResourceRegistry`) are still created once and
  injected — inject the object, but let its mode/settings come from config, not a
  threaded scalar.
- A leaf may keep an optional explicit override param defaulting to `None`
  (→ ambient) so it can be unit-tested in isolation; just don't thread it.

## Determinism
Most flags must be debug/output-only and never change a measured metric — that
keeps an ambient read safe under the contract enforced by
`realsim/tools/check_contract.py`. A flag that changes simulated timing (e.g.
`contention`) is a deliberate, documented exception: call it out in the field
comment, keep it fixed for the whole run, and keep any ordering seq-tie-broken.

## Opt-in, default-off
New fidelity/perf features default to the historical behavior (`contention="none"`,
`collapse_charges=False`, `real_directory=True`, `trace=True`) so the default path
stays byte-identical and nothing changes unless a run opts in.

## Verification: a measurement you will make twice is a file, not a heredoc
A one-off question ("does this attribute exist") is an inline script. A measurement
you will repeat — a metric sweep, a parity check between two trees — is a saved
script, uniquely named and overwritten deliberately; re-deriving it each time is how
it drifts, and a stale one silently answers the wrong question.

A measurement that checks a repo invariant belongs in `realsim/tools/`, run by
`python -m`, printing a stable diffable report and knowing nothing about which
checkout it runs in — comparing two trees is then `diff` of two runs. A measurement
that must always hold is a test, not a tool.

Print the assertion, not the evidence: a full metrics dump answering "did the
fingerprint move" costs more to read than the answer is worth.

## Prose: the docs serve the code, not the other way round
Comments and docstrings exist to make the code faster to read. Prose that does not
do that is deleted, not shortened.

Write:
- what the thing is and how to use it, in as few words as it takes;
- why it is **correct** when that is not obvious (an invariant, a tie-break, an
  ordering constraint, a reason a race cannot happen);
- what is **missing** — remaining work, a known gap, a limit worth trusting less.

Do not write:
- **history.** No "used to", "no longer", "previously", "this replaced", "a claim
  withdrawn", "it was N, now it is M". A comment describes the code beside it, not
  the code it replaced. Git holds the past.
- **justification of a decision already visible in the code.** If the signature says
  it, the docstring does not need a paragraph arguing for it.
- **essays.** A class docstring that runs longer than the class is a design doc in
  the wrong file; put it in `docs/` and link, or cut it.
- restating what the next line plainly does.

Rules of thumb: a module's prose should not exceed roughly a third of its lines; a
docstring longer than ~15 lines needs a reason; one idea is stated **once**, in the
one place it belongs, and cross-referenced from anywhere else that needs it.
