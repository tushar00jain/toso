# toso — project conventions

## Run knobs: ambient config, never threaded arguments
Run-wide options (fidelity models, debug toggles, resource limits) live in
`sim_common/config.py` (`SimConfig`) and are read ambiently.

- **NEVER** add a knob as a parameter on intermediate functions. Threading a scalar
  through every layer is churn across many files; ambient config is one edit.
- Copy an existing flag: `trace`, `fingerprint`, `real_directory`, `contention`,
  `collapse_charges`.

To add one:
1. Add a field to `SimConfig`, and parse its `TOSO_*` env var in `_from_env`.
2. Read it at the leaf: `config.current().<flag>`. For an object built once per run,
   use a `from_config()` factory that reads the flag when its explicit arg is `None`.
3. Wire the CLI once at startup: `config.configure(<flag>=args.<flag>)`. `configure`
   ignores `None`, so an unset flag defers to env/default.
4. In tests use `config.overrides(<flag>=...)` (scoped, auto-restored) — **not** an
   argument.

Two exceptions:
- Shared **objects** (e.g. a per-run `ResourceRegistry`) are still built once and
  injected. Inject the object; read its settings from config.
- A leaf may keep an optional override param defaulting to `None` (→ ambient) so it
  is unit-testable alone. Do not thread it further.

## Determinism
- A flag must be debug/output-only and must not move a measured metric. That is what
  makes an ambient read safe under `realsim/tools/check_contract.py`.
- A flag that changes simulated timing (e.g. `contention`) is a documented exception:
  say so in the field comment, hold it fixed for the whole run, keep ordering
  seq-tie-broken.

## Opt-in, default-off
New fidelity/perf features default to historical behavior (`contention="none"`,
`collapse_charges=False`, `real_directory=True`, `trace=True`), so the default path
stays byte-identical until a run opts in.

## Verification
- A one-off question ("does this attribute exist") is an inline script.
- A measurement you will repeat is a **saved script**, uniquely named. Re-deriving it
  each time is how it drifts.
- A measurement of a repo invariant goes in `realsim/tools/`, run by `python -m`,
  printing a stable diffable report, knowing nothing about which checkout it is in.
  Comparing two trees is then `diff` of two runs.
- A measurement that must **always** hold is a test, not a tool.
- Print the assertion, not the evidence. A metrics dump answering "did the
  fingerprint move" costs more to read than the answer is worth.

## Comments and docstrings
Prose exists to make the code faster to read. Prose that does not is deleted, not
shortened. Check with `python -m realsim.tools.prose_budget`.

**Write:**
- what the thing is and how to use it;
- why it is **correct** when that is not obvious — an invariant, a tie-break, an
  ordering constraint, a reason a race cannot happen;
- what is **missing** — a gap, a simplification, a limit to trust less;
- which case a branch is: 3–8 words, at the line.

**NEVER write:**
- **history** — "used to", "no longer", "previously", "this replaced", "it was N, now
  M". Git holds the past.
- **narration** — walking the code in the order it runs. Delete it; the code is right
  there.
- **argument for the code** — no defending a signature, no praising the design, no
  "which is the point". A rejected alternative gets one clause with its cost, or a
  `TODO:`.
- **machinery** — how `attach`, a `View`, a `Dispatcher`, a base class or a
  `proposed/` helper works. Document it once, where it is implemented.
- **glue** — "which is what/why…", "and that is the whole of it", one idea restated
  three ways.
- restating what the next line plainly does.

### Where prose goes
Python has no header/impl split, so apply the budget **per unit**, not per file.

- **Declarations** may carry real prose: module docstring, abstract/Protocol members,
  dataclass fields, a constructor's `Args:`.
- **Concrete bodies** stay near-bare: branch labels, plus invariants that cannot be
  asserted.
- A **concrete** docstring must not outrun the code it heads. Abstract members are
  exempt.
- Put a comment **at the field**. This is the one place prose here is underweight.

### How to write it
- One idea, one sentence, flattest phrasing. A comment is read at a skim, once.
- No em-dash appositive chains, no inverted syntax.
- Link where a reader would go hunting. A `:class:` on every noun is overhead on
  every sentence.
- Prefer a check to a sentence: if an assert, a type, a validator or a raised
  `ValueError` can state it, state it there and write no comment.
- Quantify or cut. "73% of reported handoff bytes" earns its line; "which is what
  settles a placement" does not.
- Say what is true of **this** use, not of the mechanism: a threshold's units, why
  this argument and not the obvious other, an ordering this composition depends on.
- A section header is the reader's question ("how does a read find the right
  block?"), not a conclusion.

### Length
Design needed to read **this** module stays in this module, at whatever length that
takes. Only cross-module system design goes in a document.

- Length is earned by teaching a model the code cannot show: a picture the types do
  not draw, an invariant graph, the states a field ranges over, what happens when
  inputs contradict each other.
- It is not earned by narration.
- Test: would reading the code top to bottom say the same thing? Then delete it.
