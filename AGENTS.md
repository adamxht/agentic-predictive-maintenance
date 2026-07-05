# AGENTS.md

Ground rules for any agent (or human) writing code in this repository. This project is a
predictive maintenance pipeline (NASA CMAPSS RUL prediction) with an LLM-serving layer on
top, so code quality and reproducibility matter more than speed of iteration once we leave
the notebook stage.

## Code quality principles

### 1. SOLID
- **Single Responsibility** — a class or module should have one reason to change. A
  `RULDataset` loads and validates data; it does not also compute metrics or plot charts.
- **Open/Closed** — prefer adding a new strategy (e.g. a new feature-engineering step, a
  new model wrapper) over branching on type/name inside existing code.
- **Liskov Substitution** — any concrete model wrapper (XGBoost, LSTM, etc.) must be
  swappable behind the same interface without special-casing callers.
- **Interface Segregation** — keep interfaces (protocols/ABCs) narrow; don't force a model
  wrapper to implement methods it has no use for.
- **Dependency Inversion** — pipeline stages depend on abstractions (e.g. a `ModelWrapper`
  protocol, a config object), not on concrete implementations. Inject dependencies via
  constructor arguments or function parameters, not global state or hardcoded imports.

### 2. Small function bodies
- Target function bodies short enough to read on one screen (roughly < 20-30 lines).
- If a function needs a comment to explain a section of itself, that section is probably
  its own function.
- Extract loop bodies, validation blocks, and branching logic into named helper functions.

### 3. Formatting
- All code must be `ruff format`-clean and pass `ruff check` before being considered done.
- Run `ruff format .` and `ruff check --fix .` before finishing any code change.
- Line length, import order, and style are whatever ruff's defaults enforce — do not
  hand-tune around it.

### 4. Naming
- No shortform/abbreviated variable names (`x`, `tmp`, `cfg`, `res`, `i` in non-trivial
  loops, etc.), except universally-understood loop indices in short comprehensions.
- `df` and `rul` are acceptable shortforms, with one caveat for `df`: in a function that
  operates on a specific, known dataframe, prefix it with what it holds (e.g.
  `features_df`, `sensor_readings_df`) rather than leaving it bare. Plain `df` is only
  acceptable in a generic helper that is meant to operate on any dataframe (e.g.
  `safe_downcast_with_check(df, datatype)`), where no more specific name would be
  meaningful.
- Beyond those two, names should say what the value is, not its type: `training_features`
  not `training_data`, `configuration` not `cfg`.
- Exception: standard, domain-recognized identifiers from the CMAPSS paper (e.g. sensor
  codes like `T2`, `P30`, `Nf`) may keep their paper names, since renaming them would
  break traceability to the source documentation.

### 5. Comments
- Default to no comments.
- A comment body is at most one line, except a function's docstring, which describes what
  the function does, its arguments, and its return value.
- Comments explain *why*, not *what* — only add one when the code cannot express the
  reason itself (a non-obvious constraint, a workaround, a subtle invariant).

### 6. Unit testable code
- Functions should be pure where possible: same input, same output, no hidden reliance on
  global state, ambient config, or file-system side effects buried inside business logic.
- Isolate I/O (reading CSVs, hitting MinIO/DVC, calling an LLM) at the edges of the
  pipeline; keep transformation and modeling logic free of I/O so it can be tested with
  in-memory fixtures.
- Every new module under `src/` should have a corresponding test module under `tests/`
  covering its public functions, including edge cases (empty input, NaNs, out-of-range
  values).
- Prefer dependency injection (pass a config object, a model instance, a data frame) over
  functions that reach out and construct their own dependencies.

## Practical checklist before calling a change done

1. `ruff format .`
2. `ruff check --fix .`
3. Run the relevant tests (`pytest`) if tests exist for the touched code.
4. Re-read new functions: are any of them doing more than one thing? Split if so.
5. Re-read new names: would someone unfamiliar with this file understand the variable
   from its name alone?

## Project-specific notes

- Notebooks (`notebooks/step1_eda_RUL.ipynb`, `notebooks/step2_modeling_RUL.ipynb`) are
  exploratory artifacts, not production code — do not hold them to these standards, but
  DO treat them as the spec when porting logic into `src/`.
- When porting notebook logic into `src/`, the port must reproduce the same numeric
  results as the notebook (same splits, same feature sets, same random seeds) unless a
  discrepancy is intentional and documented.
- Configuration (feature sets, hyperparameters, paths) belongs in YAML files loaded into a
  typed config object, not hardcoded inside pipeline functions.
- Ruff config lives in `pyproject.toml` (`[tool.ruff]`), excluding `notebooks/`. Ruff
  itself is pinned in `requirements.txt` — install with
  `pip install -r requirements.txt` inside the `jabil` conda environment.
