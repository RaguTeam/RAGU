---

name: ragu-build
description: Interview the user about their RAGU use case, select an appropriate RAGU pipeline, and generate a validated ragu_build.yaml plus a runnable build_<name>.py script.

---

# Assembling a RAGU build

Goal: determine what the user actually needs through a short interview, then produce two artifacts:

* `ragu_build.yaml` — the recorded decisions and rationale;
* `build_<name>.py` — a working RAGU build script.

The RAGU library is already documented: every module ships a `README.md` describing its role in the pipeline and providing examples.

This skill is a **decision process and navigation map**, not a retelling of that documentation.

**Do not read module sources unless explicitly allowed below. Do not summarize whole READMEs.**
Read narrowly, guided by `references/module-map.md`.

Conduct the interview in whatever language the user writes in.

## Skill resources

All paths under:

* `references/`
* `assets/`
* `scripts/`

are relative to **this skill's own directory**, not to the user's project root.

Resolve the skill directory using whatever mechanism the current agent environment provides. Never assume that the skill lives inside the user's repository or that its resources are reachable through paths relative to the current working directory.

---

# Phase 0. Inventory

Before asking the user anything, infer everything that can reasonably be discovered from the project.

This removes unnecessary interview questions.

## 0.1 Read the module map

Read:

`references/module-map.md`

Use it only as a navigation map for RAGU components and documentation.

## 0.2 Inspect the project

Inspect relevant project state in one batch where practical.

### Corpus

If the user named a data directory, inspect:

* file extensions;
* file count where useful;
* total size.

For example:

```bash
find <dir> -type f | sed 's/.*\.//' | sort | uniq -c
du -sh <dir>
```

This establishes:

* approximate corpus size;
* which file types are present;
* how much of the corpus RAGU can actually ingest.

Do not recursively inspect document contents unless necessary.

### Existing infrastructure

Look for already-running or already-configured storage infrastructure.

Useful signals include:

```bash
docker ps
```

and project-local evidence such as:

* `qdrant_storage/`;
* Qdrant configuration;
* Neo4j configuration;
* existing storage-related compose services.

Use only obvious project configuration. Do not search through unrelated files.

### Local model capability

If available, run:

```bash
nvidia-smi
```

Use the result only to determine whether local GPU-backed models are realistic.

Failure or absence of `nvidia-smi` is not itself an error.

## 0.3 Credentials boundary

Do **not** search for:

* API keys;
* credentials;
* `.env` contents;
* secrets.

Which model provider or runtime the user intends to use is an interview question, not something to discover from private credentials.

## 0.4 State findings

Before starting the interview, tell the user in one short statement what was discovered and what will be treated as given.

For example:

> Your data directory contains about 900 text files totaling ~40 MB. No running vector or graph storage is visible, and a local NVIDIA GPU is available. I'll plan around those facts unless you want to override them.

Anything already established in Phase 0 must **not** be asked again.

Present inferred facts as correctable facts rather than questions.

---

# Phase 1. Interview

The purpose of the interview is to eliminate incompatible pipeline branches with as few questions as possible.

## Interview mechanism

Use the current environment's interactive question mechanism when one is available and appropriate.

Otherwise ask directly in chat.

Do **not** depend on any specific platform tool name such as `AskUserQuestion`.

Ask **one or two questions at a time**.

Never dump the whole interview into a single form or message.

## Interview rules

* Phrase questions using **example queries and example data**, not RAGU implementation terminology.
* Avoid terms such as:

  * "extractive";
  * "abstractive";
  * "hybrid retrieval";
  * "community detection";
  * internal RAGU class names.

Technical terminology may appear in the explanation after the user answers, but not unnecessarily in the question itself.

Q1 is the deliberate exception: whether to build a graph is asked explicitly because this decision dominates both cost and architecture, and many users already know whether they want one.

If they do not know, use Q1b.

After each answer, briefly explain which pipeline branches were eliminated.

Order questions by branching factor.

Use at most **six primary questions**.

If a question's answer follows from:

* Phase 0;
* an earlier interview answer;
* something the user already stated;

skip it.

## Missing answers

Never let an unanswered question stall the deliverable.

If the user:

* skips a question;
* gives an unusable answer;
* selects an equivalent of "I'll specify later" but does not specify it;

ask once more.

If the answer is still unavailable:

1. choose the most defensible default;
2. clearly state that it is an **assumption**, not a finding;
3. continue.

Every such assumption must appear in all relevant outputs.

In the Phase 2 decision table, write:

`ASSUMPTION`

in the **based on what** column.

In `ragu_build.yaml`, mark it with:

```yaml
# ASSUMPTION
```

In `build_<name>.py`, keep the assumed value as a named constant near the top of the script.

Do not bury assumed values deep inside the implementation.

In the final report, list unresolved assumptions as things the user should verify before the first real run.

An unanswered question should cost one line in the final report.

It must never cost the whole deliverable.

---

# Interview decision order

Exact wording and options live in:

`references/decision-matrix.md`

section:

`Questions`

Use that wording when available.

The decision sequence is:

| #  | About                                                                   | Eliminates                 |
| -- | ----------------------------------------------------------------------- | -------------------------- |
| 1  | whether to build a graph, including its cost                            | graph / flat index         |
| 1b | how answers are distributed across documents — only if Q1 is "not sure" | graph / flat index         |
| 2  | examples of typical queries                                             | search engine              |
| 3  | exact terms, codes, IDs or part numbers in queries                      | lexical / sparse retrieval |
| 4  | where models run                                                        | LLM / embedder client      |
| 5  | corpus size and update pattern                                          | storage backends           |
| 6  | extraction quality versus cost — graph builds only                      | artifact extractor         |

## Q1 branching

If Q1 selects a vector-only / flat-index build:

* do not ask Q6;
* skip graph-specific decisions that no longer matter.

Q1b runs **only** when Q1 is effectively:

`not sure`

Never ask both Q1 and Q1b as independent decisions.

---

# Phase 2. Decision summary

Before generating any files, show the user a concise decision table with:

| choice | why | based on what |
| ------ | --- | ------------- |

The **based on what** column must identify the source of each decision:

* a specific user answer;
* a Phase 0 finding;
* `ASSUMPTION`.

Also explicitly state important components that are **not included** and why.

For example:

> No Global engine — none of the example queries asked for corpus-wide themes or summaries. It can be added later without changing the ingestion strategy.

Do not list every conceivable unused RAGU component. Mention exclusions only when they represent meaningful architectural branches.

Wait for user confirmation before Phase 3.

If the user changes one decision:

* revise that decision;
* revise decisions that depend on it;
* do not restart the whole interview unless the change genuinely invalidates all previous answers.

---

# Phase 3. Generation

After the user confirms the decision summary, generate the build.

## 3.1 Read the decision matrix

Read:

`references/decision-matrix.md`

in full.

It is the source of truth for:

* exact imports;
* RAGU class names;
* constructor signatures;
* stock build structure;
* component substitutions.

Take every class name and constructor parameter from the matrix rather than from memory.

## 3.2 Resolve missing component details

If the matrix does not contain enough detail for a selected component:

1. use `references/module-map.md` to locate the corresponding RAGU module;
2. read only the relevant section of that module's `README.md`.

Do not read the whole README unless the required information cannot otherwise be located.

Do not inspect source code merely for convenience.

### Last-resort signature lookup

If a required constructor detail exists in neither:

* the decision matrix;
* the relevant README;

then inspect only the real constructor signature.

For example:

```bash
grep -n "def __init__" -A 25 <file>
```

Use this only as a last resort.

Do not explore implementation internals.

Never invent constructor parameters.

---

# 3.3 Generate `ragu_build.yaml`

Write:

`ragu_build.yaml`

It must record the selected build decisions.

For every meaningful decision include:

* selected option;
* one-line rationale;
* enough information to reconstruct why the pipeline was chosen.

Keep rejected architectural alternatives out of the Python script; they belong here when worth recording.

Mark unresolved defaults explicitly:

```yaml
# ASSUMPTION
```

---

# 3.4 Generate `build_<name>.py`

Start from:

`assets/build_template.py`

The template represents stock build B:

graph + local search.

Adapt it to the decisions selected during the interview.

Use:

* Part 2 of `references/decision-matrix.md` for exact signatures;
* Part 3 for the shape of other stock builds.

A comparable hand-written example is:

`examples/extract_with_llm_and_local_search.py`

when it exists in the user's RAGU checkout.

## Script style

Keep the generated script intentionally flat.

Preferred structure:

1. imports;
2. named configuration constants;
3. `async def main(...)`;
4. top-to-bottom construction and execution;
5. `if __name__ == "__main__"` guard.

Do not create helper functions that are called only once unless they materially improve correctness.

Do not include:

* commented-out alternatives;
* unused imports;
* dead code;
* speculative components;
* rejected pipeline choices.

Keep choices that were considered but not selected in `ragu_build.yaml`.

## Assumptions

Any unresolved assumption must remain visible as a named constant near the top.

For example:

```python
# ASSUMPTION: user did not specify the collection name.
COLLECTION_NAME = "ragu"
```

## Indexing safety

The validator runs `main()` for real.

Therefore:

* construct components inside `main()`;
* keep expensive corpus indexing behind an explicit flag;
* do not index the corpus unconditionally at import time;
* do not initiate expensive work merely by importing the generated module.

The generated script is for the user to run intentionally.

---

# 3.5 Validate the build

The validator belongs to this skill at:

`scripts/validate_build.py`

Resolve this path relative to the skill's own directory.

Never assume it is relative to the user's project.

## Choose Python

Run validation using a Python interpreter capable of importing the user's `ragu` installation.

Prefer, in order:

1. an active `$VIRTUAL_ENV`;
2. project `.venv/bin/python`;
3. project `venv/bin/python`;
4. `python3`.

Confirm that the selected interpreter can import RAGU before relying on the validator.

A typical check is:

```bash
if [ -n "$VIRTUAL_ENV" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PY="$VIRTUAL_ENV/bin/python"
elif [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
elif [ -x "venv/bin/python" ]; then
    PY="venv/bin/python"
else
    PY="python3"
fi

"$PY" -c "import ragu"
```

Then run:

```bash
"$PY" <skill-dir>/scripts/validate_build.py build_<name>.py
```

Use the actual resolved skill directory in place of `<skill-dir>`.

## Validation failure

If no available interpreter can import `ragu`:

* say so clearly;
* do not pretend validation succeeded;
* do not claim the generated build is checked.

Stop trying to execute the validator.

The files may still be generated, but the final report must identify validation as blocked.

## Validator behavior

The validator:

* checks RAGU calls against real signatures;
* runs the generated script's `main()`;
* replaces external/network/model calls with local stand-ins.

Validation must not intentionally send requests to external model providers or execute real model inference.

Fix every validator error caused by the generated script before reporting success.

Do not suppress or ignore validator failures merely to finish the task.

---

# Phase 4. Final report

Report the result concisely.

Open with anything that blocks the first real run, including:

* unresolved assumptions;
* placeholder values;
* missing Python environment;
* services that need to be running;
* model/provider configuration still required.

Then state:

* where `ragu_build.yaml` was written;
* where `build_<name>.py` was written;
* whether validation passed;
* how to run the generated script;
* approximate first-build cost;
* approximate first-build duration;
* the first things to adjust if answer quality is poor.

Cost and duration estimates must be presented as estimates, with the assumptions behind them.

Do not imply precision that the available corpus size, model provider, hardware, or extraction strategy does not support.

---

# Boundaries

## Input formats

**RAGU ingests plain text and nothing else.**

RAGU does not itself provide:

* PDF parsing;
* office-document parsing;
* OCR;
* ASR;
* image understanding;
* video transcription.

If the user's corpus contains:

* PDFs;
* Word documents;
* presentations;
* images;
* audio;
* video;
* other non-text formats;

say plainly that RAGU cannot ingest those files directly.

Converting them to text is a separate preprocessing step outside the build produced by this skill.

Never generate a build that pretends unsupported files can be read directly.

## Library modifications

This skill produces:

* configuration;
* a runnable script.

It does **not** modify the RAGU library itself.

Do not patch RAGU source code as part of this workflow.

## Expensive execution

Do not:

* build the real graph;
* index the full corpus;
* call paid models;
* burn tokens;
* start large extraction jobs;

unless the user explicitly asks for execution.

Generating and locally validating the script is allowed.

The resulting script is the user's build to run.

## Source inspection

Prefer information sources in this order:

1. `references/decision-matrix.md`;
2. `references/module-map.md`;
3. relevant narrow sections of module `README.md`;
4. exact constructor signature inspection as a last resort.

Do not browse RAGU implementation source for architecture understanding.

Never invent APIs, class names, arguments, or constructor parameters.
