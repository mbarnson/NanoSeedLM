# AGENTS.md

Rules for people and AI agents who change this repository.

## Code

- The engine, the server and the packer are C99. Objective-C is only thin Metal glue. GPU code is Metal and CUDA. Do
  not use C++ or Python in these parts.
- Do not use architecture `#ifdef`s or intrinsics. CPU SIMD uses Clang `ext_vector_type`.
- Python is only for tools in `tools/`.
- Write a failing test first. Each GPU kernel has a C reference, and a test compares the kernel with it on Metal and
  on CUDA.
- Keep Metal and CUDA at parity. If a feature exists on one engine only, add it to the feature matrix in
  `docs/concepts/parity.md`.
- Each benchmark records machine state: power, load and other CPU use. Run a noisy benchmark again.

## Documentation layout

| Path | Contents | Changes |
|---|---|---|
| `README.md` | Purpose, trade-offs, how to start, models. No detailed numbers | Rarely |
| `llms.txt` | A short entry point for AI agents | When pages are added or removed |
| `docs/index.md` | A catalog of all pages, one line each | With each new page |
| `docs/log.md` | Experiments, results and releases in date order | Append only |
| `docs/concepts/` | What a method is, how it works, its current numbers | When the current state changes |
| `docs/guides/` | How to do a task | When the commands change |
| `docs/reference/` | Facts to look up: models, tools, variables, formats | When the facts change |
| `docs/results/YYYY-MM-DD-slug.md` | One dated measurement | Only the status line, after it is written |

## How to add a result

1. Write `docs/results/YYYY-MM-DD-slug.md`. Use the result page template below.
2. Add one line to `docs/index.md` under Results.
3. Add an entry to `docs/log.md`: `## [YYYY-MM-DD] result | title`, with the commit and a link to the page.
4. If the result changes the current state, update the concept page. If it replaces an earlier result, change the
   status line of the earlier page to name the new page.
5. Do not add the numbers to `README.md`. The README describes trade-offs in general terms.

## Page templates

A concept page:

1. A title, then one paragraph in plain language that a new reader can understand.
2. `## Key numbers`: a short table, if there are numbers.
3. `## How it works`.
4. `## Measurements`: links to results pages.
5. `## See also`.

A result page:

1. A title with the date.
2. `**Result.**`: the finding in two to four sentences.
3. `**Status.**`: current, or replaced by a named page.
4. `## Setup`: machine, model, reference, commands.
5. `## Results`: tables.
6. `## Notes`.

## Writing style

The documentation uses Simplified Technical English (ASD-STE100) for structure. Readers include people whose first
language is not English, and AI agents.

- Write sentences of 25 words or fewer. Write instructions of 20 words or fewer.
- Do not use semicolons. Write two sentences.
- Use the active voice. Name the actor.
- Use simple tenses. Do not write "has been" or "have made".
- Use a verb for an action, not a noun ("analyze", not "perform an analysis").
- Do not use phrasal verbs. Write "start", not "spin up". Write "read", not "dive into".
- Use one term for one thing. Use the same term each time.
- Do not use marketing words (powerful, seamless, blazing-fast). Give the measurement.
- Do not use the words "honest" or "honestly".
- Keep the confidence of a claim. "May" stays "may".
- State costs and failures as clearly as gains.
- Use a list for three or more steps or conditions.

To check a file, run an STE linter, for example `ste-lint.py` from
[asd-ste100-skill](https://github.com/danyuchn/asd-ste100-skill). Tables and code blocks are exempt from the
sentence limits.
