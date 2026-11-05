# Contributing

Thanks for looking. The project is paused but open to feedback and help. The [review paper](paper/REVIEW.md) says
where it stands and what is next.

## Most wanted

- **Turn-taking:** ideas or results on why a fine-tuned audio model answers real voices only with "yeah", and how
  to fix it on small hardware.
- **Data:** free, properly licensed two-person conversation audio, and ways to write good replies into real timing.
- **Decoding:** the stop rule, plan chunking, and anything else in `app/duplex.py`.
- **Evaluation:** reviews of the harness and scorers, and runs on other GPUs.
- **The paper:** corrections, missing related work, unclear parts.

## How

1. For anything bigger than a small fix, open an issue first and say what you want to change and why.
2. Keep pull requests small and focused, with evidence: numbers from a saved script, not impressions.
3. Fill in the pull request checklist. The owner reviews every change; nothing merges automatically.
4. An automatic audit checks every pull request (`tools/audit_contribution.py`): it fails on binaries, weights,
   files over 1 MB, hidden characters and unlogged changes to the frozen tests, and flags risky code for the
   reviewer. It never runs your code.

## Rules that keep results honest

- **Frozen tests and scorers are never changed to make something pass.** Any change to `data/tests/` or a
  scorer needs a reason, logged in `docs/lab-notebook.md`.
- **Decisions use validation data only,** never the frozen tests.
- **Data must be free and licensed,** with its source and license named. No personal data, and no outputs of
  proprietary AI assistants as training data.
- **No binaries, weights, pickles or archives in git** (see [`SECURITY.md`](SECURITY.md)). Models are fetched from
  official sources and checked against `models/manifests/`.

By contributing you agree to license your contribution under the repository's MIT license, and to follow the
[code of conduct](CODE_OF_CONDUCT.md).
