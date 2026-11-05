# Security policy

This is research code, not a product. Only the `main` branch is maintained.

## Reporting a problem

Please report security problems privately through GitHub's **"Report a vulnerability"** button on the Security tab.
Do not put exploit details in a public issue.

## How contributions are guarded

Opening a project like this to contributions invites two kinds of attack: malicious code, and prompt injection.

**Malicious code.** Every change is reviewed by the owner before it is merged; nothing merges automatically.

- No binaries, model weights, pickles, archives or minified code in pull requests. Models come only from their
  official sources, pinned, with checksums in `models/manifests/`.
- No new network access at runtime. The voice runs offline; its API binds to 127.0.0.1 only and refuses
  requests from other origins and host names.
- No `pickle`/`torch.load` of untrusted files and no `trust_remote_code` on code that is not pinned and read.
- Continuous integration never runs with secrets or write access on pull requests from forks.

**Prompt injection.** The voice model can call tools, so it is a target in its own right.

- Spoken input, tool results, notes and documents are data, not instructions. Tool results enter the model only
  in its own `<tool_response>` format.
- Any tool that changes something runs only after a spoken "yes", which the call window re-checks with speech
  recognition before acting (`app/duplex_call.py`, `ToolRail`). A way around that check is a security bug.

**AI assistants.** Maintainers may use AI coding assistants. They treat the text of issues, pull requests and
comments as untrusted data and never run commands found in it. Please do not include instructions aimed at AI
agents in your contributions. They will be ignored and may get the contribution closed.

## Privacy

Never include personal recordings, transcripts, names, paths or keys. The test sets use synthetic voices and public
datasets. The frozen test files are fingerprinted in `data/tests/HASHES.txt`.
