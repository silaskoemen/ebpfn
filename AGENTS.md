# Setup
Lockfiles must be consistent with package metadata. After any change to `pixi.toml`, run `pixi lock`.

Everything runs in a pixi environment. Any command requiring the installed python environment (like `pytest`) must be prefixed with `pixi run` (e.g. `pixi run pytest` or `pixi run python`); local editing/inspection tools do not.

Code formatting must align with standards. Run `pixi run lint` and `pixi run pytest` before `git commit`s to ensure this.

## Code Style
- Code should be secure, performant, elegant/clean, robust and maintainable/extensible.
- No overly conservative default values like `dict.get(key, <default>)` as this may hide missed wiring.
- No blanket `from __future__ import annotations` if there are no forward/self references
- No hacky workarounds to ensure backward compatibility if a meaningful code change has been implemented. If critical code hinges on compatibility, ask back for advice.
- No `git stash && ... ` to verify 'what might have passed before'. All test should pass always.
- Comments should be concisely explaining code, not verbose details of e.g. planning steps.
- Planning files may not be git tracked so comments should not reference specific sections of plans.
- Do not automatically revert other active git changes from e.g. a user; ask before.
- No overly complex code styles if a simpler implementation/flow suffices; while elegant/clean code is important, understandable and maintainable code takes priority.
- Prefer f-strings to `("{}", value)` formatting
