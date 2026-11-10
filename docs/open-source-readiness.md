# Open-source readiness

The repository remains private. This review does not authorize a visibility change or package publication.

## Review on 2026-11-10

The maintainer reported no known history of secrets or confidential data. Gitleaks scanned all locally available Git refs with redacted output and found no secrets. This is a detector result, not a guarantee that all confidential material is absent.

GitHub metadata showed no Actions runs, releases, or issues, and the wiki was disabled. Twelve pull requests exist. Their descriptions, comments, and attachments still need a confidentiality review before publication. Remote-only refs and external caches were not audited.

The readiness script missed the Python tests because they are under `packages/python/tests`. The repository also has an Astro documentation site. These are present despite the script's generic indicators.

A clean export of the tracked source installed in a new Python 3.12 virtual environment. The installed CLI, offline example, and golden checkpoint identity check passed. Unit discovery ran 209 tests with no failures and 51 skips. Optional model workflows, native sandbox execution, and the documentation build were not verified in this review.

## Before publication

- Select a license, add its full text, and align the README and package metadata. No license has been selected by this review.
- Review pull request history and attachments for confidential material.
- Add a security policy with a verified reporting channel. Do not invent a contact address or assume private vulnerability reporting is enabled.
- Decide whether to add standalone contribution guidance and a code of conduct. The README includes basic contribution and verification instructions.
- Restore CI only when intended. Actions is disabled, and the saved workflow has no pull-request trigger. Review action pins, token permissions, and supported runtimes before enabling it.
- Review branch rules, required checks, secret scanning, and push protection in GitHub settings. These settings were not changed or fully audited.
- Consider an `.editorconfig` and automated dependency updates. Preserve the existing package managers and testing tools.
- Verify a fresh-clone install, model workflows, documentation build, and supported runtimes. Keep optional model and host-dependent checks distinct from the offline example.
- Choose release ownership, versioning, and trusted publishing before distributing packages. This review creates no tags, releases, or package uploads.

Run the readiness checklist again after addressing these items. Keep any remaining gaps explicit before changing visibility.
