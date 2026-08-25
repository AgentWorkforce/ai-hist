# Changelog

Notable changes to the native `ai-hist` CLI are documented here.

## [Unreleased]

### Breaking

- Remove the legacy Python CLI and the public `ai-hist-python` and
  `ai-hist-rust` compatibility launchers. `AI_HIST_CLI` is no longer supported;
  the source-checkout launcher exits with an explanatory error when it is set.
- Make installation Rust-only. Upgrades remove recognized installer-managed
  legacy launchers and report both removals and unrecognized files left intact.

### Changed

- Replace Python-based installer and end-to-end verification with shell,
  SQLite, Node.js, and the public Rust CLI interfaces.
