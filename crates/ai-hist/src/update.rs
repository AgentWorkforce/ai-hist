//! The "new version available" notice printed by `ai-hist --version`.
//!
//! Release binaries are attached to GitHub releases tagged `sdk-ts-v<semver>`
//! (see `.github/workflows/publish.yml`). The latest release is discovered
//! without the GitHub API: `GET /releases/latest` answers with a redirect
//! whose `Location` header names the release tag, which avoids API rate
//! limits and JSON parsing entirely.
//!
//! The check is best-effort by design: it only runs when stderr is a
//! terminal, it is bounded by a short timeout, and any failure — offline,
//! proxy, unexpected redirect target — silently skips the notice. `--version`
//! must always print the version and exit 0.

use std::io::IsTerminal;
use std::time::Duration;

/// Repository whose GitHub releases carry the CLI binaries.
const REPO_SLUG: &str = "AgentWorkforce/relayhistory";
/// Prefix of release tags, e.g. `sdk-ts-v0.6.0`.
const TAG_PREFIX: &str = "sdk-ts-v";
/// Whole-request budget for the release lookup; `--version` must never hang.
const CHECK_TIMEOUT: Duration = Duration::from_secs(3);
/// Environment opt-out honored in addition to the `--no-warning` flag, for
/// scripts that cannot change the invocation.
const OPT_OUT_ENV: &str = "AI_HIST_NO_UPDATE_CHECK";

/// A release `major.minor.patch` triple, ordered numerically so `0.10.0`
/// beats `0.9.1`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct ReleaseTriple(u64, u64, u64);

impl std::fmt::Display for ReleaseTriple {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.0, self.1, self.2)
    }
}

/// Parse `major.minor.patch` from the front of a version string, tolerating a
/// `-prerelease` or `+build` suffix on the running version (`0.7.0-next.1`
/// compares as `0.7.0`).
fn parse_triple(version: &str) -> Option<ReleaseTriple> {
    let core = version
        .split_once(['-', '+'])
        .map_or(version, |(core, _)| core);
    let mut parts = core.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some(ReleaseTriple(major, minor, patch))
}

/// Extract the release triple from a `Location` redirect target such as
/// `https://github.com/…/releases/tag/sdk-ts-v0.6.0`, or from a bare tag.
/// Only stable `sdk-ts-vX.Y.Z` tags parse; anything else (a prerelease tag,
/// the no-releases redirect back to `/releases`) yields `None`.
fn parse_release_tag(location: &str) -> Option<ReleaseTriple> {
    let tag = location.rsplit('/').next()?;
    let version = tag.strip_prefix(TAG_PREFIX)?;
    // Reject prerelease/build suffixes outright rather than rounding them
    // down: a `latest` release should always be a stable triple.
    if version.contains(['-', '+']) {
        return None;
    }
    parse_triple(version)
}

/// Whether the notice is suppressed by `--no-warning` or the opt-out
/// environment variable. Argument scanning is deliberate: clap never yields
/// parsed matches on the `--version` path, it short-circuits with
/// `ErrorKind::DisplayVersion` instead.
fn suppressed(mut args: impl Iterator<Item = String>, opt_out_env: Option<&str>) -> bool {
    if args.any(|arg| arg == "--no-warning") {
        return true;
    }
    matches!(opt_out_env, Some(value) if !value.is_empty() && value != "0")
}

/// Latest stable release version, from the `/releases/latest` redirect.
fn latest_release_triple() -> Option<ReleaseTriple> {
    let agent = ureq::builder()
        .redirects(0)
        .timeout(CHECK_TIMEOUT)
        // Best-effort check: honor HTTPS_PROXY et al. so proxied machines
        // still see the notice.
        .try_proxy_from_env(true)
        .build();
    let url = format!("https://github.com/{REPO_SLUG}/releases/latest");
    let response = agent.get(&url).call().ok()?;
    parse_release_tag(response.header("location")?)
}

/// Print a stderr notice when a newer release exists. Never errors and never
/// prints on failure: a `--version` that cannot reach GitHub stays silent.
pub(crate) fn maybe_print_update_notice() {
    if !std::io::stderr().is_terminal() {
        return;
    }
    if suppressed(std::env::args(), std::env::var(OPT_OUT_ENV).ok().as_deref()) {
        return;
    }
    let Some(current) = parse_triple(crate::CLI_VERSION) else {
        return;
    };
    let Some(latest) = latest_release_triple() else {
        return;
    };
    if latest > current {
        eprintln!(
            "\nA new version of ai-hist is available: {current} -> {latest}\n\
             Update with:\n  \
             curl -fsSL https://raw.githubusercontent.com/{REPO_SLUG}/main/install.sh | sh\n\
             (pass --no-warning or set {OPT_OUT_ENV}=1 to hide this notice)"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triples_parse_and_order_numerically() {
        assert_eq!(parse_triple("0.6.0"), Some(ReleaseTriple(0, 6, 0)));
        assert_eq!(parse_triple("1.2.30"), Some(ReleaseTriple(1, 2, 30)));
        assert!(parse_triple("0.10.0") > parse_triple("0.9.9"));
        assert!(parse_triple("1.0.0") > parse_triple("0.99.99"));
    }

    #[test]
    fn a_prerelease_running_version_compares_by_its_core_triple() {
        assert_eq!(parse_triple("0.7.0-next.1"), Some(ReleaseTriple(0, 7, 0)));
        assert_eq!(parse_triple("0.7.0+abcdef"), Some(ReleaseTriple(0, 7, 0)));
    }

    #[test]
    fn malformed_versions_do_not_parse() {
        for bad in ["", "0.6", "0.6.0.1", "v0.6.0", "0.6.x", "six"] {
            assert_eq!(parse_triple(bad), None, "{bad:?} should not parse");
        }
    }

    #[test]
    fn release_tags_parse_from_redirect_targets_and_bare_tags() {
        assert_eq!(
            parse_release_tag(
                "https://github.com/AgentWorkforce/relayhistory/releases/tag/sdk-ts-v0.6.0"
            ),
            Some(ReleaseTriple(0, 6, 0))
        );
        assert_eq!(
            parse_release_tag("sdk-ts-v1.2.3"),
            Some(ReleaseTriple(1, 2, 3))
        );
    }

    #[test]
    fn non_release_redirects_do_not_parse() {
        // No releases yet: GitHub redirects back to the releases index.
        assert_eq!(
            parse_release_tag("https://github.com/AgentWorkforce/relayhistory/releases"),
            None
        );
        // Prerelease and foreign tags never trigger the notice.
        assert_eq!(parse_release_tag("sdk-ts-v0.7.0-next.1"), None);
        assert_eq!(parse_release_tag("v0.6.0"), None);
    }

    #[test]
    fn no_warning_flag_and_env_opt_out_suppress() {
        let args = |list: &[&str]| list.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert!(suppressed(
            args(&["ai-hist", "--version", "--no-warning"]).into_iter(),
            None
        ));
        assert!(suppressed(
            args(&["ai-hist", "--no-warning", "--version"]).into_iter(),
            None
        ));
        assert!(suppressed(
            args(&["ai-hist", "--version"]).into_iter(),
            Some("1")
        ));
        assert!(suppressed(
            args(&["ai-hist", "--version"]).into_iter(),
            Some("true")
        ));
        assert!(!suppressed(
            args(&["ai-hist", "--version"]).into_iter(),
            None
        ));
        assert!(!suppressed(
            args(&["ai-hist", "--version"]).into_iter(),
            Some("")
        ));
        assert!(!suppressed(
            args(&["ai-hist", "--version"]).into_iter(),
            Some("0")
        ));
    }
}
