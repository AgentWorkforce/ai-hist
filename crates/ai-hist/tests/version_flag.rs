//! `--version` contract: the version line is stable, machine-readable stdout,
//! and the update notice never leaks into it. With stderr not a terminal (as
//! here), the update check is skipped entirely, so these tests are offline.

use std::process::Command;

fn version_output(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_ai-hist"))
        .args(args)
        .output()
        .expect("run ai-hist")
}

fn expected_version_line() -> String {
    // Tests build without AI_HIST_RELEASE_VERSION, so the binary falls back
    // to the crate version.
    format!("ai-hist {}", env!("CARGO_PKG_VERSION"))
}

#[test]
fn version_prints_only_the_version_line() {
    let out = version_output(&["--version"]);
    assert!(out.status.success());
    assert_eq!(
        String::from_utf8_lossy(&out.stdout).trim(),
        expected_version_line()
    );
    assert!(
        out.stderr.is_empty(),
        "unexpected stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn no_warning_is_accepted_on_either_side_of_version() {
    for args in [
        &["--version", "--no-warning"][..],
        &["--no-warning", "--version"][..],
    ] {
        let out = version_output(args);
        assert!(out.status.success(), "{args:?} failed");
        assert_eq!(
            String::from_utf8_lossy(&out.stdout).trim(),
            expected_version_line(),
            "{args:?}"
        );
        assert!(out.stderr.is_empty(), "{args:?} wrote to stderr");
    }
}

#[test]
fn short_version_flag_still_works() {
    let out = version_output(&["-V"]);
    assert!(out.status.success());
    assert_eq!(
        String::from_utf8_lossy(&out.stdout).trim(),
        expected_version_line()
    );
}

#[test]
fn help_still_renders_and_documents_no_warning() {
    let out = version_output(&["--help"]);
    assert!(out.status.success());
    let help = String::from_utf8_lossy(&out.stdout);
    assert!(help.contains("--no-warning"));
}

#[test]
fn parse_errors_still_fail_with_usage() {
    let out = version_output(&["--definitely-not-a-flag"]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("Usage"));
}
