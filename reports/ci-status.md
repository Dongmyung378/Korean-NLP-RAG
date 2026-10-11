# Hosted verification status

[English](ci-status.md) | [한국어](ci-status.ko.md)

The repository contains a read-only Windows/Linux CPU verification workflow. The earlier lightweight workflow was removed because it did not install the dependencies required by the current suite.

On 2026-10-11, a dispatch of `Verify portable CPU code` against the published main branch was rejected by GitHub with HTTP 422: `Actions has been disabled for this user.` No hosted run was created, so this release does not claim passing GitHub Actions. Local checks and isolated environment verification are recorded in [results](results.md) and [environment verification](environment-verification.md).

Authenticated publication to the existing repository succeeded. Anonymous HTTP access returned 404 during validation; no cause is inferred. A publicly served [source ZIP](https://park-dongmyung.com/downloads/korean-nlp-rag-source.zip) provides the code and documentation without requiring GitHub access.
