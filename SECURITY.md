# Security policy

## Reporting a vulnerability

Please report vulnerabilities **privately** through GitHub's private vulnerability reporting:
[Report a vulnerability](https://github.com/niravpatidar37/distillroi/security/advisories/new)
(Security tab → "Report a vulnerability"). Do not open a public issue for security problems.

Include what you found, how to reproduce it, and the impact you expect. You should get a first reply within
7 days. This is a single-maintainer portfolio project, so there is no bug bounty.

## Scope

In scope: the router API (`src/distill/app.py`, `src/distill/router.py`), the teacher/student clients, the spend
cap and kill switch, log redaction, the Dockerfile/compose stack, the Terraform module and the CI/CD workflows.

The controls and the tests that check them are listed in the README's
[Security model](README.md#security-model). Known residual risks (for example, prompt injection can still
change which valid label is returned) are documented there and are not vulnerabilities by themselves.

## Supported versions

Only `main` is supported.
