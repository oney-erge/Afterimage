# Security policy

## Scope

Afterimage is a local inference engine and control server. It is designed
to run on hardware you control, not as a multi-tenant public service. The
FastAPI server (`afterimage serve`) has no authentication by default and
binds to `127.0.0.1` unless you explicitly pass `--host 0.0.0.0`. Do not
expose it to an untrusted network without putting a reverse proxy with
authentication in front of it.

## Reporting a vulnerability

Please report security issues privately rather than as a public GitHub
issue. Open a private security advisory on the repository
(GitHub → Security → Report a vulnerability), or contact the maintainer
directly through the contact information on their GitHub profile
([@oney-erge](https://github.com/oney-erge)).

Include what you found, how to reproduce it, and its potential impact.
We'll acknowledge receipt and follow up with next steps.

## Supported versions

This project is pre-1.0 and does not yet maintain parallel release
branches. Security fixes land on `main`.
