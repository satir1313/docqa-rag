---
title: Engineering Onboarding Guide
audience: developer
tags: onboarding, process
updated: 2026-01-09
---
# Engineering Onboarding Guide

## First week checklist
On day one you receive your laptop and identity account. By the end of the first week complete security training, request repository access through the access portal, and ship a small documentation change to learn the release flow.

## Code review rules
Changes to production services need 2 approving reviews, at least one from a code owner. Documentation-only changes need 1 approval. Authors may not approve their own pull requests.

## Branch naming
Branches follow `type/ticket-short-description`, for example `fix/NW-1234-null-cursor`. Commits use the conventional commits format.

## On-call shadowing
Engineers shadow the on-call rotation after 8 weeks and join the primary rotation after 6 months. Shadowing means joining one full week of the rotation without being paged first.

## Development environment
Local development uses containers defined in the repository. Secrets are never stored in the repository; fetch development credentials from the secrets manager with `nwctl login`.

## Who to ask
Your onboarding buddy is the first contact for questions. Platform questions go to the platform channel; security questions go to the security channel.
