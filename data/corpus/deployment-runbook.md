---
title: Deployment Runbook
audience: developer, sre
tags: deploy, release
updated: 2026-03-01
---
# Deployment Runbook

## Release process
Every production release goes through a canary stage. The new version receives 5% of traffic for 15 minutes before progressive rollout to 25%, 50% and 100%. Each step is gated on health checks.

## Automatic rollback
The deploy pipeline rolls back automatically when the error rate of the canary exceeds 2% or p99 latency exceeds 1.5 times the baseline. An automatic rollback restores the previous artifact within 3 minutes.

## Deploy freezes
Production deploys are frozen after 14:00 local time on Fridays and during company holidays. Emergency fixes during a freeze require approval from the on-call engineering manager.

## Database migrations
Schema changes follow the expand and contract pattern: add new columns first, deploy code that writes both, backfill, switch reads, and only then drop the old columns in a later release. Migrations must be backward compatible with the previous application version.

## Feature flags
New user-facing behaviour ships behind a feature flag that defaults to off. Flags must have an owner and an expiry date, and stale flags are removed within 60 days of full rollout.

## Manual rollback procedure
If the automatic rollback fails, run `deployctl rollback --service <name> --to previous` and notify the incident channel. Verify recovery on the service dashboard before closing the change.
