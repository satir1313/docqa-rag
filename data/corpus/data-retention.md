---
title: Data Retention Standard
audience: all
tags: compliance, storage
updated: 2026-02-18
---
# Data Retention Standard

## Application logs
Application logs are kept in hot storage for 30 days and then moved to compressed archive storage for 365 days, after which they are deleted. Logs must not contain Restricted data such as tokens or customer content.

## Log rotation
On virtual machines, log files rotate when they reach 100 MB or at midnight UTC, whichever happens first. Up to 7 rotated files are retained locally before being shipped and removed.

## Backups
Daily backups are retained for 35 days. Monthly backups are retained for 12 months. Backups are encrypted and stored in a separate region from the primary database.

## Audit logs
Audit logs of administrative actions are immutable and retained for 7 years to satisfy regulatory requirements.

## Customer data deletion
When a customer terminates their contract, their data is deleted within 30 days. Deletion is confirmed to the customer in writing. Data in backups expires through the normal backup schedule and is not restored.

## Legal holds
A legal hold suspends deletion for the data in scope. Only the legal team can place or release a hold, and every hold is recorded in the compliance register.
