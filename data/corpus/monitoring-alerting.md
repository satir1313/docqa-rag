---
title: Monitoring and Alerting Guide
audience: sre
tags: monitoring, reliability
updated: 2026-02-27
---
# Monitoring and Alerting Guide

## Service level objectives
The public API targets 99.9% monthly availability and a p99 latency of 800 ms. The batch processing pipeline targets completion of daily jobs within 4 hours of the scheduled start.

## Error budget policy
A 99.9% availability objective allows about 43 minutes of downtime per month. When more than 50% of the monthly error budget is consumed, feature releases for that service require a reliability review. When the budget is exhausted, only reliability work ships until the next window.

## Alert severity mapping
Alerts map to incident severities: a page-level alert indicates SEV1 or SEV2 conditions and wakes the on-call engineer. Ticket-level alerts create a ticket for the next business day. Informational alerts only appear on dashboards.

## Infrastructure alerts
Disk usage above 85% on any node raises a ticket alert; above 95% raises a page. Memory saturation and CPU throttling alerts require 10 consecutive minutes above threshold to avoid noise.

## Dashboards
The Golden Signals dashboard shows latency, traffic, errors and saturation for each service. The On-call Overview dashboard lists active alerts and recent deploys.

## Reducing alert noise
Alerts must be actionable and link to a runbook. Alerts that fire more than 5 times per week without any action are reviewed and either tuned or deleted.
