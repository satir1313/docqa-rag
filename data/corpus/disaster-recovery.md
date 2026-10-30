---
title: Disaster Recovery Plan
audience: sre
tags: dr, reliability, compliance
updated: 2026-03-05
---
# Disaster Recovery Plan

## Service tiers
Services are grouped by criticality. Tier 1 services have a recovery time objective (RTO) of 1 hour and a recovery point objective (RPO) of 5 minutes. Tier 2 services have an RTO of 4 hours and an RPO of 1 hour. Tier 3 services have an RTO of 24 hours and an RPO of 24 hours.

## Regional failover
Tier 1 services run active-passive across two regions. Failover is initiated by the incident commander and executed with the `drctl failover` command, which promotes the replica database and shifts DNS traffic. DNS time to live is kept at 60 seconds to speed up cutover.

## Restore testing
Backups are restore-tested monthly on a sample of databases. A full regional failover exercise is held every 6 months and results are reviewed with leadership.

## Declaring a disaster
A disaster is declared when a primary region is expected to be unavailable for longer than the RTO of its Tier 1 services. Only the VP of Engineering or the incident commander can declare it.

## Post-recovery steps
After failing over, freeze deploys, confirm replication lag is zero on the new primary, and begin planning the failback. Failback is performed during a low-traffic window and is treated as a planned change.
