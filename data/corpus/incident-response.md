---
title: Incident Response Policy
audience: sre
tags: incident, oncall, reliability
updated: 2026-03-14
---
# Incident Response Policy

## Overview
This policy defines how Northwind engineers detect, classify and resolve production incidents. It applies to all customer-facing services and to internal systems that customer-facing services depend on.

## Severity levels
- SEV1: complete outage or data loss affecting more than 25% of customers. Acknowledge within 5 minutes and staff an incident commander immediately.
- SEV2: a major feature is degraded for a subset of customers, or a critical internal system is down. Acknowledge within 15 minutes.
- SEV3: minor degradation with a workaround available. Acknowledge within 4 business hours.
- SEV4: cosmetic issue or request with no customer impact. Handled in normal sprint planning.

## Escalation path
The primary on-call engineer has 10 minutes to acknowledge a page. If the page is unacknowledged it escalates to the secondary on-call engineer, and after a further 10 minutes to the engineering manager on duty. For SEV1 incidents the VP of Engineering is notified automatically.

## Incident commander role
The incident commander (IC) coordinates the response and does not debug. The IC assigns a communications lead and a scribe, decides when to roll back, and declares the incident resolved. A handover between incident commanders must be explicit and recorded in the incident channel.

## Communication cadence
For SEV1 incidents, status page updates are posted every 30 minutes until mitigation. For SEV2 incidents, updates are posted every 60 minutes. Customer-facing messages must be reviewed by the communications lead before publishing.

## Postmortems
A blameless postmortem is required for every SEV1 and SEV2 incident. The draft must be published within 5 business days of resolution and reviewed in the weekly reliability meeting. Each action item needs an owner and a due date.
