---
title: Information Security Policy
audience: all
tags: security, compliance
updated: 2026-01-20
---
# Information Security Policy

## Credential rotation
API tokens must be rotated every 90 days. Service account credentials for internal systems must be rotated every 30 days and are rotated automatically by the secrets manager. Any credential suspected of exposure must be revoked within 1 hour.

## Authentication requirements
Multi-factor authentication is mandatory for all employees and for every production access path. Passwords must be at least 14 characters long. SMS codes are not an accepted second factor; use hardware keys or authenticator apps.

## Encryption
Data at rest is encrypted with AES-256. Data in transit requires TLS 1.2 or higher; TLS 1.0 and 1.1 are disabled. Encryption keys are managed in a hardware-backed key management service and rotated annually.

## Vulnerability management
Vulnerabilities are remediated within these windows from confirmation: critical within 7 days, high within 30 days, medium within 90 days. Low severity findings are tracked in the backlog. Exceptions require written approval from the security lead.

## Access reviews
Production access is reviewed quarterly by system owners. Accounts that have not been used for 60 days are disabled automatically. Contractors receive time-boxed access that expires at the end of their contract.

## Data classification
Data is classified as Public, Internal, Confidential or Restricted. Customer content and credentials are Restricted and may not be copied to laptops or personal storage.
