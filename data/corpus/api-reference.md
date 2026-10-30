---
title: Platform API Reference
audience: developer
tags: api, integration
updated: 2026-02-02
---
# Platform API Reference

## Authentication
All requests must include an `Authorization: Bearer <token>` header. Tokens are created in the developer console and are scoped to a single project. Tokens expire after 90 days unless they are rotated before expiry.

## Rate limits
Limits are applied per token per minute: Free tier 60 requests, Pro tier 600 requests, Enterprise tier 6,000 requests. Exceeding the limit returns HTTP 429 with a `Retry-After` header giving the number of seconds to wait. A burst allowance of 2x the per-minute limit is available for up to 10 seconds.

## Pagination
List endpoints use cursor-based pagination. Pass `limit` (default 50, maximum 200) and the `next_cursor` value from the previous response. Offset pagination is not supported because cursors stay stable while the underlying data changes.

## Idempotency
POST requests accept an `Idempotency-Key` header. The server stores the result of the first request with a given key for 24 hours and replays it for retries. Reusing a key with a different payload returns HTTP 422.

## Error format
Errors return a JSON body with `code`, `message` and `request_id`. Include the request_id when contacting support. Server errors (5xx) are safe to retry with exponential backoff; client errors (4xx) other than 429 should not be retried.

## Webhooks
Webhook deliveries are signed with HMAC-SHA256 using the endpoint secret. The signature is sent in the `X-Northwind-Signature` header. Consumers must verify the signature and reject deliveries older than 5 minutes to prevent replay attacks. Failed deliveries are retried with exponential backoff, up to 8 attempts over 24 hours.
