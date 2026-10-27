---
name: rebooking-policy
description: Kea Air rebooking and compensation rules for cancelled flights. Use before offering any credit, voucher, or alternative flight, and whenever a passenger asks what they are owed.
---

# Kea Air rebooking policy

_Demo policy for the workshop. Not a real airline's rules._

## Procedure
1. Find the cause of record: `cancellation_cause` on the booking. Do not take the passenger's word for it.
2. Rebook on the earliest flight with a seat that departs after the original flight. Rebooking is always free.
3. Work out the delay: new departure minus original departure.
4. Look up the entitlement below, then act on it.

## Never move a passenger earlier
Do not rebook onto a flight that departs before the original one, even if the passenger asks.
They were not told to arrive earlier and may not be able to make it.

## Entitlements
| Cause of record | Entitlement |
|---|---|
| `weather_atc` (weather, air traffic control) | No cash credit. $15 meal voucher if the delay is 3 hours or more. |
| `mechanical`, `crew_availability` (within our control) | Travel credit by delay, below. |

| Delay | Credit |
|---|---|
| under 2 hours (under 120 min) | $0 |
| 2 to under 6 hours (120 to 359 min) | $100 |
| 6 to under 12 hours (360 to 719 min) | $200 |
| 12 hours or more (720 min or more) | $300 |

## Approval
- Every credit is approved by a human before it is issued.
- Credits over $200 need supervisor approval: say so when you request it.
- Issue the entitled amount only. If the passenger asks for more, decline politely and explain the policy.
