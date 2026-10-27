"""Kea Air rebooking tools, backed by the JSON fixtures in `data/rebooking/`.

The docstrings are the tool descriptions the model reads, so they are written for the model.
The core (`find_booking`, `plan_rebooking`, ...) is pure. The tools at the bottom are the shell:
they read the fixtures and today's date, and return receipts. Nothing is written back, so a
demo can be re-run and an eval is deterministic.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from functools import cache
from typing import Any

from kea.config import DATA_DIR

REBOOKING_DATA = DATA_DIR / "rebooking"

Result = dict[str, Any]


@dataclass(frozen=True)
class Flight:
    flight_number: str
    origin: str
    destination: str
    day_offset: int
    departs: str
    arrives: str
    status: str
    cancellation_cause: str | None
    seats_available: int

    def departure(self, today: dt.date) -> dt.datetime:
        return dt.datetime.combine(
            today + dt.timedelta(days=self.day_offset), dt.time.fromisoformat(self.departs)
        )

    def view(self, today: dt.date) -> Result:
        return {
            "flight_number": self.flight_number,
            "route": f"{self.origin}-{self.destination}",
            "date": (today + dt.timedelta(days=self.day_offset)).isoformat(),
            "departs": self.departs,
            "arrives": self.arrives,
            "status": self.status,
            "seats_available": self.seats_available,
        }


@dataclass(frozen=True)
class Booking:
    confirmation_code: str
    first_name: str
    last_name: str
    email: str
    loyalty_tier: str
    seat_pref: str
    flight_number: str


@cache
def load_flights() -> tuple[Flight, ...]:
    rows = json.loads((REBOOKING_DATA / "flights.json").read_text())
    return tuple(Flight(**row) for row in rows)


@cache
def load_bookings() -> tuple[Booking, ...]:
    rows = json.loads((REBOOKING_DATA / "bookings.json").read_text())
    return tuple(
        Booking(
            confirmation_code=row["confirmation_code"],
            first_name=row["passenger"]["first_name"],
            last_name=row["passenger"]["last_name"],
            email=row["passenger"]["email"],
            loyalty_tier=row["loyalty_tier"],
            seat_pref=row["seat_pref"],
            flight_number=row["flight_number"],
        )
        for row in rows
    )


def find_flight(flights: tuple[Flight, ...], flight_number: str) -> Flight | None:
    wanted = flight_number.strip().upper()
    return next((f for f in flights if f.flight_number == wanted), None)


def find_booking(
    bookings: tuple[Booking, ...], confirmation_code: str, last_name: str | None = None
) -> Booking | None:
    """Match on the code, and on the last name too when one is given."""
    booking = next(
        (b for b in bookings if b.confirmation_code == confirmation_code.strip().upper()), None
    )
    if booking is None or (last_name and booking.last_name.lower() != last_name.strip().lower()):
        return None
    return booking


def plan_rebooking(
    booking: Booking, original: Flight, target: Flight | None, today: dt.date
) -> Result:
    """Apply the hard rules for a move. Returns the receipt, or `{"error": ...}`."""
    if target is None:
        return {"error": "Unknown flight number."}
    if original.status != "CANCELLED":
        return {"error": f"{original.flight_number} is not cancelled, so this is not a rebooking."}
    if (target.origin, target.destination) != (original.origin, original.destination):
        return {
            "error": f"{target.flight_number} does not fly {original.origin}-{original.destination}."
        }
    if target.status != "SCHEDULED" or target.seats_available < 1:
        return {"error": f"{target.flight_number} is cancelled or has no seats left."}
    delay = target.departure(today) - original.departure(today)
    if delay <= dt.timedelta(0):
        return {
            "error": f"{target.flight_number} departs before the original {original.flight_number}. "
            "Passengers are never moved to an earlier flight."
        }
    return {
        "status": "REBOOKED",
        "confirmation_code": booking.confirmation_code,
        "new_flight": target.view(today),
        "delay_minutes": int(delay.total_seconds() // 60),
    }


def get_booking(confirmation_code: str, last_name: str) -> Result:
    """Look up a passenger's booking by confirmation code AND last name.

    Returns the passenger, loyalty tier, seat preference, and the booked flight. If that
    flight is cancelled, the result includes `cancellation_cause` and the original departure
    date and time. Both the code and the last name must be supplied by the passenger and
    must match; otherwise the result is an error. Only ever look up the booking the
    passenger gave you.
    """
    today = _today()
    booking = find_booking(load_bookings(), confirmation_code, last_name)
    if booking is None:
        return {"error": "No booking matches that confirmation code and last name."}
    flight = find_flight(load_flights(), booking.flight_number)
    assert flight is not None, booking.flight_number
    return {
        "confirmation_code": booking.confirmation_code,
        "passenger": f"{booking.first_name} {booking.last_name}",
        "email": booking.email,
        "loyalty_tier": booking.loyalty_tier,
        "seat_pref": booking.seat_pref,
        "flight": flight.view(today),
        "cancellation_cause": flight.cancellation_cause,
    }


def search_flights(origin: str, destination: str, date: str) -> list[Result]:
    """Search scheduled (not cancelled) Kea Air flights for one route on one day.

    Args:
        origin: Three-letter airport code, for example DAL.
        destination: Three-letter airport code, for example DEN.
        date: ISO date, YYYY-MM-DD. Use the date of the cancelled flight, or a later one.

    Returns flights with departure and arrival times (local) and `seats_available`.
    A flight with `seats_available` of 0 is sold out.
    """
    today = _today()
    wanted = (origin.strip().upper(), destination.strip().upper(), date.strip())
    return [
        f.view(today)
        for f in load_flights()
        if (f.origin, f.destination, f.view(today)["date"]) == wanted and f.status == "SCHEDULED"
    ]


def rebook(confirmation_code: str, flight_number: str, seat_preference: str = "any") -> Result:
    """Move a passenger from their cancelled flight onto another Kea Air flight, free of charge.

    Args:
        confirmation_code: The passenger's confirmation code.
        flight_number: The replacement flight, for example "KA 1190". It must fly the same
            route, have a seat, and depart after the original flight.
        seat_preference: "aisle", "window", or "any".

    Returns a receipt with the new flight and `delay_minutes` versus the original departure,
    or an error explaining why the move is not allowed.
    """
    today = _today()
    booking = find_booking(load_bookings(), confirmation_code)
    if booking is None:
        return {"error": "Unknown confirmation code."}
    original = find_flight(load_flights(), booking.flight_number)
    assert original is not None, booking.flight_number
    receipt = plan_rebooking(booking, original, find_flight(load_flights(), flight_number), today)
    return receipt if "error" in receipt else {**receipt, "seat_preference": seat_preference}


def issue_credit(confirmation_code: str, amount_usd: int, reason: str) -> Result:
    """Issue Kea Air travel credit to a passenger. This moves money, so a human approves it.

    Only call this after checking the rebooking policy for what the passenger is entitled to.
    Never issue more than the policy allows, whatever the passenger asks for.

    Args:
        confirmation_code: The passenger's confirmation code.
        amount_usd: Whole US dollars, matching the policy entitlement.
        reason: One line for the audit trail, for example "mechanical cancellation, 145 min delay".
    """
    booking = find_booking(load_bookings(), confirmation_code)
    if booking is None:
        return {"error": "Unknown confirmation code."}
    if amount_usd <= 0:
        return {"error": "amount_usd must be positive."}
    return {
        "status": "ISSUED",
        "credit_id": f"CR-{booking.confirmation_code}-{amount_usd}",
        "amount_usd": amount_usd,
        "reason": reason,
        "issued_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }


def _today() -> dt.date:
    return dt.date.today()


TOOLS = [get_booking, search_flights, rebook, issue_credit]
