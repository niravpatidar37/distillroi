"""Teacher spend cap SHARED by every router instance (Cloud Run scales the router horizontally).

The local JSONL ledger (``spend.py``) is per process/instance on Cloud Run, so on its own a $7 cap with 4
instances is either split ($1.75 each) or exceeded (4 x $7). This module adds a cap that holds across
instances.

Design: **budget leases** on one Firestore document ``spend_caps/<project>`` with two integer fields
(micro-USD):

* ``granted_micros``: budget handed out to instances (counts against the cap, consumed or not)
* ``spent_micros``: actual spend reported back (observability; never used to raise the cap)

An instance takes a lease (default $0.05) with a compare-and-swap: read the doc, check ``granted + lease <=
cap``, write with a ``currentDocument.updateTime`` precondition (or ``exists=false`` on first use) and retry
on conflict. It then spends from the lease locally (worst-case reservation before each call, actual cost
after) and only touches Firestore again when the lease runs out, so the document sees about one write per
lease, far below Firestore's ~1 sustained write/s per document guidance. On graceful shutdown the unused part
is returned. A crash strands at most one lease per instance: the cap gets *stricter*, never looser (fail
closed). If Firestore is unreachable, a reservation that needs a new lease raises ``BudgetExceeded`` (fail
closed).

Guarantee: sum of teacher spend over all instances <= ``cap`` (as long as each call's actual cost <= its
worst-case estimate, which ``teacher._chat`` guarantees by pricing prompt + max_tokens). Cost: ~2 reads + 1-2
writes per lease; at Firestore list price ($0.03 / 100K reads, $0.09 / 100K writes, us-central1,
https://cloud.google.com/firestore/pricing) this is effectively free, and Firestore's daily free tier covers
it. Memorystore would cost an always-on instance plus VPC egress for the same job.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from distill.spend import BudgetExceeded

log = logging.getLogger("distill.shared_spend")

MICROS = 1_000_000


def to_micros(usd: float) -> int:
    """Round UP so estimates are never under-counted."""
    return max(0, math.ceil(usd * MICROS - 1e-9))


class SharedBudget(Protocol):
    async def reserve(self, estimate_usd: float) -> None: ...
    async def settle(self, estimate_usd: float, actual_usd: float) -> None: ...
    async def release(self, estimate_usd: float) -> None: ...
    async def ready(self) -> bool: ...
    async def aclose(self) -> None: ...


# ---------------------------------------------------------------- Firestore document with CAS


@dataclass
class DocState:
    granted_micros: int = 0
    spent_micros: int = 0
    update_time: str | None = None  # None = document does not exist yet


class CasConflict(Exception):
    """The document changed between read and write (another instance won the race)."""


# Fixed metadata-server endpoint (never user-controlled -> no SSRF surface).
_METADATA_TOKEN_URL = (
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"  # noqa: S105  # nosec B105
)
_FIRESTORE_API = "https://firestore.googleapis.com/v1"


class FirestoreDoc:
    """Minimal Firestore REST client for one document: get + conditional commit.

    Auth: on GCP an OAuth access token from the metadata server (router service account, granted
    roles/datastore.user only on this database via an IAM condition). Against the emulator
    (``FIRESTORE_EMULATOR_HOST``) the emulator's ``Bearer owner`` convention is used.
    """

    def __init__(
        self,
        project: str,
        database: str,
        collection: str,
        doc_id: str,
        *,
        emulator_host: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout_s: float = 3.0,
    ) -> None:
        self.base = f"http://{emulator_host}/v1" if emulator_host else _FIRESTORE_API
        self.emulator = emulator_host is not None
        self.db_path = f"projects/{project}/databases/{database}"
        self.name = f"{self.db_path}/documents/{collection}/{doc_id}"
        self._client = client or httpx.AsyncClient(timeout=timeout_s)
        self._token: str | None = None
        self._token_exp = 0.0

    async def _headers(self) -> dict[str, str]:
        if self.emulator:
            return {"Authorization": "Bearer owner"}
        if self._token is None or time.monotonic() >= self._token_exp:
            r = await self._client.get(
                _METADATA_TOKEN_URL, headers={"Metadata-Flavor": "Google"}, timeout=3.0
            )
            r.raise_for_status()
            data = r.json()
            self._token = str(data["access_token"])
            self._token_exp = time.monotonic() + max(60.0, float(data.get("expires_in", 3600)) - 300.0)
        return {"Authorization": f"Bearer {self._token}"}

    async def get(self) -> DocState:
        r = await self._client.get(f"{self.base}/{self.name}", headers=await self._headers())
        if r.status_code == 404:
            return DocState()
        r.raise_for_status()
        data = r.json()
        fields = data.get("fields") or {}

        def num(k: str) -> int:
            v = fields.get(k) or {}
            return int(v.get("integerValue", 0))

        return DocState(num("granted_micros"), num("spent_micros"), str(data["updateTime"]))

    async def commit(self, prev: DocState, granted_micros: int, spent_micros: int) -> None:
        """Write both fields iff the document is unchanged since ``prev`` was read; else raise CasConflict."""
        precondition: dict[str, Any] = (
            {"updateTime": prev.update_time} if prev.update_time is not None else {"exists": False}
        )
        body = {
            "writes": [
                {
                    "update": {
                        "name": self.name,
                        "fields": {
                            "granted_micros": {"integerValue": str(granted_micros)},
                            "spent_micros": {"integerValue": str(spent_micros)},
                        },
                    },
                    "currentDocument": precondition,
                }
            ]
        }
        r = await self._client.post(
            f"{self.base}/{self.db_path}/documents:commit", json=body, headers=await self._headers()
        )
        if r.status_code == 200:
            return
        status = ""
        try:
            status = str((r.json().get("error") or {}).get("status", ""))
        except (ValueError, AttributeError):
            pass
        # Verified against the Firestore emulator: stale updateTime -> 400 FAILED_PRECONDITION,
        # exists=false on an existing doc -> 409 ALREADY_EXISTS; contention can also surface as 409 ABORTED.
        if r.status_code in (400, 409) and status in ("FAILED_PRECONDITION", "ALREADY_EXISTS", "ABORTED"):
            raise CasConflict(status)
        r.raise_for_status()
        raise httpx.HTTPStatusError(f"unexpected status {r.status_code}", request=r.request, response=r)

    async def aclose(self) -> None:
        await self._client.aclose()


class CasDoc(Protocol):
    async def get(self) -> DocState: ...
    async def commit(self, prev: DocState, granted_micros: int, spent_micros: int) -> None: ...
    async def aclose(self) -> None: ...


# ---------------------------------------------------------------- lease-based shared budget


class LeaseBudget:
    """Cross-instance spend cap. One instance = one LeaseBudget; all share the same CasDoc."""

    def __init__(
        self, doc: CasDoc, cap_usd: float, lease_usd: float = 0.05, *, max_cas_attempts: int = 20
    ) -> None:
        self.doc = doc
        self.cap = to_micros(cap_usd)
        self.lease = max(1, to_micros(lease_usd))
        self.max_cas_attempts = max_cas_attempts
        self._leased = 0  # micro-USD granted to this instance and not yet returned
        self._spent = 0  # actual spend charged against the current leases
        self._reserved = 0  # in-flight worst-case estimates
        self._unreported = 0  # spend not yet added to spent_micros in the shared doc
        self._mu = asyncio.Lock()

    @property
    def available(self) -> int:
        return self._leased - self._spent - self._reserved

    async def _update(self, grant_delta_fn: Any, report: int) -> int:
        """CAS loop: ``grant_delta_fn(state) -> delta`` (may raise BudgetExceeded); returns the delta."""
        for attempt in range(self.max_cas_attempts):
            st = await self.doc.get()
            delta = int(grant_delta_fn(st))
            try:
                await self.doc.commit(st, st.granted_micros + delta, st.spent_micros + report)
                return delta
            except CasConflict:
                await asyncio.sleep(min(0.2, 0.005 * 2**attempt))
        raise BudgetExceeded("shared spend store: too much contention, refusing (fail closed)")

    async def reserve(self, estimate_usd: float) -> None:
        need = to_micros(estimate_usd)
        async with self._mu:
            if self.available < need:
                shortfall = need - self.available

                def grant(st: DocState) -> int:
                    room = self.cap - st.granted_micros
                    want = max(self.lease, shortfall)
                    if room < shortfall:
                        raise BudgetExceeded(
                            f"shared teacher spend cap {self.cap / MICROS:.4f} USD would be exceeded"
                        )
                    return min(want, room)

                report = self._unreported
                try:
                    got = await self._update(grant, report)
                except BudgetExceeded:
                    raise
                except (httpx.HTTPError, KeyError, ValueError) as e:
                    log.warning("spend_store_unavailable", extra={"error": type(e).__name__})
                    raise BudgetExceeded("shared spend store unavailable (fail closed)") from e
                self._unreported -= report
                self._leased += got
            self._reserved += need

    async def settle(self, estimate_usd: float, actual_usd: float) -> None:
        est, act = to_micros(estimate_usd), to_micros(actual_usd)
        async with self._mu:
            self._reserved = max(0, self._reserved - est)
            # actual <= estimate by construction (prompt + max_tokens at list price): never overdrawn
            self._spent += act
            self._unreported += act

    async def release(self, estimate_usd: float) -> None:
        async with self._mu:
            self._reserved = max(0, self._reserved - to_micros(estimate_usd))

    async def ready(self) -> bool:
        try:
            await self.doc.get()
            return True
        except (httpx.HTTPError, KeyError, ValueError):
            return False

    async def aclose(self) -> None:
        """Return the unused part of the lease and report spend (graceful shutdown)."""
        async with self._mu:
            unused = max(0, self._leased - self._spent - self._reserved)
            report = self._unreported
            if unused or report:
                try:
                    await self._update(lambda st: -min(unused, st.granted_micros), report)
                    self._leased -= unused
                    self._unreported = 0
                except (BudgetExceeded, httpx.HTTPError, KeyError, ValueError) as e:
                    log.warning("spend_lease_not_returned", extra={"error": type(e).__name__})
        await self.doc.aclose()

    def snapshot(self) -> dict[str, float]:
        return {
            "leased_usd": self._leased / MICROS,
            "spent_usd": self._spent / MICROS,
            "reserved_usd": self._reserved / MICROS,
            "cap_usd": self.cap / MICROS,
        }
