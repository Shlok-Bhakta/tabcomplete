"""Hard budget gate for paid synthetic data (ALLOW_PAID_SYNTHETIC=1).

Caps (env-overridable, tonight's defaults):
  MAX_TOTAL_SPEND_USD=2.00, MAX_PAID_EXAMPLES=2000, MAX_CANDIDATES_PER_STATE=3.
Accounting persists in data/generated/budget.json via atomic writes so a
restarted process cannot reset the budget. Unknown exact costs fall back to
conservative per-token estimates with a safety margin.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass

__all__ = [
    "Budget",
    "BudgetState",
    "DEFAULT_SPEND_CAP",
    "DEFAULT_EXAMPLE_CAP",
    "DEFAULT_CANDIDATE_CAP",
    "SAFETY_MARGIN",
    "BUDGET_PATH",
    "estimate_cost_usd",
]

DEFAULT_SPEND_CAP = 2.00
DEFAULT_EXAMPLE_CAP = 2000
DEFAULT_CANDIDATE_CAP = 3
SAFETY_MARGIN = 1.5  # 50% margin on estimated (non-reported) costs
BUDGET_PATH = os.path.join("data", "generated", "budget.json")


@dataclass
class BudgetState:
    spent_usd: float = 0.0
    paid_examples: int = 0
    updated_at: float = 0.0


def estimate_cost_usd(
    prompt_tokens: int, output_tokens: int, price_in_per_1k: float, price_out_per_1k: float
) -> float:
    """Conservative estimate with safety margin (exact reported cost preferred)."""
    raw = prompt_tokens / 1000 * price_in_per_1k + output_tokens / 1000 * price_out_per_1k
    return raw * SAFETY_MARGIN


class Budget:
    def __init__(
        self,
        path: str = BUDGET_PATH,
        spend_cap: float | None = None,
        example_cap: int | None = None,
        candidate_cap: int | None = None,
    ) -> None:
        self.path = path
        self.spend_cap = DEFAULT_SPEND_CAP if spend_cap is None else spend_cap
        self.example_cap = DEFAULT_EXAMPLE_CAP if example_cap is None else example_cap
        self.candidate_cap = DEFAULT_CANDIDATE_CAP if candidate_cap is None else candidate_cap
        self.state = self._load()

    @staticmethod
    def paid_enabled() -> bool:
        return os.environ.get("ALLOW_PAID_SYNTHETIC", "") == "1"

    @classmethod
    def from_env(cls, path: str = BUDGET_PATH) -> Budget:
        def env_float(name: str, default: float) -> float:
            try:
                return float(os.environ.get(name, str(default)))
            except ValueError:
                return default

        def env_int(name: str, default: int) -> int:
            try:
                return int(os.environ.get(name, str(default)))
            except ValueError:
                return default

        return cls(
            path=path,
            spend_cap=env_float("MAX_TOTAL_SPEND_USD", DEFAULT_SPEND_CAP),
            example_cap=env_int("MAX_PAID_EXAMPLES", DEFAULT_EXAMPLE_CAP),
            candidate_cap=env_int("MAX_CANDIDATES_PER_STATE", DEFAULT_CANDIDATE_CAP),
        )

    def _load(self) -> BudgetState:
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            return BudgetState(
                spent_usd=float(data.get("spent_usd", 0.0)),
                paid_examples=int(data.get("paid_examples", 0)),
                updated_at=float(data.get("updated_at", 0.0)),
            )
        except (FileNotFoundError, ValueError, KeyError, TypeError):
            return BudgetState()

    def _save(self) -> None:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = json.dumps(
            {
                "spent_usd": self.state.spent_usd,
                "paid_examples": self.state.paid_examples,
                "updated_at": time.time(),
            },
            indent=2,
        )
        fd, tmp = tempfile.mkstemp(dir=directory or ".", prefix=".budget-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp, self.path)  # atomic on POSIX
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def can_spend(self, est_cost_usd: float, num_examples: int = 1) -> bool:
        if not self.paid_enabled():
            return False
        if self.state.paid_examples + num_examples > self.example_cap:
            return False
        return self.state.spent_usd + est_cost_usd <= self.spend_cap

    def record(self, cost_usd: float, num_examples: int = 1) -> BudgetState:
        self.state.spent_usd += max(0.0, cost_usd)
        self.state.paid_examples += max(0, num_examples)
        self.state.updated_at = time.time()
        self._save()
        return self.state
