import random  # noqa: F401 - available for the learner implementation
from dataclasses import dataclass

import torch


class DeviceUnavailableError(RuntimeError):
    """Raised when a requested accelerator cannot be used."""


@dataclass(frozen=True, slots=True)
class DevicePolicy:
    """A visible record of requested and actually selected tensor settings."""

    requested: str
    selected: torch.device
    dtype: torch.dtype
    reason: str

    @classmethod
    def resolve(cls, requested: str = "auto", dtype: torch.dtype = torch.float32) -> "DevicePolicy":
        """Resolve ``auto``, ``cpu``, or ``mps`` without a silent explicit fallback.

        Implement this in chapter Steps 2 and 3. Start with the explicit CPU
        branch, run its named checkpoint, and only then add validation, MPS,
        and automatic selection. Return ``cls(requested, device, dtype, reason)``.
        """
        raise NotImplementedError("Resolve the requested device policy")

    def report(self) -> dict[str, str]:
        """Return startup metadata suitable for logs and benchmark reports."""
        return {
            "requested_device": self.requested,
            "selected_device": self.selected.type,
            "dtype": str(self.dtype),
            "reason": self.reason,
        }


def seed_everything(seed: int) -> None:
    """Seed Python and every PyTorch backend used by this lesson.

    This is chapter Step 1. Guided experiment 5 demonstrates the calls you
    need; transfer that pattern here, including MPS when it is available.
    """
    del seed
    raise NotImplementedError("Seed Python and PyTorch random number generators")
