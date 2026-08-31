import random
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
        """Resolve ``auto``, ``cpu``, or ``mps`` without a silent explicit fallback."""
        if requested not in {"auto", "cpu", "mps"}:
            raise ValueError("requested device must be one of: auto, cpu, mps")
        if requested == "cpu":
            return cls(requested, torch.device("cpu"), dtype, "CPU was explicitly requested")
        if requested == "mps":
            if not torch.backends.mps.is_available():
                if not torch.backends.mps.is_built():
                    detail = "this PyTorch build has no MPS support"
                else:
                    detail = "MPS is not available on this macOS device"
                raise DeviceUnavailableError(f"MPS was explicitly requested, but {detail}")
            return cls(requested, torch.device("mps"), dtype, "MPS was explicitly requested")
        if torch.backends.mps.is_available():
            return cls(requested, torch.device("mps"), dtype, "auto selected available MPS")
        return cls(
            requested,
            torch.device("cpu"),
            dtype,
            "auto selected CPU because MPS is unavailable",
        )

    def report(self) -> dict[str, str]:
        """Return startup metadata suitable for logs and benchmark reports."""
        return {
            "requested_device": self.requested,
            "selected_device": self.selected.type,
            "dtype": str(self.dtype),
            "reason": self.reason,
        }


def seed_everything(seed: int) -> None:
    """Seed Python and every PyTorch backend used by this lesson."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
