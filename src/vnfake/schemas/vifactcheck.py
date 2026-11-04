"""ViFactCheck sample schema (Data contract per ARCHITECTURE.md §4.3)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ViFactCheckSample(BaseModel):
    """One ViFactCheck sample.

    Field mapping (raw JSON keys are capitalised):
        Statement -> statement
        Context   -> context    (analysis only — NEVER model input)
        labels    -> labels     (0=real, 1=fake, 2=unverifiable)
        Evidence  -> evidence

    The 3-class mapping is authoritative in ``configs/base.yaml: label_map``;
    the schema only enforces the set of valid integers.
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    statement: str = Field(alias="Statement", min_length=1)
    context: str = Field(default="", alias="Context")
    labels: Literal[0, 1, 2]
    evidence: str = Field(alias="Evidence")
