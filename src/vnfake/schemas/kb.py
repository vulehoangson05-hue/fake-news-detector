"""KB record schema (Data contract per ARCHITECTURE.md §4.1)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class KBRecord(BaseModel):
    """A single KB article (Dân Trí / VnExpress).

    The raw JSONL field is ``date``; it is exposed here as ``crawl_date``
    because it is the **crawl timestamp**, NOT the publish date
    (ARCHITECTURE §4.1). Do NOT use it for temporal filtering.

    KB only contains real news; ``label`` is constrained to ``"real"``.
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: str = Field(min_length=1)
    title: str
    text: str
    label: Literal["real"] = "real"
    url: str = Field(min_length=1)
    crawl_date: str = Field(alias="date")
    published_date: str | None = None
    ingest_source: Literal["static", "rss"] = "static"
