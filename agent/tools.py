#!/usr/bin/env python3
"""내일 Agent가 호출할 수 있는 로컬 검색 도구 계층."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from council_vector_runtime import CouncilSearchService, SearchFilters


SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_council_evidence",
        "description": "광역 지방의회 회의록에서 정책 질의와 관련된 원문 근거를 검색한다.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "검색할 정책·현안"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 30, "default": 10},
                "council": {"type": ["string", "null"]},
                "year": {"type": ["integer", "null"]},
                "committee": {"type": ["string", "null"]},
                "date_from": {"type": ["string", "null"]},
                "date_to": {"type": ["string", "null"]},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


@dataclass
class CouncilAgentTools:
    """외부 API 없이 검색 서비스를 Agent tool-call 형태로 노출한다."""

    search_service: CouncilSearchService

    def search_council_evidence(
        self,
        query: str,
        top_k: int = 10,
        council: str | None = None,
        year: int | None = None,
        committee: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict[str, Any]:
        return self.search_service.search(
            query=query,
            top_k=top_k,
            candidate_k=max(100, top_k * 10),
            filters=SearchFilters(
                council=council, year=year, committee=committee,
                date_from=date_from, date_to=date_to,
            ),
            include_parent=True,
        )

    def dispatch(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if tool_name != "search_council_evidence":
            raise ValueError(f"지원하지 않는 도구: {tool_name}")
        return self.search_council_evidence(**arguments)
