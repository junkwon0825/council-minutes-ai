#!/usr/bin/env python3
"""질문 파이프라인(agent/query_pipeline.py)의 실행 단계가 부르는 외부 데이터 작업 모음.

LLM은 이 함수들을 직접 고르지 않는다. 파이프라인이 질문 해석 결과(의도·주제·지역·모드)를 검증한 뒤
의도별로 정해진 순서대로 호출한다. 단계·후보 순위·점수는 src/external_followup.py의 결정적 규칙으로 계산하며,
외부로 나가는 값은 check_keyword를 통과한 짧은 정책어와 KOSIS 통계표 ID뿐이다.
"""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import external_followup as FU  # noqa: E402
import external_policy_sources as EXT  # noqa: E402


@dataclass
class FollowupTools:
    """대표 이슈 결과(32a 사전 계산)를 우선 쓰고, 조건이 다르거나 새 주제면 같은 규칙으로 즉석 계산한다."""

    issue_payloads: list[dict[str, Any]]
    search_service: Any | None = None
    default_mode: str = field(default_factory=lambda: os.environ.get("COUNCIL_FOLLOWUP_MODE", "backtest"))
    minutes_cutoff: str = field(
        default_factory=lambda: os.environ.get("COUNCIL_MINUTES_CUTOFF", FU.DEFAULT_MINUTES_CUTOFF)
    )
    search_top_k: int = 40
    _cache: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        cutoff = FU.parse_day(self.minutes_cutoff)
        if cutoff is None:
            raise ValueError(f"회의록 기준일 형식 오류: {self.minutes_cutoff!r}")
        if self.default_mode not in FU.MODES:
            raise ValueError(f"COUNCIL_FOLLOWUP_MODE는 {FU.MODES} 중 하나여야 합니다")
        self._minutes_cutoff = cutoff

    def as_of_for(self, mode: str) -> date:
        """live = 조회일(오늘), backtest = 회의록 기준일 — LLM이 아니라 코드가 정한다"""
        return date.today() if mode == "live" else self._minutes_cutoff

    def available_sources(self) -> dict[str, bool]:
        return {source: EXT.key_available(source) for source in EXT.DATA_SOURCES}

    def _cached(self, key: str, build) -> dict[str, Any]:
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        result = build()
        with self._lock:
            self._cache[key] = result
        return result

    def issue_followup(
        self, payload: dict[str, Any], mode: str, terms: list[str], indicator_ids: list[str],
    ) -> dict[str, Any]:
        """대표 이슈 후속 흐름. 32a 결과가 같은 모드이고 충분히 최신이면 그대로, 아니면 즉석 계산
        (live는 외부 API 캐시 주기와 같은 7일 이내, backtest는 같은 기준일)"""
        as_of = self.as_of_for(mode)
        stored = payload.get("external_followup") or {}
        stored_as_of = FU.parse_day(stored.get("as_of"))
        fresh = stored_as_of is not None and (
            (as_of - stored_as_of).days <= 7 if mode == "live" else stored_as_of == as_of
        )
        if stored.get("mode") == mode and fresh:
            return {"mode_used": "precomputed_32a", "sent_externally": [], "result": stored}
        meta = payload.get("issue") or {}
        keywords, rejected = FU.external_keywords(meta, None, 3)
        keywords = list(dict.fromkeys([*terms, *keywords]))[:4]
        relevance = list(dict.fromkeys([*keywords, *(meta.get("aliases") or [])]))
        indicators = indicator_ids or FU.match_indicator_ids(relevance)
        cache_key = f"issue|{meta.get('issue_id')}|{mode}|{as_of}|{keywords}|{indicators}"
        result = self._cached(cache_key, lambda: FU.build_followup(
            {"issue_id": meta.get("issue_id"), "issue_name": meta.get("issue_name")},
            payload.get("propagation") or [], keywords, as_of,
            mode=mode, minutes_cutoff=self._minutes_cutoff,
            issue_forecast=payload.get("multi_issue_forecast") or {},
            relevance_terms=relevance, indicator_ids=indicators, rejected_keywords=rejected,
        ))
        return {"mode_used": "computed_on_request", "sent_externally": keywords, "result": result}

    def topic_followup(
        self, topic: str, terms: list[str], mode: str, indicator_ids: list[str],
    ) -> dict[str, Any]:
        """대표 이슈에 없는 주제: 로컬 회의록 검색으로 의회별 논의를 모아 같은 규칙으로 계산 (검수 전 참고값)"""
        if self.search_service is None:
            return {"error": "회의록 검색(pgvector)이 연결되지 않아 새 주제의 회의록 관측을 모을 수 없음"}
        keywords = [EXT.check_keyword(term) for term in terms][:4]
        as_of = self.as_of_for(mode)

        def build() -> dict[str, Any]:
            # 로컬 벡터 검색만 사용 — 회의록 원문은 서버 밖으로 나가지 않음
            rows: list[dict[str, Any]] = []
            for keyword in keywords[:2]:
                found = self.search_service.search(
                    query=keyword, top_k=self.search_top_k,
                    candidate_k=max(100, self.search_top_k * 5), include_parent=False,
                )
                rows.extend(found.get("results") or [])
            return FU.build_followup(
                {"issue_id": None, "issue_name": topic, "id_status": "provisional"},
                FU.propagation_from_search(rows, self._minutes_cutoff), keywords, as_of,
                mode=mode, minutes_cutoff=self._minutes_cutoff,
                relevance_terms=keywords, indicator_ids=indicator_ids,
            )

        result = self._cached(f"topic|{keywords}|{mode}|{as_of}|{indicator_ids}", build)
        return {"mode_used": "provisional_search_based", "sent_externally": keywords, "result": result}

    def regional_statistics(self, indicator_ids: list[str], mode: str) -> dict[str, Any]:
        if not indicator_ids:
            return {"error": "질문 주제와 연결되는 KOSIS 지표가 없음"}
        if not EXT.key_available("kosis"):
            return {"error": "KOSIS_API_KEY가 설정되지 않아 통계를 조회할 수 없음"}
        import regional_stats as RS  # 순환 import 방지

        as_of = self.as_of_for(mode)
        results = [
            EXT.safe_call(RS.get_regional_indicator, indicator_id, as_of.isoformat(), mode)
            for indicator_id in indicator_ids
        ]
        return {"as_of": as_of.isoformat(), "mode": mode, "indicators": results,
                "sent_externally": [f"KOSIS:{indicator_id}" for indicator_id in indicator_ids]}

    def national_bills(self, terms: list[str], mode: str) -> dict[str, Any]:
        if not EXT.key_available("assembly_bills"):
            return {"error": "ASSEMBLY_API_KEY가 설정되지 않아 국회 의안을 조회할 수 없음"}
        as_of = self.as_of_for(mode).isoformat()
        items: dict[tuple[str, str], dict[str, Any]] = {}
        errors = []
        for term in terms[:3]:
            result = EXT.safe_call(EXT.search_national_bills, term, as_of)
            if result.get("error"):
                errors.append(result["error"])
                continue
            for item in result.get("items") or []:
                if FU.relevant(item.get("bill"), terms):
                    items.setdefault((str(item.get("bill")), str(item.get("date"))), item)
        rows = sorted(items.values(), key=lambda row: row.get("date") or "", reverse=True)
        return {"as_of": as_of, "count": len(rows), "items": rows[:8], "errors": errors,
                "attribution": EXT.DATA_SOURCES["assembly_bills"]["attribution"],
                "sent_externally": terms[:3]}
