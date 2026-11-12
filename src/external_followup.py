"""타 지방의회 후속 논의 흐름·전망 계산 로직 (32a 배치와 Agent 질문 파이프라인이 공유).

입력은 회의록 관측(의회별 최초·최근 관측일, 언급량)과 외부 공공데이터(자치법규·국회 의안·KOSIS 통계)이며,
단계·후보 순위·점수는 모두 이 모듈의 결정적 규칙으로 계산한다. LLM은 결과를 설명만 하고 숫자를 만들지 않는다.

시점 모드
  live      외부 데이터는 조회일(as_of)까지 공개된 것 전부. 회의록 기준일 이후 조례는 '이미 일어난 후속 조치'로
            따로 표시하고, 전망 대상은 조회일 이후. 회의록 기준 모델 확률(2026-01 대상)은 점수에 쓰지 않음
  backtest  외부 데이터는 기준일(as_of)까지만, 통계는 기준일 시점에 공표된 값만. 기준일 이후 실제 '제정'으로 사후 점검
회의록 근거는 두 모드 모두 회의록 기준일(minutes_cutoff)까지다(보유 데이터 범위).
결과는 reference_only로 회의록 기반 예측 모델 입력에 쓰지 않는다.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import date, datetime
from typing import Any

import external_policy_sources as EXT


SCHEMA_VERSION = "council_external_followup_v2"
MODES = ("live", "backtest")
DEFAULT_MINUTES_CUTOFF = "2025-12-31"
ORDINANCE_KINDS = {"조례", "규칙"}
GENERIC_WORDS = {"정책", "대응", "지원", "대책", "문제", "강화", "확대", "개선", "및", "등"}
# 가중치는 백테스트 결과를 보고 고르지 않고 미리 고정한다(같은 자료로 맞추면 정확도가 부풀려짐).
# 신호가 '해당 없음'(None)이면 그 가중치를 빼고 남은 가중치로 나눠 0~1을 유지한다.
SCORING_RULE = {
    "discussed_in_minutes": 0.30,
    "recent_discussion": 0.15,
    "basic_level_ordinances": 0.25,
    "regional_need": 0.15,
    "issue_discussion_probability": 0.15,
}
USAGE_POLICY = {
    "scope": "reference_only",
    "included_in_issue_detection": False,
    "included_in_forecast": False,
    "meeting_text_sent_externally": False,
    "sent_externally": "check_keyword를 통과한 짧은 정책어와 KOSIS 통계표 ID만",
}
LIMITATIONS = [
    "조례·의안은 '이름'에 정책어가 들어간 것만 잡히므로 이름이 다른 관련 조례는 누락될 수 있음",
    "후속 점수는 미리 고정한 가중합 규칙이며 학습된 확률이 아님; 인과관계를 의미하지 않음",
    "회의록 관측은 제공된 2024~2025 광역의회 회의록 범위에 한정됨",
    "국회 의안은 지역 구분이 없어 점수에 넣지 않고 설명용 참고 근거로만 씀",
    "KOSIS 지역 정책 수요는 '필요가 크다'는 간접 신호이며, 자동 선택된 통계표는 확인이 필요함",
]


def parse_day(value: Any) -> date | None:
    try:
        return datetime.strptime(str(value or "")[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def months_before(day: date, months: int) -> date:
    month_index = day.year * 12 + day.month - 1 - months
    return date(month_index // 12, month_index % 12 + 1, 1)


def compact(text: str) -> str:
    return "".join(str(text or "").split())


def relevant(name: str, terms: list[str] | None) -> bool:
    """조례·의안 이름에 이슈어 중 하나가 들어 있는가 (띄어쓰기 무시). 넓은 검색어로 걸린 무관한 조례를 점수에서 뺀다"""
    if not terms:
        return True
    target = compact(name)
    return any(len(compact(term)) >= 2 and compact(term) in target for term in terms)


def external_keywords(
    issue: dict[str, Any], planned: dict[str, str] | None, limit: int,
) -> tuple[list[str], list[dict[str, str]]]:
    """조례·의안 '이름' 검색에 맞는 짧은 정책어. check_keyword를 통과한 것만 사용"""
    candidates: list[str] = []
    if planned and planned.get("external_keywords"):
        candidates += planned["external_keywords"].split("|")
    else:
        candidates += [str(alias) for alias in issue.get("aliases") or []]
        name = str(issue.get("issue_name") or "")
        candidates += name.split("·") + [name]
    accepted: list[str] = []
    rejected: list[dict[str, str]] = []
    for raw in candidates:
        words = [word for word in raw.split() if word not in GENERIC_WORDS]
        keyword = " ".join(words).strip()
        if not keyword or keyword in accepted:
            continue
        try:
            accepted.append(EXT.check_keyword(keyword))
        except EXT.PolicyError as error:
            rejected.append({"keyword": keyword[:40], "reason": str(error)})
        if len(accepted) >= limit:
            break
    return accepted, rejected


def propagation_from_search(rows: list[dict[str, Any]], minutes_cutoff: date) -> list[dict[str, Any]]:
    """로컬 검색 결과(metadata.council·meeting_date)를 의회별 관측으로 집계 (외부 전송 없음)"""
    dates: dict[str, list[date]] = defaultdict(list)
    for row in rows:
        metadata = row.get("metadata") or {}
        council = EXT.normalize_council(str(metadata.get("council") or ""))
        day = parse_day(metadata.get("meeting_date"))
        if council and day and day <= minutes_cutoff:
            dates[council].append(day)
    return [
        {
            "council": council,
            "first_observation_date": min(days).isoformat(),
            "last_observation_date": max(days).isoformat(),
            "mention_count": len(days),
            "stage": "search_based_provisional",
        }
        for council, days in sorted(dates.items(), key=lambda item: min(item[1]))
    ]


def collect_external(keywords: list[str]) -> tuple[list[dict], list[dict], list[dict]]:
    """키워드별 조회 결과를 law_id·의안명 기준으로 합친다 (시점 필터 전 전체 목록)"""
    ordinances: dict[str, dict[str, Any]] = {}
    bills: dict[tuple[str, str], dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    for keyword in keywords:
        if EXT.key_available("law_ordin"):
            result = EXT.safe_call(EXT.fetch_local_ordinances, keyword)
            if "error" in result:
                errors.append({"source": "law_ordin", "keyword": keyword, "error": result["error"]})
            for item in result.get("items") or []:
                if item.get("kind") and item["kind"] not in ORDINANCE_KINDS:
                    continue
                identifier = item.get("law_id") or f"{item.get('org')}|{item.get('name')}"
                ordinances.setdefault(identifier, {**item, "matched_keywords": []})
                ordinances[identifier]["matched_keywords"].append(keyword)
        if EXT.key_available("assembly_bills"):
            result = EXT.safe_call(EXT.fetch_national_bills, keyword)
            if "error" in result:
                errors.append({"source": "assembly_bills", "keyword": keyword, "error": result["error"]})
            for item in result.get("items") or []:
                bills.setdefault((str(item.get("bill")), str(item.get("date"))), item)
    return list(ordinances.values()), list(bills.values()), errors


def partition_ordinances(
    ordinances: list[dict], as_of: date, mode: str, terms: list[str] | None,
) -> dict[str, Any]:
    """모드별 시점 규칙을 한 곳에서 적용한다.

    목록 API 공포일은 현행 버전 기준이다. backtest에서는 기준일 이후 개정본의 기준일 이전 존재 여부를 알 수 없어
    신호·사후점검 모두에서 제외한다. live에서는 조회일까지의 현행 조례를 그대로 쓴다."""
    visible, after_enacted, unrelated = [], [], []
    uncertain = 0
    for item in ordinances:
        day = parse_day(item.get("promulgated"))
        if day is None or not item.get("council"):
            continue
        if not relevant(item.get("name"), terms):
            unrelated.append(item)
            continue
        if day <= as_of:
            visible.append(item)
        elif mode == "backtest" and item.get("type") == "제정":
            after_enacted.append(item)
        elif mode == "backtest":
            uncertain += 1
    return {"visible": visible, "after_enacted": after_enacted,
            "uncertain_excluded": uncertain, "unrelated": unrelated}


def match_indicator_ids(terms: list[str], limit: int = 2) -> list[str]:
    """이슈어 → KOSIS 지표 id (config/kosis_indicators.json 키워드 비교, 로컬 처리).
    여러 이슈어에 공통으로 걸린 지표를 앞에 둔다 (넓은 별칭 하나에만 걸린 지표가 앞서지 않게)"""
    import regional_stats as RS  # 순환 import 방지

    hits: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    for term in terms:
        for row in RS.match_indicators(term):
            hits[row["id"]] = hits.get(row["id"], 0) + 1
            first_seen.setdefault(row["id"], len(first_seen))
    return sorted(hits, key=lambda key: (-hits[key], first_seen[key]))[:limit]


def regional_need(indicator_ids: list[str], as_of: date, mode: str) -> dict[str, Any]:
    """KOSIS 지표별 17개 시도 수요 점수(0~1)를 의회별 평균으로. 키·지표가 없으면 빈 결과"""
    if not indicator_ids or not EXT.key_available("kosis"):
        return {"by_council": {}, "indicators": [], "errors": []}
    import regional_stats as RS  # 순환 import 방지

    per_council: dict[str, list[float]] = defaultdict(list)
    used, errors = [], []
    for indicator_id in dict.fromkeys(indicator_ids):
        result = EXT.safe_call(RS.get_regional_indicator, indicator_id, as_of.isoformat(), mode)
        if result.get("error"):
            errors.append({"source": "kosis", "keyword": indicator_id, "error": result["error"]})
            continue
        for council, score in result["need_score"].items():
            for member in EXT.members_of(council):   # 통합 의회 → 전신 의회 양쪽
                per_council[member].append(score)
        used.append({key: result.get(key) for key in (
            "indicator", "name", "need", "period", "item", "unit", "attribution")}
            | {"pinned": (result.get("table") or {}).get("pinned")})
    return {
        "by_council": {council: round(sum(v) / len(v), 3) for council, v in per_council.items()},
        "indicators": used,
        "errors": errors,
    }


def council_flow(
    propagation: list[dict[str, Any]], ordinances: list[dict], minutes_cutoff: date, recent_months: int,
) -> list[dict[str, Any]]:
    observed = {}
    for row in propagation:
        council = EXT.normalize_council(row.get("council"))
        if council:
            observed[council] = row
    recent_from = months_before(minutes_cutoff, recent_months)
    flows = []
    for council in EXT.REGION:
        regional = sorted(
            (item for item in ordinances if item["council"] == council and item["level"] == "광역"),
            key=lambda item: item["promulgated"],
        )
        basic = sorted(
            (item for item in ordinances if item["council"] == council and item["level"] == "기초"),
            key=lambda item: item["promulgated"],
        )
        minutes = observed.get(council)
        first_minutes = parse_day((minutes or {}).get("first_observation_date"))
        last_minutes = parse_day((minutes or {}).get("last_observation_date"))
        first_regional = parse_day(regional[0]["promulgated"]) if regional else None
        after_minutes = [item for item in regional if parse_day(item["promulgated"]) > minutes_cutoff]
        if minutes and first_regional:
            stage = (
                "논의_후_광역조례화" if first_regional >= first_minutes
                else "광역조례_선행_후_논의"
            )
        elif minutes:
            stage = "논의_진행_조례_미확인"
        elif regional:
            stage = "회의록_미관측_광역조례_존재"
        elif basic:
            stage = "기초조례_선행_광역논의_미관측"
        else:
            stage = "미관측"
        flows.append({
            "council": council,
            "stage": stage,
            "minutes_first_observation": first_minutes.isoformat() if first_minutes else None,
            "minutes_last_observation": last_minutes.isoformat() if last_minutes else None,
            "minutes_mention_count": int((minutes or {}).get("mention_count") or 0),
            "minutes_propagation_stage": (minutes or {}).get("stage"),
            "recent_discussion": bool(last_minutes and last_minutes >= recent_from),
            "regional_ordinance_count": len(regional),
            "first_regional_ordinance": first_regional.isoformat() if first_regional else None,
            "regional_after_minutes_cutoff": len(after_minutes),
            "lag_days_minutes_to_ordinance": (
                (first_regional - first_minutes).days
                if first_minutes and first_regional and first_regional >= first_minutes else None
            ),
            "basic_ordinance_count": len(basic),
            "first_basic_ordinance": basic[0]["promulgated"] if basic else None,
            "regional_ordinances": [
                {key: item.get(key) for key in ("name", "org", "type", "promulgated", "law_id")}
                for item in regional[:5]
            ],
        })
    return flows


def weighted_score(signals: dict[str, float | None]) -> float:
    available = {name: value for name, value in signals.items() if value is not None}
    total = sum(SCORING_RULE[name] for name in available)
    if not total:
        return 0.0
    return sum(SCORING_RULE[name] * value for name, value in available.items()) / total


def outlook(
    flows: list[dict[str, Any]], need: dict[str, float], issue_probability: float | None, top_k: int,
) -> dict[str, Any]:
    candidates = []
    for flow in flows:
        if flow["regional_ordinance_count"] > 0:
            continue  # 이미 광역 조례화 → 신규 후속 후보가 아니라 개정·이행 점검 대상
        signals = {
            "discussed_in_minutes": 1.0 if flow["minutes_mention_count"] > 0 else 0.0,
            "recent_discussion": 1.0 if flow["recent_discussion"] else 0.0,
            "basic_level_ordinances": min(1.0, flow["basic_ordinance_count"] / 5.0),
            "regional_need": need.get(flow["council"]) if need else None,
            "issue_discussion_probability": (
                float(issue_probability) if issue_probability is not None else None
            ),
        }
        expected = (
            "광역_조례화" if flow["minutes_mention_count"] > 0
            else "광역의회_논의_진입" if flow["basic_ordinance_count"] > 0
            else "관망"
        )
        candidates.append({
            "council": flow["council"],
            "expected_next_step": expected,
            "followup_score": round(weighted_score(signals), 4),
            "signals": signals,
        })
    candidates.sort(key=lambda row: (-row["followup_score"], row["council"]))
    for rank, row in enumerate(candidates, start=1):
        row["rank"] = rank
    lags = [flow["lag_days_minutes_to_ordinance"] for flow in flows
            if flow["lag_days_minutes_to_ordinance"] is not None]
    return {
        "top_candidates": candidates[:top_k],
        "all_candidates": candidates,
        "already_institutionalized": [flow["council"] for flow in flows if flow["regional_ordinance_count"] > 0],
        "followups_after_minutes_cutoff": [
            flow["council"] for flow in flows if flow["regional_after_minutes_cutoff"] > 0
        ],
        "median_lag_days_minutes_to_ordinance": statistics.median(lags) if lags else None,
        "scoring_rule": SCORING_RULE,
    }


def post_cutoff_check(candidates: list[dict], after_enacted: list[dict], top_k: int) -> dict[str, Any]:
    """backtest: 기준일 이후 실제 광역 '제정' 조례와 후보 순위를 비교 (모델 학습에는 쓰지 않음)"""
    enacted = sorted({item["council"] for item in after_enacted if item["level"] == "광역"})
    if not enacted:
        return {"status": "no_new_regional_enactment", "enacted_councils": []}
    top = [row["council"] for row in candidates[:top_k]]
    hits = sorted(set(top) & set(enacted))
    return {
        "status": "complete",
        "checked_until": max(item["promulgated"] for item in after_enacted),
        "enacted_councils": enacted,
        "top_k": top_k,
        "top_k_hits": hits,
        "precision_at_k": round(len(hits) / max(1, len(top)), 4),
        "recall_at_k": round(len(hits) / len(enacted), 4),
    }


def prediction_integration(
    forecast: dict[str, Any], counts: dict[str, Any],
    regional: dict[str, Any], view: dict[str, Any], mode: str,
) -> dict[str, Any]:
    """공식 회의록 예측과 외부 공공데이터를 설명 가능한 2층 결과로 묶는다.

    외부정보가 공식 확률·언급량을 다시 계산한 것처럼 보이지 않도록 숫자는
    변경하지 않는다. 조례화·국회 모멘텀·지역 수요는 후속 단계 후보와
    조건부 해석을 보강하는 참고 신호로만 사용한다.
    """
    probability = forecast.get("discussion_active_probability")
    mentions = forecast.get("predicted_mention_count")
    target = forecast.get("target_month") or "다음 달"
    institutionalized = list(view.get("already_institutionalized") or [])
    top = list(view.get("top_candidates") or [])
    indicators = list(regional.get("indicators") or [])
    bills = int(counts.get("national_bills_last_12m") or 0)
    ordinances = int(counts.get("ordinances_visible") or 0)
    lag = view.get("median_lag_days_minutes_to_ordinance")

    if probability is None:
        base_observation = f"{target} 공식 확률 없음"
        base_outlook = "회의록 예측 근거가 부족합니다."
    else:
        base_observation = (
            f"{target} 논의 지속 {float(probability):.1%}"
            + (f" · 예상 언급 {int(mentions)}건" if mentions is not None else "")
        )
        if float(probability) >= 0.70:
            base_outlook = "회의록 시계열은 후속 논의 지속 가능성이 높은 상태입니다."
        elif float(probability) >= 0.45:
            base_outlook = "회의록 시계열은 제한적 지속 또는 기존 의회 중심 논의를 시사합니다."
        else:
            base_outlook = "회의록 시계열만 보면 단기 논의 약화 가능성이 큽니다."

    indicator_names = [
        str(row.get("name") or row.get("indicator_id")) for row in indicators[:2]
    ]
    evidence_table = [
        {
            "source": "회의록 시계열 모델",
            "observation": base_observation,
            "reflection": "공식 논의 지속 확률·예상 언급량",
            "model_input": "사용",
        },
        {
            "source": "법제처 자치법규",
            "observation": (
                f"관련 조례 {ordinances}건 · 광역 조례화 {len(institutionalized)}곳"
                + (f" · 중앙 소요일 {int(lag)}일" if lag is not None else "")
            ),
            "reflection": "후속 조례화 단계와 지역 후보 순위 보강",
            "model_input": "참고 신호",
        },
        {
            "source": "열린국회정보",
            "observation": f"최근 12개월 관련 의안 {bills}건",
            "reflection": "전국 단위 정책 모멘텀 해석",
            "model_input": "참고 신호",
        },
        {
            "source": "KOSIS",
            "observation": (
                f"연결 지표 {len(indicators)}개"
                + (f" · {', '.join(indicator_names)}" if indicator_names else "")
            ),
            "reflection": "지역별 정책 수요와 후속 논의 후보 보강",
            "model_input": "참고 신호",
        },
    ]

    supporting = sum((bool(institutionalized), bills > 0, bool(indicators)))
    step_labels = {
        "광역_조례화": "광역 조례화",
        "광역의회_논의_진입": "광역의회 논의 진입",
        "관망": "관망",
    }
    candidate_text = (
        f" 후속 단계 1순위는 {top[0]['council']}"
        f"({step_labels.get(top[0]['expected_next_step'], top[0]['expected_next_step'])})입니다."
        if top else ""
    )
    if mode == "live":
        integrated = (
            f"{base_outlook} 최신 외부정보는 공식 대회 예측과 분리된 사후 참고자료입니다."
            f"{candidate_text}"
        )
    elif supporting >= 2:
        integrated = (
            f"{base_outlook} 조례·국회·지역수요 가운데 {supporting}개 외부 신호가 확인되어 "
            f"후속 논의 또는 제도화 가능성을 보강합니다.{candidate_text}"
        )
    elif supporting == 1:
        integrated = (
            f"{base_outlook} 외부 보강 신호는 1개로 제한적이므로 방향 해석에만 참고합니다."
            f"{candidate_text}"
        )
    else:
        integrated = (
            f"{base_outlook} 확인된 외부 보강 신호가 부족해 회의록 모델 결과를 우선합니다."
            f"{candidate_text}"
        )
    return {
        "official_forecast_unchanged": True,
        "numeric_forecast_adjusted": False,
        "external_role": "후속 단계 후보 순위와 조건부 해석 보강",
        "evidence_table": evidence_table,
        "integrated_outlook": integrated,
        "caution": "외부 참고 점수는 확률 또는 인과적 전파 판정이 아님",
    }


def build_followup(
    issue: dict[str, Any],
    propagation: list[dict[str, Any]],
    keywords: list[str],
    as_of: date,
    mode: str = "live",
    minutes_cutoff: date | None = None,
    issue_forecast: dict[str, Any] | None = None,
    relevance_terms: list[str] | None = None,
    indicator_ids: list[str] | None = None,
    recent_months: int = 6,
    momentum_months: int = 12,
    top_k: int = 5,
    rejected_keywords: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """회의록 관측 + 외부 공공데이터 → council_external_followup_v2 결과 한 건"""
    if mode not in MODES:
        raise ValueError(f"mode는 {MODES} 중 하나여야 합니다: {mode!r}")
    minutes_cutoff = minutes_cutoff or parse_day(DEFAULT_MINUTES_CUTOFF)
    forecast = issue_forecast or {}
    terms = relevance_terms if relevance_terms is not None else keywords
    ordinances, bills, errors = collect_external(keywords)
    parts = partition_ordinances(ordinances, as_of, mode, terms)
    bills_visible = [
        item for item in bills
        if (day := parse_day(item.get("date"))) and day <= as_of and relevant(item.get("bill"), terms)
    ]
    momentum_from = months_before(as_of, momentum_months)
    need = regional_need(indicator_ids or [], as_of, mode)
    errors += need["errors"]
    flows = council_flow(propagation, parts["visible"], minutes_cutoff, recent_months)
    # live에서는 회의록 기준 모델 확률(기준일 직후 1개월 대상)이 조회일 이후 전망에 맞지 않아 점수에서 뺀다
    probability = forecast.get("discussion_active_probability") if mode == "backtest" else None
    view = outlook(flows, need["by_council"], probability, top_k)
    external_counts = {
        "ordinances_visible": len(parts["visible"]),
        "ordinances_unrelated_excluded": len(parts["unrelated"]),
        "enacted_after_as_of": len(parts["after_enacted"]),
        "amended_after_as_of_uncertain_excluded": parts["uncertain_excluded"],
        "national_bills_visible": len(bills_visible),
        "national_bills_last_12m": sum(
            1 for item in bills_visible if parse_day(item.get("date")) >= momentum_from
        ),
    }
    regional_result = {
        "indicators": need["indicators"],
        "by_council": need["by_council"],
    }
    used = [source for source in EXT.DATA_SOURCES if EXT.key_available(source)]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if used and keywords and not errors else "partial",
        "issue": issue,
        "mode": mode,
        "as_of": as_of.isoformat(),
        "minutes_cutoff": minutes_cutoff.isoformat(),
        "outlook_horizon": (
            f"{as_of.isoformat()} 이후" if mode == "live" else f"{as_of.isoformat()} 이후 (사후 점검 대상)"
        ),
        "keywords": keywords,
        "relevance_terms": terms,
        "rejected_keywords": rejected_keywords or [],
        "usage_policy": USAGE_POLICY,
        "model_forecast_reference": {
            "target_month": forecast.get("target_month"),
            "discussion_active_probability": forecast.get("discussion_active_probability"),
            "predicted_mention_count": forecast.get("predicted_mention_count"),
            "used_in_score": probability is not None,
        },
        "external_counts": external_counts,
        "followup_flow": flows,
        "regional_need": regional_result,
        "outlook": {key: value for key, value in view.items() if key != "all_candidates"},
        "prediction_integration": prediction_integration(
            forecast, external_counts, regional_result, view, mode,
        ),
        "post_cutoff_check": (
            post_cutoff_check(view["all_candidates"], parts["after_enacted"], top_k)
            if mode == "backtest" else {"status": "not_applicable_live_mode"}
        ),
        "national_bills": sorted(bills_visible, key=lambda row: row.get("date") or "", reverse=True)[:10],
        "unrelated_ordinances": [
            {key: item.get(key) for key in ("name", "org", "promulgated")} for item in parts["unrelated"][:10]
        ],
        "errors": errors,
        "data_sources": EXT.sources_used(used),
        "limitations": LIMITATIONS,
    }


def compact_for_llm(result: dict[str, Any], max_flows: int = 8, councils: list[str] | None = None) -> dict[str, Any]:
    """LLM 컨텍스트용 요약: 미관측 의회와 긴 목록은 빼고 계산된 값만 전달 (councils를 주면 그 의회를 우선)"""
    rows = [row for row in result.get("followup_flow") or [] if row.get("stage") != "미관측"]
    if councils:
        rows = [row for row in result.get("followup_flow") or [] if row["council"] in councils] + [
            row for row in rows if row["council"] not in councils
        ]
    flows = [
        {key: row[key] for key in (
            "council", "stage", "minutes_first_observation", "minutes_mention_count",
            "first_regional_ordinance", "regional_after_minutes_cutoff",
            "lag_days_minutes_to_ordinance", "basic_ordinance_count",
        )}
        # 질문한 지역은 조례 이름까지 넘겨 LLM이 '관련 조례'처럼 뭉뚱그리지 않게
        | ({"regional_ordinances": row.get("regional_ordinances") or []} if councils and row["council"] in councils else {})
        for row in rows
    ][:max_flows]
    view = result.get("outlook") or {}
    return {
        "issue": (result.get("issue") or {}).get("issue_name"),
        "status": result.get("status"),
        "mode": result.get("mode"),
        "as_of": result.get("as_of"),
        "minutes_cutoff": result.get("minutes_cutoff"),
        "keywords": result.get("keywords"),
        "followup_flow": flows,
        "already_institutionalized": view.get("already_institutionalized"),
        "followups_after_minutes_cutoff": view.get("followups_after_minutes_cutoff"),
        "top_candidates": [
            {key: row[key] for key in ("rank", "council", "expected_next_step", "followup_score")}
            for row in view.get("top_candidates") or []
        ],
        "median_lag_days_minutes_to_ordinance": view.get("median_lag_days_minutes_to_ordinance"),
        "regional_need_indicators": [
            {key: row.get(key) for key in ("name", "period", "pinned")}
            for row in (result.get("regional_need") or {}).get("indicators") or []
        ],
        "national_bills_last_12m": (result.get("external_counts") or {}).get("national_bills_last_12m"),
        "recent_national_bills": [
            {key: row.get(key) for key in ("bill", "date", "result")}
            for row in (result.get("national_bills") or [])[:3]
        ],
        "post_cutoff_check": result.get("post_cutoff_check"),
        "prediction_integration": {
            key: (result.get("prediction_integration") or {}).get(key)
            for key in (
                "official_forecast_unchanged", "external_role",
                "integrated_outlook", "caution",
            )
        },
        "errors": [row.get("error") for row in result.get("errors") or []][:3],
        "attribution": [row.get("attribution") for row in result.get("data_sources") or []],
        "note": "점수는 미리 고정한 규칙 기반 참고값이며 인과관계·확률이 아님. 회의록 원문은 외부로 전송되지 않음",
    }
