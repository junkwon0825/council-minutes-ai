#!/usr/bin/env python3
"""자유 질문을 '해석 → 검증·보정 → 의도별 실행 → 답변 → 답변 검증'으로 처리하는 질문 파이프라인.

LLM 호출은 두 번으로 고정한다.
  ① 질문 해석: Ollama format(JSON 스키마)으로 의도·주제·지역·모드만 채운다. 실패하면 규칙 기반 해석으로 대체
  ④ 답변 생성: 코드가 계산한 결과와 회의록 근거만 설명한다
그 사이 ② 검증·보정(대표 이슈 의미 연결, 검색어의 로컬 회의록 등장 확인, KOSIS 지표 연결)과
③ 실행(의도별로 정해진 순서)은 코드가 하고, ⑤ 답변 속 숫자·의회명을 계산 결과와 대조한다.
질문 원문은 외부로 나가지 않는다. 외부 API에는 ②에서 검증하고 check_keyword를 통과한 정책어만 보낸다.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import external_followup as FU  # noqa: E402
import external_policy_sources as EXT  # noqa: E402
import regional_stats as RS  # noqa: E402


INTENTS = {
    "propagation": "이슈가 어느 의회에서 처음 나왔고 어떻게 퍼졌는지",
    "followup_outlook": "타 지방의회의 후속 논의·조례화와 앞으로의 전망",
    "region_compare": "특정 지역끼리 비교",
    "evidence": "회의록 원문 근거·발언 내용",
    "statistics": "지역별 공공 통계 지표",
    "national_bills": "국회 법률안 동향",
    "explain": "개념·분석 방법 설명",
    "other": "그 밖의 질문",
}
COUNCILS = list(EXT.REGION)
REGION_FORMS: dict[str, str] = {}
for _council, _forms in EXT.REGION.items():
    REGION_FORMS[_council] = _council
    REGION_FORMS[_council.removesuffix("의회")] = _council
    for _form in _forms:
        REGION_FORMS[_form] = _council
for _short, _council in RS.SHORT.items():
    REGION_FORMS[_short] = _council
    REGION_FORMS[_short + ("도" if _council.endswith("도의회") else "시")] = _council   # 경북도, 부산시 …
REGION_FORMS.update({"전라북도": "전북특별자치도의회", "강원도": "강원특별자치도의회", "제주도": "제주특별자치도의회"})

PARSE_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": list(INTENTS)},
        "topic": {"type": "string", "description": "질문의 정책 주제 (짧은 명사구)"},
        "topic_terms": {"type": "array", "items": {"type": "string"}, "maxItems": 5,
                        "description": "주제의 동의어·공식 용어·조례에 쓰일 표현 (각 2~15자)"},
        "regions": {"type": "array", "items": {"type": "string", "enum": COUNCILS}},
        "mode": {"type": "string", "enum": ["live", "backtest"]},
        "needs_clarification": {"type": "boolean"},
        "clarification_question": {"type": "string"},
    },
    "required": ["intent", "topic", "topic_terms", "regions", "mode", "needs_clarification"],
}
INTENT_RULES = [
    ("evidence", ("원문", "발언", "근거", "누가 말", "회의록 내용", "언급한 내용")),
    ("national_bills", ("국회", "법안", "법률안", "발의")),
    ("region_compare", ("비교", "차이", "보다")),
    ("statistics", ("통계", "출산율", "수치", "지표", "인구수", "비율")),
    ("followup_outlook", ("후속", "조례", "전망", "예측", "다음", "앞으로", "가능성", "이어질", "따라")),
    ("propagation", ("처음", "최초", "확산", "퍼졌", "흐름", "어디서 먼저", "어느 의회")),
    ("explain", ("무슨 뜻", "뭐야", "설명", "어떻게 계산", "방법")),
]
STOP_WORDS = {"관련", "정책", "논의", "의회", "지방의회", "조례", "어디", "어느", "다음", "지금", "현재", "앞으로",
              "가능성", "어떻게", "무엇", "뭐야", "알려줘", "있어", "있나", "할까", "될까", "높아", "그럼", "그리고",
              "어디서", "먼저", "처음", "최초", "비교", "차이", "후속", "전망", "예측", "통계", "수치", "지표",
              "국회", "법안", "법률안", "발의", "원문", "발언", "근거", "확산", "흐름", "회의록", "내용"}
PLANS = {
    "propagation": ["issue_summary", "minutes_evidence"],
    "followup_outlook": ["issue_summary", "followup", "minutes_evidence"],
    "region_compare": ["issue_summary", "followup", "statistics"],
    "evidence": ["minutes_evidence"],
    "statistics": ["statistics"],
    "national_bills": ["national_bills"],
    "explain": ["issue_summary"],
    "other": ["issue_summary", "minutes_evidence"],
}


PARTICLES = {"에서", "에게", "에서도", "으로", "까지", "부터", "하고", "이랑", "이나"}
VERB_ENDING = re.compile(r"(야|요|까|줘|해|했|할|된|될|길|온|아|어|게|니|냐)$")


def _strip_particles(token: str) -> str:
    return re.sub(r"(에서도|에서|에게|으로|까지|부터|이랑|하고|은|는|이|가|을|를|에|의|도|로|과|와|랑)$", "", token)


def _topic_tokens(text: str) -> list[str]:
    """규칙 기반 주제어: 조사·서술어 꼴을 뺀 2자 이상 명사형 토큰"""
    tokens = []
    for raw in re.findall(r"[가-힣A-Za-z0-9·]+", text):
        if raw in PARTICLES or VERB_ENDING.search(raw):
            continue
        token = _strip_particles(raw)
        if len(token) >= 2 and token not in STOP_WORDS and token not in PARTICLES:
            tokens.append(token)
    return tokens


def find_regions(text: str) -> list[str]:
    found = []
    for form in sorted(REGION_FORMS, key=len, reverse=True):
        if form and form in text:
            council = REGION_FORMS[form]
            if council not in found:
                found.append(council)
            text = text.replace(form, " ")
    return found


def heuristic_parse(question: str, previous: dict[str, Any] | None) -> dict[str, Any]:
    """LLM 해석이 실패했을 때의 규칙 기반 해석 (외부 호출 없음)"""
    intent = next((name for name, words in INTENT_RULES if any(word in question for word in words)), "other")
    regions = find_regions(question)
    stripped = question
    for form in sorted(REGION_FORMS, key=len, reverse=True):
        stripped = stripped.replace(form, " ")
    terms = _topic_tokens(stripped)
    topic = " ".join(terms[:2])
    if not topic and previous:   # "그럼 충남은?" — 직전 주제·의도를 이어받는다
        topic = previous.get("topic") or ""
        terms = list(previous.get("topic_terms") or [])
        if intent == "other":
            intent = previous.get("intent") or intent
    requested_live = any(word in question for word in ("현재", "최신", "오늘", "실시간", "live"))
    default_mode = os.environ.get("COUNCIL_FOLLOWUP_MODE", "backtest")
    mode = "live" if requested_live else (previous or {}).get("mode", default_mode)
    return {"intent": intent, "topic": topic, "topic_terms": terms[:5], "regions": regions,
            "mode": mode, "needs_clarification": False,
            "clarification_question": ""}


def normalize_slots(raw: dict[str, Any], question: str, previous: dict[str, Any] | None) -> dict[str, Any]:
    """LLM 출력 검증: 허용 값만 남기고 비면 규칙 기반 값·직전 대화로 채운다"""
    fallback = heuristic_parse(question, previous)
    intent = raw.get("intent") if raw.get("intent") in INTENTS else fallback["intent"]
    regions = [REGION_FORMS.get(str(value), str(value)) for value in raw.get("regions") or []]
    regions = [value for value in dict.fromkeys(regions) if value in COUNCILS] or fallback["regions"]
    topic = str(raw.get("topic") or "").strip()[:40] or fallback["topic"]
    terms = [str(value).strip() for value in raw.get("topic_terms") or [] if str(value).strip()]
    if not topic and previous:  # "그럼 경북은?" 같은 후속 질문은 직전 주제를 이어받는다
        topic = previous.get("topic") or ""
        terms = terms or list(previous.get("topic_terms") or [])
    mode = raw.get("mode") if raw.get("mode") in FU.MODES else fallback["mode"]
    # LLM이 임의로 최신 정보를 선택하지 못하게 한다. 사용자가 최신·현재를
    # 명시했거나 직전 대화가 live였을 때만 live를 허용한다.
    if mode == "live" and fallback["mode"] != "live":
        mode = "backtest"
    return {
        "intent": intent, "topic": topic, "topic_terms": list(dict.fromkeys([topic, *terms]))[:6],
        "regions": regions, "mode": mode,
        "needs_clarification": bool(raw.get("needs_clarification")) and not topic,
        "clarification_question": str(raw.get("clarification_question") or "")[:200],
    }


def _cosine_top(query: Any, matrix: Any) -> list[tuple[int, float]]:
    import numpy as np

    scores = matrix @ np.asarray(query, dtype=np.float32)
    order = np.argsort(-scores)
    return [(int(index), float(scores[index])) for index in order]


@dataclass
class QuestionPipeline:
    """LocalCouncilAgent가 쓰는 질문 처리기. llm(messages, schema, num_predict) -> str"""

    issue_payloads: list[dict[str, Any]]
    tools: Any                                    # agent.followup_tools.FollowupTools
    search_service: Any | None = None
    route_threshold: float = 0.55
    route_margin: float = 0.03
    _issue_matrix: Any = field(default=None, repr=False)
    _indicator_matrix: Any = field(default=None, repr=False)

    # ------------------------------------------------------------------ ① 질문 해석
    def parse(self, question: str, previous: dict[str, Any] | None, llm: Callable | None) -> tuple[dict, str, str | None]:
        if llm is None:
            return normalize_slots({}, question, previous), "heuristic", None
        system = (
            "너는 지방의회 정책 질문을 구조화하는 해석기다. 답하지 말고 JSON만 출력한다. "
            "intent는 다음 중 하나: " + "; ".join(f"{key}={value}" for key, value in INTENTS.items()) + ". "
            "topic은 정책 주제의 짧은 명사구, topic_terms는 그 주제의 동의어·공식 용어·조례 이름에 쓰일 표현이다. "
            "regions는 질문에 나온 광역의회만 정식 명칭으로 쓴다(경북→경상북도의회). "
            "질문이 '현재', '최신', '오늘', '실시간'을 명시하면 mode=live, "
            "그 밖에는 대회 데이터 기준을 지키도록 mode=backtest. "
            "직전 해석이 주어지고 질문이 짧은 후속 질문이면 바뀐 항목만 바꾸고 나머지는 이어받는다."
        )
        user = json.dumps({"question": question, "previous": previous or None}, ensure_ascii=False)
        try:
            raw = json.loads(llm([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                 PARSE_SCHEMA, 300))
            return normalize_slots(raw if isinstance(raw, dict) else {}, question, previous), "llm", None
        except Exception as error:  # LLM 미연결·시간 초과·형식 오류 → 규칙 기반
            return normalize_slots({}, question, previous), "heuristic", f"질문 해석 대체: {type(error).__name__}: {error}"

    # ------------------------------------------------------------------ ② 검증·보정
    def _encoder(self):
        return getattr(self.search_service, "encoder", None) if self.search_service is not None else None

    def route_issue(self, slots: dict[str, Any], question: str) -> dict[str, Any]:
        """주제 → 대표 이슈. 임베딩 의미 유사도 우선, 없으면 문자열 매칭"""
        topic = slots["topic"] or question
        encoder = self._encoder()
        if encoder is not None and self.issue_payloads:
            import numpy as np

            if self._issue_matrix is None:
                texts = [
                    f"{(p.get('issue') or {}).get('issue_name')}: {', '.join((p.get('issue') or {}).get('aliases') or [])}"
                    for p in self.issue_payloads
                ]
                self._issue_matrix = np.asarray(encoder.encode(texts), dtype=np.float32)
            ranked = _cosine_top(encoder.encode([topic])[0], self._issue_matrix)
            (best, score), second = ranked[0], (ranked[1][1] if len(ranked) > 1 else 0.0)
            candidates = [
                {"issue_name": (self.issue_payloads[i].get("issue") or {}).get("issue_name"), "score": round(s, 3)}
                for i, s in ranked[:3]
            ]
            if score >= self.route_threshold:
                ambiguous = score - second < self.route_margin and second >= self.route_threshold
                return {"payload": None if ambiguous else self.issue_payloads[best], "method": "embedding",
                        "score": score, "candidates": candidates, "ambiguous": ambiguous}
        else:
            candidates = []
        # 의미 유사도가 기준 미만이거나 인코더가 없으면 이슈명·별칭 문자열 매칭으로 한 번 더
        from agent.local_issue_agent import route_fixed_issue  # 순환 import 방지

        payload, score = route_fixed_issue(f"{topic} {question}", self.issue_payloads)
        return {"payload": payload, "method": "string", "score": score, "candidates": candidates}

    def verify_terms(self, terms: list[str], regions: list[str]) -> list[dict[str, Any]]:
        """검색어 후보마다 check_keyword + 로컬 회의록에 실제로 나오는지 확인 (외부 전송 없음)"""
        checked = []
        for term in terms:
            try:
                keyword = EXT.check_keyword(term)
            except EXT.PolicyError as error:
                checked.append({"term": term[:30], "status": "blocked", "reason": str(error)})
                continue
            if self.search_service is None:
                checked.append({"term": keyword, "status": "unverified_no_local_search", "hits": None})
                continue
            try:
                found = self.search_service.search(query=keyword, top_k=20, candidate_k=100, include_parent=False)
            except Exception as error:
                checked.append({"term": keyword, "status": "unverified_search_error", "reason": str(error)})
                continue
            target = FU.compact(keyword)
            hits = sum(
                1 for row in found.get("results") or []
                if target in FU.compact(row.get("text")) or target in FU.compact((row.get("metadata") or {}).get("agenda_title"))
            )
            checked.append({"term": keyword, "status": "verified" if hits else "not_in_minutes", "hits": hits})
        return checked

    def match_indicators(self, slots: dict[str, Any], limit: int = 2) -> list[str]:
        """주제 → KOSIS 지표: 키워드 비교 + (가능하면) 지표 설명과의 의미 유사도"""
        ids = FU.match_indicator_ids([slots["topic"], *slots["topic_terms"]], limit)
        encoder = self._encoder()
        if encoder is not None and len(ids) < limit and slots["topic"]:
            import numpy as np

            catalog = RS.catalog()
            if self._indicator_matrix is None:
                self._indicator_matrix = np.asarray(
                    encoder.encode([RS.indicator_text(ind) for ind in catalog]), dtype=np.float32)
            for index, score in _cosine_top(encoder.encode([slots["topic"]])[0], self._indicator_matrix):
                if score < 0.5 or len(ids) >= limit:
                    break
                if catalog[index]["id"] not in ids:
                    ids.append(catalog[index]["id"])
        return ids

    def ground(self, slots: dict[str, Any], question: str) -> dict[str, Any]:
        route = self.route_issue(slots, question)
        terms = self.verify_terms(slots["topic_terms"], slots["regions"])
        usable = [row["term"] for row in terms if row["status"] in {"verified", "unverified_no_local_search"}]
        if not usable:  # 회의록에 없는 표현뿐이면 검사를 통과한 주제어 하나만 쓴다
            usable = [row["term"] for row in terms if row["status"] != "blocked"][:1]
        return {
            "route": {key: value for key, value in route.items() if key != "payload"},
            "payload": route["payload"],
            "terms": terms,
            "external_terms": usable[:4],
            "indicators": self.match_indicators(slots),
        }

    # ------------------------------------------------------------------ ③ 의도별 실행
    def execute(self, slots: dict[str, Any], grounded: dict[str, Any], retrieve: Callable) -> dict[str, Any]:
        steps, facts, evidence = [], {}, []
        payload = grounded["payload"]
        mode, terms, indicators = slots["mode"], grounded["external_terms"], grounded["indicators"]
        sent: list[str] = []
        for step in PLANS[slots["intent"]]:
            try:
                if step == "issue_summary":
                    if payload:
                        facts["issue"] = issue_facts(payload, slots["regions"])
                        steps.append({"step": step, "status": "ok", "detail": facts["issue"]["issue_name"]})
                    else:
                        steps.append({"step": step, "status": "skipped", "detail": "연결된 대표 이슈 없음"})
                elif step == "minutes_evidence":
                    query = " ".join([slots["topic"] or "", *[r.removesuffix("의회") for r in slots["regions"]]]).strip()
                    rows, how = retrieve(query or slots["topic"])
                    if slots["regions"]:
                        rows = [r for r in rows if EXT.normalize_council((r.get("metadata") or {}).get("council")) in slots["regions"]] or rows
                    evidence = rows
                    steps.append({"step": step, "status": "ok" if rows else "empty", "detail": f"{how} {len(rows)}건"})
                elif step == "followup":
                    if payload:
                        out = self.tools.issue_followup(payload, mode, terms, indicators)
                    elif terms:
                        out = self.tools.topic_followup(slots["topic"], terms, mode, indicators)
                    else:
                        out = {"error": "외부 조회에 쓸 수 있는 검색어가 없음"}
                    if out.get("error"):
                        steps.append({"step": step, "status": "error", "detail": out["error"]})
                        continue
                    sent += out["sent_externally"]
                    facts["followup"] = FU.compact_for_llm(out["result"], councils=slots["regions"])
                    facts["followup"]["source_mode"] = out["mode_used"]
                    steps.append({"step": step, "status": "ok", "detail": out["mode_used"]})
                elif step == "statistics":
                    out = self.tools.regional_statistics(indicators, mode)
                    if out.get("error"):
                        steps.append({"step": step, "status": "error", "detail": out["error"]})
                        continue
                    sent += out["sent_externally"]
                    facts["statistics"] = [statistics_facts(row, slots["regions"]) for row in out["indicators"]]
                    steps.append({"step": step, "status": "ok", "detail": f"지표 {len(indicators)}개"})
                elif step == "national_bills":
                    out = (self.tools.national_bills(terms, mode) if terms
                           else {"error": "외부 조회에 쓸 수 있는 검색어가 없음"})
                    if out.get("error"):
                        steps.append({"step": step, "status": "error", "detail": out["error"]})
                        continue
                    sent += out["sent_externally"]
                    facts["national_bills"] = {key: out[key] for key in ("as_of", "count", "items", "attribution")}
                    steps.append({"step": step, "status": "ok", "detail": f"{out['count']}건"})
            except EXT.PolicyError as error:
                steps.append({"step": step, "status": "blocked", "detail": str(error)})
            except Exception as error:  # 한 단계 실패가 답변 전체를 멈추지 않게
                steps.append({"step": step, "status": "error", "detail": f"{type(error).__name__}: {error}"})
        return {"steps": steps, "facts": facts, "evidence": evidence, "sent_externally": list(dict.fromkeys(sent))}


# ---------------------------------------------------------------------- 결과 요약 (LLM 컨텍스트용)
def issue_facts(payload: dict[str, Any], regions: list[str]) -> dict[str, Any]:
    issue = payload.get("issue") or {}
    summary = payload.get("summary") or {}
    forecast = payload.get("multi_issue_forecast") or payload.get("forecast") or {}
    order = [
        {"council": row.get("council"), "first_observation_date": row.get("first_observation_date"),
         "mention_count": row.get("mention_count")}
        for row in payload.get("propagation") or []
    ]
    if regions:
        order = [row for row in order if EXT.normalize_council(row["council"]) in regions] + [
            row for row in order if EXT.normalize_council(row["council"]) not in regions]
    return {
        "issue_name": issue.get("issue_name"),
        "data_scope": "제공된 2024~2025 광역의회 회의록",
        "first_observation_date": summary.get("first_observation_date"),
        "first_observation_councils": summary.get("first_observation_councils"),
        "council_count": summary.get("council_count"),
        "observation_order": order[:10],
        "model_forecast": {key: forecast.get(key) for key in (
            "target_month", "discussion_active_probability", "predicted_mention_count", "direction")},
        "note": "의회 간 순서는 시간적 관측 순서이며 인과적 전파가 아님",
    }


def statistics_facts(result: dict[str, Any], regions: list[str]) -> dict[str, Any]:
    if result.get("error"):
        return {"indicator": result.get("indicator"), "error": result["error"]}
    values = result.get("values") or {}
    need = result.get("need_score") or {}
    ranked = sorted(need, key=lambda council: -need[council])
    focus = [council for council in regions if council in values]
    return {
        "indicator": result.get("name"), "period": result.get("period"), "unit": result.get("unit"),
        "direction": "값이 클수록 수요 큼" if result.get("need") == "high" else "값이 작을수록 수요 큼",
        "selected_regions": {council: {"value": values[council], "need_score": need.get(council)} for council in focus},
        "top_need": [{"council": c, "value": values.get(c), "need_score": need[c]} for c in ranked[:5]],
        "table_pinned": (result.get("table") or {}).get("pinned"),
        "attribution": result.get("attribution"),
    }


# ---------------------------------------------------------------------- ⑤ 답변 검증
NUMBER_PATTERN = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


def _numbers(text: str) -> list[float]:
    return [float(value.replace(",", "")) for value in NUMBER_PATTERN.findall(text)]


def _number_supported(value: float, allowed: list[float]) -> bool:
    """반올림(0.531→0.53)·백분율(0.531→53.1%) 표현까지 허용"""
    for known in allowed:
        for candidate in (known, known * 100):
            if abs(candidate - value) <= max(0.006, abs(candidate) * 0.01):
                return True
    return False


KOREAN_DATE = re.compile(r"(\d{4})\s*년\s*(\d{1,2})\s*월(?:\s*(\d{1,2})\s*일)?")


def _iso_dates(text: str) -> str:
    """'2023년 5월 25일' → '2023-05-25' (날짜 대조를 위해)"""
    return KOREAN_DATE.sub(
        lambda m: f"{m[1]}-{int(m[2]):02d}" + (f"-{int(m[3]):02d}" if m[3] else ""), text)


def verify_answer(
    answer: str, facts_text: str, evidence_text: str, question: str, evidence_count: int | None = None,
) -> dict[str, Any]:
    """답변 속 숫자·날짜·의회명·근거 번호가 계산 결과·근거·질문에 있는지 대조 (연도·10 이하 정수는 제외)"""
    allowed = facts_text + " " + evidence_text + " " + _iso_dates(question)
    allowed_numbers = _numbers(allowed)
    bad_citations = sorted({
        f"[근거 {n}]" for n in map(int, re.findall(r"\[근거\s*(\d+)\]", answer))
        if evidence_count is not None and not 1 <= n <= evidence_count
    })
    cleaned = re.sub(r"\[(근거|외부)[^\]]*\]", " ", _iso_dates(answer))
    cleaned_for_numbers = re.sub(r"\d{4}-\d{2}(-\d{2})?", " ", cleaned)   # 날짜는 아래 숫자 검사와 별개
    unknown_numbers = sorted({
        f"{value:g}" for value in _numbers(cleaned_for_numbers)
        if not (value.is_integer() and (value <= 10 or 2000 <= value <= 2100))
        and not _number_supported(value, allowed_numbers)
    })
    mentioned = {REGION_FORMS[form] for form in REGION_FORMS if len(form) >= 3 and form in cleaned}
    allowed_councils = {REGION_FORMS[form] for form in REGION_FORMS if len(form) >= 3 and form in allowed}
    unknown_councils = sorted(mentioned - allowed_councils)
    date_pattern = re.compile(r"\d{4}-\d{2}(?:-\d{2})?")
    unknown_dates = sorted(set(date_pattern.findall(cleaned)) - set(date_pattern.findall(allowed)))
    return {"passed": not unknown_numbers and not unknown_councils and not unknown_dates and not bad_citations,
            "unknown_numbers": unknown_numbers[:10] + unknown_dates[:5] + bad_citations,
            "unknown_councils": unknown_councils}


def template_answer(slots: dict[str, Any], facts: dict[str, Any]) -> str:
    """LLM이 없거나 검증에 실패했을 때의 정형 답변 (계산 결과만 그대로)"""
    lines = []
    issue = facts.get("issue")
    if issue:
        lines.append(f"- 이슈: {issue['issue_name']} (데이터: {issue['data_scope']})")
        lines.append(f"- 최초 관측: {issue['first_observation_date']} / {', '.join(issue['first_observation_councils'] or [])}")
        order = " → ".join(f"{row['council']}({row['first_observation_date']})" for row in issue["observation_order"][:6])
        if order:
            lines.append(f"- 관측 순서: {order} (인과적 전파 아님)")
    followup = facts.get("followup")
    if followup:
        lines.append(f"- [외부 공공데이터 · {followup['mode']} · 기준 {followup['as_of']} · 참고용]")
        lines.append(f"- 광역 조례화 의회: {', '.join(followup['already_institutionalized'] or []) or '없음'}")
        if followup.get("followups_after_minutes_cutoff"):
            lines.append(f"- 회의록 기준일 이후 실제 광역 조례화: {', '.join(followup['followups_after_minutes_cutoff'])}")
        candidates = ", ".join(f"{row['rank']}. {row['council']}({row['followup_score']})" for row in followup["top_candidates"])
        lines.append(f"- 후속 후보(규칙 기반 점수): {candidates or '없음'}")
        for row in followup["followup_flow"]:
            if row["council"] in slots["regions"]:
                lines.append(f"- {row['council']}: {row['stage']} · 기초 조례 {row['basic_ordinance_count']}건")
    for stat in facts.get("statistics") or []:
        if stat.get("error"):
            lines.append(f"- 통계 {stat.get('indicator')}: {stat['error']}")
            continue
        lines.append(f"- {stat['indicator']}({stat['period']}, {stat['unit']}, {stat['direction']}): "
                     + ", ".join(f"{c} {v['value']}" for c, v in stat["selected_regions"].items())
                     + f" · 수요 상위 {', '.join(row['council'] for row in stat['top_need'][:3])}")
    bills = facts.get("national_bills")
    if bills:
        recent = bills["items"][0] if bills["items"] else None
        lines.append(f"- 국회 관련 법률안 {bills['count']}건" + (f" (최근: {recent['date']} 「{recent['bill']}」)" if recent else ""))
    if not lines:
        lines.append("- 질문과 연결되는 계산 결과가 없어 근거 부족입니다.")
    return "\n".join(lines)
