#!/usr/bin/env python3
"""로컬 검색과 Ollama를 결합한 근거 제한형 의정 이슈 Agent.

기본 설정은 localhost Ollama만 허용한다. 회의록 원문을 외부 API로 전송하지
않으며, 답변은 검색된 근거와 고정 이슈 결과 안에서만 생성한다.
pipeline(agent/query_pipeline.py)이 주어지면 자유 질문을 '해석(LLM, JSON 스키마) → 검증·보정(코드)
→ 의도별 실행(코드) → 답변(LLM) → 답변 검증(코드)'으로 처리한다. LLM 호출은 질문당 두 번이며,
외부 공공 API로는 검증된 짧은 정책어와 KOSIS 통계표 ID만 나간다.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse


LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
ANSWER_RULES = (
    "당신은 '의정이슈레이더'의 로컬 분석 Agent다. [계산 결과]와 [회의록 근거]만 사용한다. "
    "근거에 없는 사실·숫자·의회는 만들지 말고 '근거 부족'이라 말한다. 숫자는 계산 결과의 값을 그대로 옮긴다. "
    "[근거 N]은 [회의록 근거]의 N번째 항목에만 쓰고, 조례·법안·통계 같은 외부 공공데이터는 "
    "[외부: 법제처]·[외부: 국회]·[외부: KOSIS]로 표시한다. 조례는 계산 결과에 있는 조례 이름을 그대로 쓴다. "
    "최초 관측과 의회 간 순서는 인과관계가 아닌 시간적 관측 순서라고 표현한다. "
    "후속 점수는 미리 고정한 규칙 기반 참고값이지 확률이 아니라고 밝힌다. "
    "mode=live이면 외부 데이터는 조회일 기준이고 회의록은 기준일까지라는 점을, "
    "backtest이면 기준일 시점 자료만 썼다는 점을 한 번 언급한다. "
    "질문의 의도와 지역에 맞춰 한국어로 간결하게 답한다."
)


def probe_ollama(
    ollama_url: str,
    model: str,
    timeout_seconds: float = 3.0,
) -> dict[str, Any]:
    """Ollama 서버와 요청 모델의 설치 상태를 빠르게 확인한다."""
    host = (urlparse(ollama_url).hostname or "").lower()
    if host not in LOCAL_HOSTS:
        return {
            "reachable": False,
            "model_available": False,
            "models": [],
            "error": "보안 정책상 localhost Ollama만 사용할 수 있습니다.",
        }
    request = urllib.request.Request(
        ollama_url.rstrip("/") + "/api/tags",
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        names = sorted({
            str(value)
            for item in (payload.get("models") or [])
            for value in (item.get("name"), item.get("model"))
            if value
        })
        requested = model.strip()
        available = requested in names
        if not available and ":" not in requested:
            available = any(name.split(":", 1)[0] == requested for name in names)
        return {
            "reachable": True,
            "model_available": available,
            "models": names,
            "error": None if available else f"로컬 모델 '{requested}'이 설치되지 않았습니다.",
        }
    except Exception as error:
        return {
            "reachable": False,
            "model_available": False,
            "models": [],
            "error": f"{type(error).__name__}: {error}",
        }


def _normalized_terms(value: str) -> set[str]:
    cleaned = re.sub(r"[^0-9A-Za-z가-힣]+", " ", value.casefold())
    return {term for term in cleaned.split() if len(term) >= 2}


def route_fixed_issue(
    question: str, payloads: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, float]:
    """질문과 가장 가까운 검수 완료 이슈를 보수적인 문자열 점수로 연결한다."""
    question_text = question.casefold()
    question_terms = _normalized_terms(question)
    best_payload = None
    best_score = 0.0
    for payload in payloads:
        issue = payload.get("issue") or {}
        names = [str(issue.get("issue_name") or ""), *(
            str(value) for value in (issue.get("aliases") or [])
        )]
        score = 0.0
        for name in names:
            normalized_name = name.strip().casefold()
            if not normalized_name:
                continue
            if normalized_name in question_text:
                score = max(score, 1.0)
                continue
            terms = _normalized_terms(normalized_name)
            if terms:
                overlap = len(question_terms & terms) / len(terms)
                score = max(score, overlap)
                distinctive = {
                    term for term in terms
                    if len(term) >= 3 and term in question_terms
                }
                if distinctive:
                    score = max(score, 0.75)
        if score > best_score:
            best_payload, best_score = payload, score
    return (best_payload, best_score) if best_score >= 0.5 else (None, best_score)


def _clean_generation(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def issue_snapshot(payload: dict[str, Any]) -> str:
    issue = payload.get("issue") or {}
    if not issue.get("issue_id"):
        return (
            "연결된 검수 완료 이슈 없음. 검색된 회의록 근거만 사용하며, "
            "최초 관측·전파·예측은 확정적으로 말하지 않는다."
        )
    summary = payload.get("summary") or {}
    forecast = payload.get("forecast") or {}
    propagation = payload.get("propagation") or []
    issue_status = str(issue.get("status") or issue.get("id_status") or "fixed")
    order = " → ".join(
        f"{row.get('council')}({row.get('first_observation_date')})"
        for row in propagation
    )
    lower, upper = (forecast.get("prediction_interval_80") or [None, None])[:2]
    return (
        f"이슈 ID: {issue.get('issue_id')}\n"
        f"이슈명: {issue.get('issue_name')}\n"
        f"이슈 상태: {issue_status}\n"
        f"데이터 범위: 2024~2025년 제공 회의록\n"
        f"최초 관측: {summary.get('first_observation_date')} / "
        f"{', '.join(summary.get('first_observation_councils') or [])}\n"
        f"관측 의회 수: {summary.get('council_count')}\n"
        f"시간적 관측 순서: {order}\n"
        f"예측 대상: {forecast.get('target_month')}\n"
        f"예측 방향: {forecast.get('direction')}\n"
        f"예상 언급: {forecast.get('predicted_mention_count')}건 "
        f"(80% 경험적 구간 {lower}~{upper})\n"
        + followup_snapshot(payload)
        + "주의: 의회 간 연결은 시간적 관측 순서이며 인과적 전파가 아니다."
        + (
            " 이 결과는 자유검색으로 생성한 검수 전 provisional 분석이다."
            if "provisional" in issue_status else ""
        )
    )


def followup_snapshot(payload: dict[str, Any]) -> str:
    """32a 사전 계산 결과 요약. 있으면 LLM이 도구를 부르지 않고도 후속 흐름을 답할 수 있다"""
    followup = payload.get("external_followup") or {}
    view = followup.get("outlook") or {}
    if not followup:
        return ""
    lag = view.get("median_lag_days_minutes_to_ordinance")
    candidates = ", ".join(
        f"{row.get('council')}({row.get('expected_next_step')}, {row.get('followup_score')})"
        for row in view.get("top_candidates") or []
    )
    return (
        f"[외부 공공데이터 후속 흐름 · 기준일 {followup.get('analysis_cutoff')} · 참고용]\n"
        f"광역 조례화 의회: {', '.join(view.get('already_institutionalized') or []) or '없음'}\n"
        f"후속 조례화 후보: {candidates or '없음'}\n"
        f"논의→조례화 중앙 소요일: {f'{int(lag)}일' if lag is not None else '근거 부족'}\n"
    )


def payload_evidence(payload: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    result = []
    for row in (payload.get("evidence") or [])[:limit]:
        result.append({
            "chunk_id": row.get("chunk_id"),
            "similarity": row.get("similarity"),
            "text": str(row.get("text") or ""),
            "metadata": {
                "council": row.get("council"),
                "meeting_date": row.get("meeting_date"),
                "committee": row.get("committee"),
                "meeting_name": row.get("meeting_name"),
                "agenda_title": row.get("agenda_title"),
            },
        })
    return result


def format_evidence(rows: list[dict[str, Any]], max_chars: int = 1100) -> str:
    blocks = []
    for index, row in enumerate(rows, start=1):
        metadata = row.get("metadata") or {}
        text = " ".join(str(row.get("text") or "").split())[:max_chars]
        blocks.append(
            f"[근거 {index}] {metadata.get('council', '미상')} | "
            f"{metadata.get('meeting_date', '날짜 미상')} | "
            f"{metadata.get('meeting_name', '회의록명 미상')} | "
            f"{metadata.get('agenda_title', '안건 미상')}\n{text}"
        )
    return "\n\n".join(blocks)


def evidence_citations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    citations = []
    for index, row in enumerate(rows, start=1):
        metadata = row.get("metadata") or {}
        citations.append({
            "citation": index,
            "chunk_id": row.get("chunk_id"),
            "council": metadata.get("council"),
            "meeting_date": metadata.get("meeting_date"),
            "meeting_name": metadata.get("meeting_name"),
            "agenda_title": metadata.get("agenda_title"),
            "similarity": row.get("similarity"),
            "text": str(row.get("text") or ""),
        })
    return citations


@dataclass
class LocalCouncilAgent:
    issue_payload: dict[str, Any]
    search_service: Any | None = None
    ollama_url: str = "http://127.0.0.1:11434"
    model: str = "qwen3:8b"
    timeout_seconds: int = 180
    pipeline: Any | None = None          # agent.query_pipeline.QuestionPipeline
    parser_model: str | None = None      # 질문 해석용 모델 (없으면 model과 같음)
    parser_timeout_seconds: int = 120

    def __post_init__(self) -> None:
        host = (urlparse(self.ollama_url).hostname or "").lower()
        if host not in LOCAL_HOSTS:
            raise ValueError(
                "회의록 외부 전송 방지를 위해 Ollama URL은 localhost만 허용됩니다"
            )

    def retrieve(self, question: str, top_k: int = 6) -> tuple[list[dict[str, Any]], str]:
        if self.search_service is not None:
            result = self.search_service.search(
                query=question, top_k=top_k, candidate_k=max(50, top_k * 10),
                include_parent=True,
            )
            rows = list(result.get("results") or [])
            if rows:
                mode = str(result.get("retrieval_mode") or "dense")
                return rows, f"pgvector_{mode}_retrieval"
        return payload_evidence(self.issue_payload, top_k), "fixed_issue_payload"

    def _ollama_answer(
        self,
        question: str,
        evidence_rows: list[dict[str, Any]],
        history: list[dict[str, str]] | None,
    ) -> str:
        status = probe_ollama(
            self.ollama_url,
            self.model,
            timeout_seconds=min(3.0, float(self.timeout_seconds)),
        )
        if not status["reachable"]:
            raise RuntimeError(f"Ollama 서버 미연결: {status['error']}")
        if not status["model_available"]:
            installed = ", ".join(status["models"][:8]) or "없음"
            raise RuntimeError(
                f"Ollama 모델 미설치: {self.model}. 설치된 모델: {installed}. "
                f"터미널에서 `ollama pull {self.model}`을 실행하세요."
            )
        system = (
            "당신은 '의정이슈레이더'의 로컬 분석 Agent다. 제공된 이슈 요약과 "
            "회의록 근거만 사용한다. 근거에 없는 사실은 추측하지 말고 '근거 부족'이라 "
            "말한다. 최초 관측과 전파는 인과관계가 아닌 시간적 관측 순서라고 표현한다. "
            "답변의 핵심 문장마다 [근거 1] 형식으로 인용하고, 2025년 이후 외부정보를 "
            "사용하지 않는다. 한국어로 간결하게 답한다."
        )
        user_context = (
            "[고정 이슈 요약]\n" + issue_snapshot(self.issue_payload)
            + "\n\n[검색된 회의록]\n" + format_evidence(evidence_rows)
            + "\n\n[질문]\n" + question
        )
        messages = [{"role": "system", "content": system}]
        for item in (history or [])[-4:]:
            if item.get("role") in {"user", "assistant"} and item.get("content"):
                messages.append({
                    "role": item["role"], "content": str(item["content"])[:3000]
                })
        messages.append({"role": "user", "content": user_context})
        return _clean_generation(str(self._chat(messages).get("content") or ""))

    def _chat(
        self,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any] | None = None,
        num_predict: int = 600,
        model: str | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "keep_alive": "10m",
            "options": {
                "temperature": 0.1,
                "num_ctx": 8192,
                "num_predict": num_predict,
            },
        }
        if response_format:
            body["format"] = response_format   # Ollama structured output: JSON 스키마 강제
        request = urllib.request.Request(
            self.ollama_url.rstrip("/") + "/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout or self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload.get("message") or {}

    def _llm_ready(self) -> tuple[bool, str | None]:
        status = probe_ollama(self.ollama_url, self.model, timeout_seconds=min(3.0, float(self.timeout_seconds)))
        if not status["reachable"]:
            return False, f"Ollama 서버 미연결: {status['error']}"
        if not status["model_available"]:
            return False, f"Ollama 모델 미설치: {self.model}"
        return True, None

    def _ask_pipeline(
        self,
        question: str,
        history: list[dict[str, str]] | None,
        previous_slots: dict[str, Any] | None,
        top_k: int,
    ) -> dict[str, Any]:
        """질문 파이프라인: ① 해석 → ② 검증·보정 → ③ 의도별 실행 → ④ 답변 → ⑤ 답변 검증"""
        from agent.query_pipeline import template_answer, verify_answer  # 순환 import 방지

        warnings: list[str] = []
        llm_ok, llm_error = self._llm_ready()
        if llm_error:
            warnings.append(llm_error)

        def parse_llm(messages, schema, num_predict):
            message = self._chat(messages, schema, num_predict, self.parser_model, self.parser_timeout_seconds)
            return str(message.get("content") or "")

        slots, parser, parse_warning = self.pipeline.parse(question, previous_slots, parse_llm if llm_ok else None)
        if parse_warning:
            warnings.append(parse_warning)
        plan: dict[str, Any] = {"slots": slots, "parser": parser}
        if slots["needs_clarification"]:
            return self._pipeline_response(
                slots.get("clarification_question") or "어떤 정책 주제에 대한 질문인지 조금 더 알려주세요.",
                [], "clarification", warnings, plan, [])
        grounded = self.pipeline.ground(slots, question)
        plan.update({"route": grounded["route"], "terms": grounded["terms"], "indicators": grounded["indicators"]})
        if grounded["route"].get("ambiguous"):
            names = " / ".join(str(row["issue_name"]) for row in grounded["route"]["candidates"][:2])
            return self._pipeline_response(
                f"다음 중 어떤 이슈를 말씀하신 건가요? {names}", [], "clarification", warnings, plan, [])
        self.issue_payload = grounded["payload"] or {}

        def retrieve(query: str) -> tuple[list[dict[str, Any]], str]:
            try:
                return self.retrieve(query, top_k=top_k)
            except Exception as error:
                warnings.append(f"검색 실패: {type(error).__name__}: {error}")
                return payload_evidence(self.issue_payload, top_k), "fixed_issue_payload_after_search_error"

        executed = self.pipeline.execute(slots, grounded, retrieve)
        plan["steps"] = executed["steps"]
        rows = executed["evidence"] or payload_evidence(self.issue_payload, top_k)
        facts = executed["facts"]
        facts_text = json.dumps(facts, ensure_ascii=False)
        evidence_text = format_evidence(rows)
        if not llm_ok:
            return self._pipeline_response(
                "로컬 LLM에 연결하지 못해 계산 결과를 그대로 요약합니다.\n" + template_answer(slots, facts),
                rows, "deterministic_fallback", warnings, plan, executed["sent_externally"])
        messages = [{"role": "system", "content": ANSWER_RULES}]
        for item in (history or [])[-4:]:
            if item.get("role") in {"user", "assistant"} and item.get("content"):
                messages.append({"role": item["role"], "content": str(item["content"])[:2000]})
        messages.append({"role": "user", "content": (
            f"[질문 해석]\n{json.dumps(slots, ensure_ascii=False)}\n\n"
            f"[계산 결과]\n{facts_text}\n\n[회의록 근거]\n{evidence_text}\n\n[질문]\n{question}"
        )})
        try:
            answer = _clean_generation(str(self._chat(messages, num_predict=700).get("content") or ""))
            if not answer:
                raise RuntimeError("Ollama가 빈 답변을 반환했습니다")
        except Exception as error:
            warnings.append(f"답변 생성 대체: {type(error).__name__}: {error}")
            return self._pipeline_response(
                "로컬 LLM 답변 생성에 실패해 계산 결과를 그대로 요약합니다.\n" + template_answer(slots, facts),
                rows, "deterministic_fallback", warnings, plan, executed["sent_externally"])
        verification = verify_answer(answer, facts_text, evidence_text, question, len(rows))
        plan["verification"] = verification
        if not verification["passed"]:
            warnings.append(
                "답변 검증 실패(계산 결과에 없는 값): "
                + ", ".join(verification["unknown_numbers"] + verification["unknown_councils"]))
            return self._pipeline_response(
                "LLM 답변에 계산 결과와 맞지 않는 값이 있어 계산 결과를 그대로 요약합니다.\n"
                + template_answer(slots, facts),
                rows, "verified_fallback", warnings, plan, executed["sent_externally"])
        return self._pipeline_response(
            answer, rows, f"ollama:{self.model}", warnings, plan, executed["sent_externally"])

    def _pipeline_response(
        self, answer: str, rows: list[dict[str, Any]], generation_mode: str,
        warnings: list[str], plan: dict[str, Any], sent: list[str],
    ) -> dict[str, Any]:
        return {
            "answer": answer,
            "citations": evidence_citations(rows),
            "retrieval_mode": "query_pipeline",
            "generation_mode": generation_mode,
            "warning": "; ".join(warnings) or None,
            "plan": plan,
            "issue_payload": self.issue_payload,
            "policy": {
                "external_llm_api_used": False,
                "meeting_text_sent_externally": False,
                "question_text_sent_externally": False,
                "answer_grounded_in_retrieved_evidence": True,
                "external_public_api_values": sent,
            },
        }

    def fallback_answer(
        self, question: str, evidence_rows: list[dict[str, Any]]
    ) -> str:
        summary = self.issue_payload.get("summary") or {}
        forecast = self.issue_payload.get("forecast") or {}
        lines = [
            "로컬 LLM에 연결하지 못해 구조화 결과를 그대로 요약합니다.",
        ]
        if (self.issue_payload.get("issue") or {}).get("issue_id"):
            lines.extend([
                f"- 최초 관측: {summary.get('first_observation_date')} / "
                f"{', '.join(summary.get('first_observation_councils') or [])}",
                f"- 관측 범위: {summary.get('council_count')}개 의회, "
                f"근거 {summary.get('evidence_unit_count')}건",
                f"- 다음 달 전망: {forecast.get('direction')} / "
                f"예상 {forecast.get('predicted_mention_count')}건",
            ])
        else:
            lines.append("- 고정 이슈가 아니므로 검색 근거만 제시합니다.")
        snapshot = followup_snapshot(self.issue_payload)
        if snapshot:
            lines.extend("- " + line for line in snapshot.strip().splitlines())
        for index, row in enumerate(evidence_rows[:3], start=1):
            metadata = row.get("metadata") or {}
            lines.append(
                f"- [근거 {index}] {metadata.get('council')} "
                f"{metadata.get('meeting_date')} · {metadata.get('meeting_name')}"
            )
        lines.append(f"- 질문: {question}")
        return "\n".join(lines)

    def ask(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        top_k: int = 6,
        previous_slots: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("질문이 비었습니다")
        if self.pipeline is not None:
            return self._ask_pipeline(question, history, previous_slots, top_k)
        retrieval_warning = None
        try:
            rows, retrieval_mode = self.retrieve(question, top_k=top_k)
        except Exception as error:
            retrieval_warning = f"검색 실패: {type(error).__name__}: {error}"
            rows = payload_evidence(self.issue_payload, top_k)
            retrieval_mode = (
                "fixed_issue_payload_after_search_error"
                if rows else "search_error_no_fallback"
            )
        if not rows:
            return {
                "answer": (
                    "질문과 관련된 회의록 근거를 찾지 못했습니다. "
                    "pgvector 연결과 검색 설정을 확인해 주세요."
                ),
                "citations": [], "retrieval_mode": retrieval_mode,
                "generation_mode": "no_evidence",
                "warning": retrieval_warning,
            }
        try:
            answer = self._ollama_answer(question, rows, history)
            if not answer:
                raise RuntimeError("Ollama가 빈 답변을 반환했습니다")
            generation_mode = f"ollama:{self.model}"
            warning = retrieval_warning
        except Exception as error:
            answer = self.fallback_answer(question, rows)
            generation_mode = "deterministic_fallback"
            warning = "; ".join(
                value for value in [retrieval_warning, str(error)] if value
            )
        return {
            "answer": answer,
            "citations": evidence_citations(rows),
            "retrieval_mode": retrieval_mode,
            "generation_mode": generation_mode,
            "warning": warning,
            "policy": {
                "external_llm_api_used": False,
                "meeting_text_sent_externally": False,
                "answer_grounded_in_retrieved_evidence": True,
            },
        }
