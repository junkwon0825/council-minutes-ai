#!/usr/bin/env python3
"""공개 포트폴리오용 로컬 Agent.

검색 결과를 새로 만들어내지 않고 CouncilAgentTools가 반환한 근거를
로컬 Ollama에 전달해 설명형 답변을 생성한다.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any

from .tools import CouncilAgentTools


@dataclass
class LocalIssueAgent:
    tools: CouncilAgentTools
    ollama_url: str = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
    model: str = os.environ.get("COUNCIL_AGENT_MODEL", "A.X-4.0-Light-gguf:Q4_K_M")

    def ask(self, question: str, top_k: int = 6) -> dict[str, Any]:
        evidence = self.tools.search_council_evidence(question, top_k=top_k)
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "제공된 지방의회 회의록 근거만 사용해 답하라. "
                        "근거에 없는 사실은 추정하지 말고 부족하다고 명시하라."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"question": question, "evidence": evidence.get("results", [])},
                        ensure_ascii=False,
                    ),
                },
            ],
        }
        request = urllib.request.Request(
            self.ollama_url.rstrip("/") + "/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
            answer = str((body.get("message") or {}).get("content") or "").strip()
            mode = "ollama"
        except Exception as error:
            answer = "로컬 LLM 연결이 없어 검색 근거만 반환합니다."
            mode = f"retrieval_only:{type(error).__name__}"
        return {"question": question, "answer": answer, "mode": mode, "evidence": evidence}
