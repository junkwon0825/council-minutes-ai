#!/usr/bin/env python3
"""KMeans 군집 대표 발언을 로컬 Ollama LLM으로 이슈명/요약화한다.

회의록 텍스트는 localhost의 Ollama에만 전달하며 외부 API를 사용하지 않는다.
먼저 --prepare-only로 프롬프트를 검수한 뒤 실제 추론을 실행할 수 있다.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path


SCRIPT_VERSION = "2026-10-19-local-issue-labeling-v2.1-strict"
QUALITY_VALUES = {"coherent", "mixed", "procedural", "institution_specific"}

# LLM이 서로 다른 정책 발언을 '정책 수립', '예산 편성'처럼 내용 없는 공통어로
# 묶거나, 조례 심사 절차를 정책 이슈로 오인하는 경우를 최종 코드 검증으로 차단한다.
PROCEDURAL_PATTERNS = tuple(re.compile(pattern) for pattern in (
    r"의사일정\s*제?\s*\d+\s*항",
    r"(?:안건|조례안).{0,40}상정",
    r"질의하실\s+위원",
    r"질의.{0,20}(?:종결|마치)",
    r"의결하도록\s+하겠습니다",
    r"검토보고.{0,50}(?:대체|갈음)",
    r"(?:회의진행|회의 진행).{0,30}(?:위해|효율)",
    r"속기록.{0,30}등재",
    r"제안설명이?\s+(?:있겠습니다|있었겠습니다)",
))
GENERIC_ISSUE_TOKENS = {
    "정책", "수립", "예산", "편성", "부족", "필요", "필요성", "문제", "문제점",
    "업무", "주요", "계획", "보고", "검토", "의견", "수렴", "논의", "추진", "사업",
    "지원", "개선", "강화", "확대", "효율", "효율화", "활성화", "운영", "관리",
    "행정", "기관", "의회", "절차", "협력", "대응", "방안", "마련", "관련",
    "지역", "지방", "지자체", "요구", "제기", "현황", "대한", "위한", "그리고", "또는",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="로컬 LLM 군집 이슈명 생성")
    parser.add_argument(
        "--cluster-summary",
        default="outputs/clustering_semantic_test_v2_k160/cluster_summary.csv",
    )
    parser.add_argument(
        "--representatives",
        default="outputs/clustering_semantic_test_v2_k160/cluster_representatives.csv",
    )
    parser.add_argument("--output-dir", default="outputs/llm_labels_semantic_test_v2_k160_strict")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434/api/generate")
    parser.add_argument("--limit", type=int, default=20, help="0이면 조건을 만족한 전체 군집")
    parser.add_argument("--min-cluster-size", type=int, default=10)
    parser.add_argument("--min-councils", type=int, default=1)
    parser.add_argument("--max-representatives", type=int, default=8)
    parser.add_argument("--max-text-chars", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument(
        "--min-dominant-ratio", type=float, default=0.75,
        help="coherent로 인정할 대표 발언의 최소 동일 이슈 비율",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="출력 폴더의 기존 JSONL 결과를 비우고 처음부터 다시 실행",
    )
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def safe_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clean_text(value: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit].rstrip()


def atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_completed(path: Path) -> dict[int, dict]:
    completed = {}
    if not path.exists():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        version = str(item.get("script_version") or "")
        required = {
            "dominant_topic_ratio", "supporting_evidence_ids",
            "outlier_evidence_ids", "review_status",
        }
        missing = required - set(item)
        if version != SCRIPT_VERSION or missing:
            raise SystemExit(
                "기존 결과가 현재 audited v2 형식과 다릅니다. "
                f"기존={version or 'unknown'}, 현재={SCRIPT_VERSION}, "
                f"누락={','.join(sorted(missing)) or '없음'}. "
                "새 --output-dir을 사용하거나 --overwrite로 다시 실행하세요."
            )
        completed[int(item["cluster_id"])] = item
    return completed


def build_prompt(cluster: dict, representatives: list[dict], max_text_chars: int) -> str:
    evidence = []
    for index, row in enumerate(representatives, start=1):
        text = clean_text(row.get("semantic_text") or row.get("issue_text"), max_text_chars)
        evidence.append(
            f"[{index}] 의회={row.get('council', '')}, 날짜={row.get('meeting_date', '')}, "
            f"유사도={row.get('similarity', '')}\n{text}"
        )

    return f"""당신은 광역 지방의회 정책 분석가입니다.
아래 군집 정보와 대표 발언만 근거로 군집의 정책 이슈를 판정하세요.
외부 지식이나 제공되지 않은 사실을 추가하지 마세요.

판정 절차:
1. 대표 발언을 각각 독립적으로 읽고 evidence_topic_map에 발언별 핵심 주제를 적으세요.
2. 가장 많은 발언이 지지하는 하나의 구체적 정책 이슈를 dominant_topic으로 정하세요.
3. 그 이슈를 직접 지지하는 번호만 supporting_evidence_ids에 넣고, 다른 주제 번호는
   outlier_evidence_ids에 넣으세요. 단순히 '예산', '지원', '검토', '업무보고'가 같다는
   이유만으로 같은 정책 이슈로 묶으면 안 됩니다.
4. dominant_topic_ratio는 supporting_evidence_ids 수 / 전체 대표 발언 수입니다.

판정 원칙:
1. 사람 이름, 기관 이름, 위원회 이름, '업무보고' 자체를 이슈명으로 사용하지 마세요.
2. 대표 발언 75% 이상이 동일한 정책 대상과 문제 또는 조치를 공유할 때만 coherent입니다.
3. 서로 다른 정책이 섞였거나 공통점이 행정적 표현뿐이면 mixed입니다.
4. 회의 진행, 인사말, 간부 소개가 중심이면 procedural입니다.
5. 한 기관 운영에만 해당하고 일반 정책 이슈로 확장하기 어려우면 institution_specific입니다.
6. issue_name은 15~35자의 구체적인 한국어 명사구로 작성하세요.
7. 일부 대표 발언에만 맞는 이름을 전체 군집 이름처럼 만들지 마세요.
8. 근거가 부족하면 confidence를 낮게 지정하세요.
9. 회의명이나 안건 형식이 같아도 발언 내용의 정책 대상이 다르면 mixed입니다.
10. '정책 수립 및 예산 편성 부족', '업무 계획 검토', '의회 운영 효율화'처럼
    구체적 정책 대상이 없는 일반 행정 표현은 coherent 이슈로 만들지 마세요.

군집 정보:
- cluster_id: {cluster['cluster_id']}
- 크기: {cluster.get('size', '')}
- 평균 중심 유사도: {cluster.get('mean_similarity', '')}
- 포함 의회 수: {cluster.get('council_count', '')}
- 최초 날짜: {cluster.get('first_date', '')}
- 최종 날짜: {cluster.get('last_date', '')}
- TF-IDF 임시 키워드: {cluster.get('keywords', '')}

대표 발언:
{chr(10).join(evidence)}

아래 키만 가진 JSON 객체 하나만 출력하세요.
{{
  "evidence_topic_map": [
    {{"evidence_id": 1, "topic": "발언 1의 핵심 정책 주제"}}
  ],
  "dominant_topic": "가장 많이 지지된 구체적 정책 주제",
  "supporting_evidence_ids": [1, 2],
  "outlier_evidence_ids": [3],
  "dominant_topic_ratio": 0.0,
  "issue_name": "정책 이슈명",
  "issue_summary": "대표 발언에 근거한 두 문장 이내 설명",
  "policy_domain": "인구·복지·교육·교통·환경·산업경제·농수산·안전·문화관광·행정 중 하나 또는 기타",
  "keywords": ["핵심어1", "핵심어2", "핵심어3"],
  "cluster_quality": "coherent 또는 mixed 또는 procedural 또는 institution_specific",
  "confidence": 0.0,
  "rationale": "판정 근거 한 문장"
}}"""


def call_ollama(url: str, model: str, prompt: str, temperature: float, timeout: int) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": temperature, "seed": 42},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    raw = str(body.get("response") or "").strip()
    if not raw:
        raise ValueError("Ollama 응답에 response가 없습니다.")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError("LLM 응답에서 JSON 객체를 찾지 못했습니다.")
        return json.loads(match.group(0))


def normalize_id_list(value, evidence_count: int) -> list[int]:
    if not isinstance(value, list):
        value = re.findall(r"\d+", str(value or ""))
    output = []
    for item in value:
        number = safe_int(item, -1)
        if 1 <= number <= evidence_count and number not in output:
            output.append(number)
    return sorted(output)


def representative_text(row: dict) -> str:
    return clean_text(row.get("semantic_text") or row.get("issue_text"), 4000)


def procedural_evidence_ratio(representatives: list[dict]) -> float:
    """대표 발언 중 회의 진행 문구가 중심인 비율을 계산한다."""
    if not representatives:
        return 0.0
    procedural_count = 0
    for row in representatives:
        text = representative_text(row)
        hits = sum(bool(pattern.search(text)) for pattern in PROCEDURAL_PATTERNS)
        if hits >= 2:
            procedural_count += 1
    return procedural_count / len(representatives)


def concrete_issue_anchors(issue_name: str) -> list[str]:
    """이슈명에서 연도·조사·일반 행정어를 빼고 구체적인 정책 대상을 남긴다."""
    anchors = []
    for token in re.findall(r"[가-힣A-Za-z0-9]+", issue_name):
        if re.fullmatch(r"(?:19|20)\d{2}년?", token):
            continue
        normalized = re.sub(r"(?:으로|에서|에게|까지|부터|처럼|보다|에는|에도|과의|와의|의)$", "", token)
        if len(normalized) < 2 or normalized in GENERIC_ISSUE_TOKENS:
            continue
        anchors.append(normalized)
    return anchors


def normalize_label(
    raw: dict,
    representatives: list[dict],
    min_dominant_ratio: float,
) -> dict:
    evidence_count = len(representatives)
    quality = str(raw.get("cluster_quality") or "mixed").strip().lower()
    if quality not in QUALITY_VALUES:
        quality = "mixed"
    keywords = raw.get("keywords")
    if not isinstance(keywords, list):
        keywords = [part.strip() for part in str(keywords or "").split(",") if part.strip()]
    confidence = min(1.0, max(0.0, safe_float(raw.get("confidence"), 0.0)))
    supporting_ids = normalize_id_list(raw.get("supporting_evidence_ids"), evidence_count)
    outlier_ids = normalize_id_list(raw.get("outlier_evidence_ids"), evidence_count)
    # 겹치는 번호가 있으면 지지 근거를 우선하고, 빠진 번호는 outlier로 간주한다.
    outlier_ids = [number for number in outlier_ids if number not in supporting_ids]
    classified = set(supporting_ids) | set(outlier_ids)
    outlier_ids.extend(number for number in range(1, evidence_count + 1) if number not in classified)
    dominant_ratio = len(supporting_ids) / evidence_count if evidence_count else 0.0

    validation_flags = []

    # LLM이 모든 군집을 coherent로 낙관 판정하는 현상을 코드 수준에서 차단한다.
    if quality == "coherent" and dominant_ratio < min_dominant_ratio:
        quality = "mixed"
        confidence = min(confidence, 0.6)
        validation_flags.append("dominant_ratio_below_threshold")

    issue_name = clean_text(raw.get("issue_name"), 80)
    procedural_ratio = procedural_evidence_ratio(representatives)
    anchors = concrete_issue_anchors(issue_name)

    # 대표 발언 절반 이상이 상정·검토보고·질의종결 같은 진행 문구이면 정책 이슈가 아니다.
    if procedural_ratio >= 0.5:
        quality = "procedural"
        confidence = min(confidence, 0.6)
        issue_name = "회의 진행성 군집"
        validation_flags.append("procedural_evidence_majority")

    # 구체적인 정책 대상이 전혀 없는 이슈명은 시계열/전파 분석에 자동 채택하지 않는다.
    if quality == "coherent" and not anchors:
        validation_flags.append("generic_issue_without_policy_object")

    if quality == "mixed":
        issue_name = "혼합 군집 - 재군집 필요"
    elif quality == "procedural" and not issue_name:
        issue_name = "회의 진행성 군집"
    supporting_councils = {
        clean_text(representatives[number - 1].get("council"), 100)
        for number in supporting_ids
        if clean_text(representatives[number - 1].get("council"), 100)
    }
    cross_council_ready = (
        quality == "coherent"
        and len(supporting_councils) >= 2
        and not validation_flags
    )
    if quality == "coherent" and confidence >= 0.75 and not validation_flags:
        review_status = "accepted_candidate" if cross_council_ready else "local_only_review"
    else:
        review_status = "manual_review"

    topic_map = raw.get("evidence_topic_map")
    if not isinstance(topic_map, list):
        topic_map = []

    return {
        "issue_name": issue_name or "검수 필요",
        "issue_summary": clean_text(raw.get("issue_summary"), 500),
        "policy_domain": clean_text(raw.get("policy_domain"), 40) or "기타",
        "keywords": [clean_text(value, 40) for value in keywords[:7] if clean_text(value, 40)],
        "cluster_quality": quality,
        "confidence": round(confidence, 4),
        "rationale": clean_text(raw.get("rationale"), 300),
        "dominant_topic": clean_text(raw.get("dominant_topic"), 120),
        "dominant_topic_ratio": round(dominant_ratio, 4),
        "supporting_evidence_ids": supporting_ids,
        "outlier_evidence_ids": sorted(set(outlier_ids)),
        "supporting_council_count": len(supporting_councils),
        "cross_council_ready": cross_council_ready,
        "review_status": review_status,
        "procedural_evidence_ratio": round(procedural_ratio, 4),
        "concrete_issue_anchors": anchors,
        "validation_flags": validation_flags,
        "evidence_topic_map": topic_map[:evidence_count],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "cluster_id", "issue_id", "issue_name", "issue_summary", "policy_domain",
        "keywords", "cluster_quality", "confidence", "rationale", "size",
        "mean_similarity", "council_count", "first_date", "last_date",
        "provisional_label", "dominant_topic", "dominant_topic_ratio",
        "supporting_evidence_ids", "outlier_evidence_ids",
        "supporting_council_count", "cross_council_ready", "review_status",
        "procedural_evidence_ratio", "concrete_issue_anchors", "validation_flags",
        "evidence_topic_map", "model", "script_version",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["keywords"] = " | ".join(output.get("keywords") or [])
            for field in (
                "supporting_evidence_ids", "outlier_evidence_ids", "concrete_issue_anchors",
                "validation_flags", "evidence_topic_map"
            ):
                output[field] = json.dumps(output.get(field), ensure_ascii=False)
            writer.writerow(output)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    print(f"스크립트 버전: {SCRIPT_VERSION}", flush=True)
    if args.limit < 0 or args.min_cluster_size < 1 or args.min_councils < 1:
        raise SystemExit("--limit은 0 이상, 최소 크기와 의회 수는 1 이상이어야 합니다.")
    if args.max_representatives < 1 or args.max_text_chars < 100 or args.timeout < 1:
        raise SystemExit("대표 발언 수, 텍스트 길이, timeout 설정을 확인하세요.")
    if not 0.5 <= args.min_dominant_ratio <= 1.0:
        raise SystemExit("--min-dominant-ratio는 0.5 이상 1.0 이하여야 합니다.")

    summary_path = Path(args.cluster_summary).expanduser().resolve()
    representatives_path = Path(args.representatives).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    for path in (summary_path, representatives_path):
        if not path.exists():
            raise SystemExit(f"입력 파일이 없습니다: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    clusters = read_csv(summary_path)
    reps_by_cluster = defaultdict(list)
    for row in read_csv(representatives_path):
        reps_by_cluster[safe_int(row.get("cluster_id"))].append(row)
    for rows in reps_by_cluster.values():
        rows.sort(key=lambda row: safe_int(row.get("representative_rank"), 999))

    eligible = []
    for cluster in clusters:
        cluster_id = safe_int(cluster.get("cluster_id"))
        cluster["cluster_id"] = cluster_id
        if safe_int(cluster.get("size")) < args.min_cluster_size:
            continue
        if safe_int(cluster.get("council_count"), 1) < args.min_councils:
            continue
        if not reps_by_cluster.get(cluster_id):
            continue
        eligible.append(cluster)
    eligible.sort(key=lambda row: (-safe_int(row.get("size")), row["cluster_id"]))
    if args.limit:
        eligible = eligible[:args.limit]

    prompts_path = output_dir / "llm_prompts.jsonl"
    results_path = output_dir / "llm_cluster_labels.jsonl"
    csv_path = output_dir / "llm_cluster_labels.csv"
    errors_path = output_dir / "llm_labeling_errors.jsonl"
    summary_output = output_dir / "llm_labeling_summary.json"

    prompt_rows = []
    for cluster in eligible:
        cluster_id = cluster["cluster_id"]
        prompt = build_prompt(
            cluster,
            reps_by_cluster[cluster_id][: args.max_representatives],
            args.max_text_chars,
        )
        prompt_rows.append({"cluster_id": cluster_id, "prompt": prompt})
    prompts_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prompt_rows),
        encoding="utf-8",
    )
    if args.overwrite and not args.prepare_only:
        for path in (results_path, errors_path, csv_path, summary_output):
            if path.exists():
                path.unlink()
    for path in (results_path, errors_path):
        if not path.exists():
            path.write_text("", encoding="utf-8")

    if args.prepare_only:
        print(f"프롬프트 {len(prompt_rows):,}개 생성: {prompts_path}")
        return

    completed = load_completed(results_path)
    errors = 0
    started = time.time()
    cluster_lookup = {row["cluster_id"]: row for row in eligible}
    for index, prompt_row in enumerate(prompt_rows, start=1):
        cluster_id = prompt_row["cluster_id"]
        if cluster_id in completed:
            print(f"[{index}/{len(prompt_rows)}] cluster {cluster_id}: 기존 결과 사용")
            continue
        print(f"[{index}/{len(prompt_rows)}] cluster {cluster_id}: {args.model} 추론", flush=True)
        try:
            raw = call_ollama(
                args.ollama_url, args.model, prompt_row["prompt"],
                args.temperature, args.timeout,
            )
            selected_representatives = reps_by_cluster[cluster_id][: args.max_representatives]
            label = normalize_label(
                raw,
                selected_representatives,
                args.min_dominant_ratio,
            )
            cluster = cluster_lookup[cluster_id]
            item = {
                "cluster_id": cluster_id,
                "issue_id": f"ISSUE_{cluster_id:03d}",
                **label,
                "size": safe_int(cluster.get("size")),
                "mean_similarity": safe_float(cluster.get("mean_similarity")),
                "council_count": safe_int(cluster.get("council_count")),
                "first_date": cluster.get("first_date", ""),
                "last_date": cluster.get("last_date", ""),
                "provisional_label": cluster.get("provisional_label", ""),
                "model": args.model,
                "script_version": SCRIPT_VERSION,
            }
            append_jsonl(results_path, item)
            completed[cluster_id] = item
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as error:
            errors += 1
            append_jsonl(errors_path, {
                "cluster_id": cluster_id,
                "error": f"{type(error).__name__}: {error}",
                "model": args.model,
            })
            print(f"  오류: {error}", flush=True)

        ordered = [completed[key] for key in sorted(completed) if key in cluster_lookup]
        write_csv(csv_path, ordered)

    ordered = [completed[key] for key in sorted(completed) if key in cluster_lookup]
    write_csv(csv_path, ordered)
    quality_counts = defaultdict(int)
    review_counts = defaultdict(int)
    validation_flag_counts = defaultdict(int)
    for row in ordered:
        quality_counts[row["cluster_quality"]] += 1
        review_counts[row["review_status"]] += 1
        for flag in row.get("validation_flags") or []:
            validation_flag_counts[flag] += 1
    atomic_write_json(summary_output, {
        "script_version": SCRIPT_VERSION,
        "security": "localhost Ollama만 사용하며 회의록 데이터 외부 전송 없음",
        "model": args.model,
        "eligible_cluster_count": len(eligible),
        "completed_cluster_count": len(ordered),
        "error_count_this_session": errors,
        "quality_counts": dict(quality_counts),
        "review_counts": dict(review_counts),
        "validation_flag_counts": dict(validation_flag_counts),
        "min_dominant_ratio": args.min_dominant_ratio,
        "elapsed_seconds_this_session": round(time.time() - started, 3),
        "generated_files": [str(prompts_path), str(results_path), str(csv_path), str(errors_path)],
    })
    print("\n완료")
    print(f"라벨 CSV: {csv_path}")
    print(f"요약: {summary_output}")


if __name__ == "__main__":
    main()
