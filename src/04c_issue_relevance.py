#!/usr/bin/env python3
"""04b 정책 후보 매니페스트에서 실제 정책 논의 후보를 재선별한다.

04b는 정책 발언을 놓치지 않는 고재현율 후보 생성 단계다. 이 스크립트는
회의 절차·출석자 명단·의사보고·웹 UI 잔여물을 강하게 제외하고, 정책 내용과
행동 표현을 점수화하여 임베딩 직전의 고정밀 후보 매니페스트를 만든다.

입력 예시
    data/index/arctic_ko_issue_test_v21/chunk_metadata.parquet

출력
    <output-dir>/chunk_metadata.parquet       최종 선택 후보, embedding_row 재부여
    <output-dir>/scored_metadata.parquet      전체 판정 결과
    <report-dir>/issue_relevance_summary.json
    <report-dir>/issue_relevance_samples.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as error:
    raise SystemExit("pyarrow가 필요합니다. 실행 전: pip install pyarrow") from error


SCRIPT_VERSION = "2026-10-16-issue-relevance-v1.1"

REQUIRED_COLUMNS = [
    "embedding_row", "chunk_id", "council", "meeting_date", "meeting_year",
    "chunk_type", "agenda_title", "issue_text", "chunk_text",
]

GENERIC_AGENDA_RE = re.compile(
    r"^(?:심사된\s*안건|부의\s*안건|부의된\s*안건|상정된\s*안건|"
    r"보고사항|업무보고(?:의\s*건)?|의사일정(?:\s*\([^)]*\))?|"
    r"가\.?\s*[^;]{0,40}\s*소관)$"
)
INTERNAL_AGENDA_RE = re.compile(
    r"회기\s*(?:결정|운영\s*계획)|의사일정.*협의|회의록\s*서명의원|위원장\s*선임|"
    r"부위원장\s*선임|의석\s*배정|휴회의\s*건|(?:의장|부의장).*선거의\s*건|"
    r"위원\s*(?:선임|보임)의\s*건"
)
NONPOLICY_INTERNAL_AGENDA_RE = re.compile(
    r"인사청문|의회사무처|의원당선인\s*의정연수|"
    r"위원장\s*및\s*부위원장\s*선임|의석\s*배정|"
    r"(?:의장|부의장)\s*선거|위원\s*선임의\s*건"
)
UI_NOISE_RE = re.compile(
    r"본문내\s*검색|환경설정메뉴|퀵메뉴|Quick\s*Index|뷰어다운로드|"
    r"동일회기\s*회의록|의회용어사전|안건별\s*보기|회의록검색|"
    r"본문\s*바로가기|하단주소\s*바로가기"
)
DOCUMENT_HEADER_RE = re.compile(
    r"제\s*\d+회.{0,100}(?:본회의|위원회)?.{0,40}회의\s*록\s*제?\s*\d*호|"
    r"(?:시|도)의회사무처.{0,100}(?:일자|의사일정)|"
    r"장소\s+[^\n]{0,40}\s+의사일정"
)
ADMIN_REPORT_START_RE = re.compile(
    r"(?:의사담당관|의사관)[^\n.]{0,80}(?:의사보고|보고사항|보고를\s*드리)|"
    r"오늘\s+본회의\s+보고사항|의사보고\s*드리겠습니다|"
    r"먼저\s+(?:이번\s+)?(?:임시회|정례회)\s+집회\s*경위|"
    r"먼저\s+안건\s+접수사항|의안\s+접수\s+현황|"
    r"전체\s+의사일정을?\s+설명"
)
REPORT_OMITTED_RE = re.compile(
    r"(?:업무|주요업무)?보고[^\n]{0,120}(?:부록에\s*실음|회의록에만\s*등재|보고는\s*생략)"
)
ROSTER_START_RE = re.compile(r"^\s*(?::\s*)?\d+인\s*\((?:시청|교육청|도청)\)")
ROSTER_ROLE_RE = re.compile(
    r"시장|도지사|교육감|부시장|부지사|실장|국장|본부장|위원장|과장|담당관|"
    r"원장|소장|처장|의장|부의장|전문위원|감사위원장|청장"
)

ACTION_PATTERNS = [
    re.compile(pattern)
    for pattern in (
        r"문제", r"부족", r"필요", r"개선", r"확대", r"축소", r"강화",
        r"완화", r"지원", r"대책", r"해결", r"요구", r"제안", r"검토",
        r"추진", r"도입", r"폐지", r"개정", r"구축", r"확보", r"예방",
        r"보호", r"감축", r"활성화", r"정비", r"회복", r"육성", r"전환",
    )
]
DOMAIN_PATTERNS = [
    re.compile(pattern)
    for pattern in (
        r"인구|저출생|출산|난임", r"보육|아동|돌봄", r"청년", r"주거|주택",
        r"교육|학교|학생", r"의료|병원|건강|보건", r"복지|장애인|노인",
        r"교통|버스|철도|도로", r"안전|재난|소방", r"기후|환경|탄소",
        r"산업|기업|경제|창업", r"일자리|고용|노동", r"농업|농촌|축산|수산",
        r"관광|문화|체육", r"원전|에너지|전력", r"도시|지역개발|균형발전",
        r"예산|재정", r"조례|법률|제도", r"공공|행정서비스", r"감염병|정신건강",
    )
]
PROCEDURAL_PATTERNS = [
    re.compile(pattern)
    for pattern in (
        r"의사일정\s*제\s*\d+항", r"(?:안건|건)을?\s+상정", r"질의.*종결",
        r"토론.*종결", r"이의가?\s*(?:없|있)", r"원안대로\s+의결|원안과\s+같이\s+의결",
        r"가결되었음을", r"정회를\s+선포|산회를\s+선포", r"회의를\s+(?:시작|개회)",
        r"회기\s*결정", r"서명의원\s*(?:선출|선임)", r"위원장\s*선임|부위원장\s*선임",
        r"의사봉\s*\d+타", r"찬반\s*의원\s*성명", r"성원이\s+되었으므로",
        r"질의하실\s+위원", r"토론하실\s+위원", r"회의\s+진행",
    )
]
ADMIN_PATTERNS = [
    re.compile(pattern)
    for pattern in (
        r"의사보고", r"보고사항", r"집회\s*공고|집회공고", r"안건\s*접수|의안\s*접수",
        r"부의된\s*안건", r"방청석|방청을", r"불참\s*사항|불참공무원",
        r"간부(?:공무원)?\s*(?:소개|인사)", r"신임\s*간부", r"인사\s*발령|인사발령",
        r"출석\s*및\s*참석공무원", r"참석하지\s*못", r"전자회의시스템",
    )
]
GREETING_PATTERNS = [
    re.compile(pattern)
    for pattern in (
        r"새해\s+복\s+많이", r"건강과\s+행복", r"감사의\s+말씀", r"깊은\s+감사",
        r"진심으로\s+환영", r"만나\s+뵙게\s+되어", r"노고에\s+감사",
    )
]

SCORE_FIELDS = [
    pa.field("source_embedding_row", pa.int64()),
    pa.field("relevance_score", pa.int16()),
    pa.field("relevance_reason", pa.string()),
    pa.field("distinct_action_count", pa.int16()),
    pa.field("distinct_domain_count", pa.int16()),
    pa.field("procedural_signal_count", pa.int16()),
    pa.field("admin_signal_count", pa.int16()),
    pa.field("greeting_signal_count", pa.int16()),
    pa.field("roster_role_count", pa.int16()),
    pa.field("sentence_count", pa.int16()),
    pa.field("numbered_agenda_count", pa.int16()),
    pa.field("substantive_agenda", pa.int8()),
    pa.field("ui_noise_detected", pa.int8()),
    pa.field("document_header_detected", pa.int8()),
    pa.field("roster_detected", pa.int8()),
    pa.field("report_omitted_detected", pa.int8()),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="정책 후보의 내용 관련성 재평가")
    parser.add_argument(
        "--input",
        default="data/index/arctic_ko_issue_test_v21/chunk_metadata.parquet",
    )
    parser.add_argument("--output-dir", default="data/index/arctic_ko_issue_relevance_test")
    parser.add_argument("--report-dir", default="outputs/issue_relevance_test")
    parser.add_argument("--min-score", type=int, default=3)
    parser.add_argument("--read-batch-size", type=int, default=4096)
    parser.add_argument("--selected-samples-per-council-year", type=int, default=3)
    parser.add_argument("--excluded-samples-per-council", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="진단용 입력 행 제한. 0이면 전체")
    return parser.parse_args()


def safe_text(value) -> str:
    return "" if value is None else str(value)


def atomic_write_json(path: Path, data: dict) -> None:
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def extract_issue_parts(issue_text: str, agenda_title: str) -> tuple[str, str]:
    text = safe_text(issue_text).strip()
    agenda = safe_text(agenda_title).strip()
    body = text
    match = re.search(r"\[안건\]\s*(.*?)\s*\[정책본문\]\s*(.*)", text, re.DOTALL)
    if match:
        agenda = match.group(1).strip() or agenda
        body = match.group(2).strip()
    elif "[정책본문]" in text:
        body = text.split("[정책본문]", 1)[1].strip()
    return agenda, body


def count_distinct(patterns: list[re.Pattern], text: str) -> int:
    return sum(1 for pattern in patterns if pattern.search(text))


def count_occurrences(patterns: list[re.Pattern], text: str) -> int:
    return sum(len(pattern.findall(text)) for pattern in patterns)


def agenda_is_substantive(agenda: str) -> bool:
    segments = [re.sub(r"^\s*\d+[.)]?\s*", "", item).strip() for item in agenda.split(";")]
    useful = [item for item in segments if len(item) >= 8 and not GENERIC_AGENDA_RE.fullmatch(item)]
    return bool(useful) and not all(INTERNAL_AGENDA_RE.search(item) for item in useful)


def score_row(row: dict, min_score: int) -> dict:
    issue_text = safe_text(row.get("issue_text"))
    agenda, body = extract_issue_parts(issue_text, safe_text(row.get("agenda_title")))
    combined = f"{agenda} {body}".strip()
    first_700 = body[:700]

    action_count = count_distinct(ACTION_PATTERNS, combined)
    domain_count = count_distinct(DOMAIN_PATTERNS, combined)
    procedural_count = count_occurrences(PROCEDURAL_PATTERNS, combined)
    admin_count = count_occurrences(ADMIN_PATTERNS, combined)
    greeting_count = count_occurrences(GREETING_PATTERNS, body)
    roster_role_count = len(ROSTER_ROLE_RE.findall(body))
    sentence_count = len(re.findall(r"[.!?]|(?:습니다|합니다|됩니다|했습니다)(?:\s|$)", body))
    numbered_agenda_count = len(re.findall(r"(?:^|\s)\d+[.)]\s*[가-힣A-Za-z]", body))
    substantive_agenda = int(agenda_is_substantive(agenda))
    generic_agenda = int(bool(agenda) and not substantive_agenda)
    ui_noise = int(bool(UI_NOISE_RE.search(combined)))
    document_header = int(
        bool(DOCUMENT_HEADER_RE.search(first_700))
        and bool(re.search(r"의사일정|상정된\s*안건|심사된\s*안건", first_700))
    )
    roster = int(
        bool(ROSTER_START_RE.search(body))
        or "출석 및 참석공무원" in body[:500]
        or (
            len(ROSTER_ROLE_RE.findall(body[:600])) >= 8
            and len(re.findall(r"[.!?]|습니다|합니다|됩니다", body[:600])) <= 4
        )
        or (roster_role_count >= 12 and sentence_count <= 8)
    )
    report_omitted = int(bool(REPORT_OMITTED_RE.search(body)))
    admin_report = bool(ADMIN_REPORT_START_RE.search(body[:350]))
    internal_agenda = bool(INTERNAL_AGENDA_RE.search(agenda))
    nonpolicy_internal_agenda = bool(NONPOLICY_INTERNAL_AGENDA_RE.search(agenda))
    roster_intro = bool(
        re.search(r"(?:간부\s*소개|먼저\s+[^.]{0,40}(?:실장|국장|본부장)입니다)", body[:500])
        and roster_role_count >= 5
        and action_count <= 1
    )
    closing_lead = bool(
        re.search(
            r"^\s*(?:그렇군요[.\s]*잘\s*알았습니다|그래요[.\s]*수고|"
            r"금일\s+보고된|질의하실\s+위원님\s+안\s+계|이상으로)",
            body[:180],
        )
    )
    closing_only = bool(
        (
            re.search(r"질의를\s+종결|토론은\s+생략하고\s+의결|회의를\s+모두\s+마칩", body[:450])
            and procedural_count >= 4
        )
        or (
            closing_lead
            and re.search(r"(?:질의|업무보고|회의|의사일정).{0,80}(?:종결|마치)", body[:550])
        )
    )

    score = min(action_count, 5) + min(domain_count, 4)
    score += 2 * substantive_agenda
    score += 2 if action_count >= 2 and domain_count >= 1 else 0
    score += 1 if sentence_count >= 4 else 0
    score += 1 if len(body) >= 500 else 0
    score -= min(procedural_count, 5)
    score -= min(admin_count, 4)
    score -= min(greeting_count, 2)
    score -= generic_agenda
    score -= 3 if numbered_agenda_count >= 4 else 0
    score -= 3 if roster_role_count >= 6 else 0

    if ui_noise:
        reason = "excluded_ui_noise"
    elif document_header:
        reason = "excluded_document_header"
    elif roster or roster_intro:
        reason = "excluded_roster"
    elif admin_report:
        reason = "excluded_admin_report"
    elif report_omitted and action_count <= 2 and domain_count <= 2:
        reason = "excluded_report_omitted"
    elif nonpolicy_internal_agenda:
        reason = "excluded_internal_agenda"
    elif closing_only:
        reason = "excluded_procedural_only"
    elif internal_agenda and procedural_count >= 2 and action_count <= 2:
        reason = "excluded_internal_procedure"
    elif procedural_count >= 5 and action_count <= 2 and domain_count <= 2:
        reason = "excluded_procedural_only"
    elif score < min_score:
        reason = "excluded_low_relevance_score"
    else:
        reason = "selected"

    return {
        "relevance_score": int(score),
        "relevance_reason": reason,
        "distinct_action_count": action_count,
        "distinct_domain_count": domain_count,
        "procedural_signal_count": procedural_count,
        "admin_signal_count": admin_count,
        "greeting_signal_count": greeting_count,
        "roster_role_count": roster_role_count,
        "sentence_count": sentence_count,
        "numbered_agenda_count": numbered_agenda_count,
        "substantive_agenda": substantive_agenda,
        "ui_noise_detected": ui_noise,
        "document_header_detected": document_header,
        "roster_detected": roster,
        "report_omitted_detected": report_omitted,
    }


class HashSampler:
    def __init__(self) -> None:
        self.rows: dict[tuple, list[tuple[int, dict]]] = defaultdict(list)

    def add(self, key: tuple, limit: int, chunk_id: str, row: dict) -> None:
        if limit <= 0:
            return
        rank = int(hashlib.sha256(f"{key}|{chunk_id}".encode("utf-8")).hexdigest()[:16], 16)
        bucket = self.rows[key]
        bucket.append((rank, row))
        bucket.sort(key=lambda item: item[0])
        if len(bucket) > limit:
            bucket.pop()

    def values(self) -> list[dict]:
        result = []
        for key in sorted(self.rows, key=lambda item: tuple(str(value) for value in item)):
            result.extend(row for _, row in self.rows[key])
        return result


def percentile_from_counter(counter: Counter, fraction: float) -> int | None:
    total = sum(counter.values())
    if not total:
        return None
    target = max(1, int(total * fraction + 0.999999))
    cumulative = 0
    for score in sorted(counter):
        cumulative += counter[score]
        if cumulative >= target:
            return int(score)
    return int(max(counter))


def write_samples(path: Path, rows: list[dict]) -> None:
    fields = [
        "relevance_reason", "relevance_score", "chunk_id", "council", "meeting_date",
        "chunk_type", "agenda_title", "speaker_label", "distinct_action_count",
        "distinct_domain_count", "procedural_signal_count", "admin_signal_count",
        "roster_role_count", "sentence_count", "substantive_agenda",
        "ui_noise_detected", "document_header_detected", "roster_detected",
        "report_omitted_detected", "issue_text_preview",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.read_batch_size < 1 or args.limit < 0:
        raise SystemExit("배치 크기는 1 이상, limit은 0 이상이어야 합니다.")

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"입력 파일이 없습니다: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    selected_path = output_dir / "chunk_metadata.parquet"
    scored_path = output_dir / "scored_metadata.parquet"
    selected_tmp = output_dir / "chunk_metadata.tmp.parquet"
    scored_tmp = output_dir / "scored_metadata.tmp.parquet"
    summary_path = report_dir / "issue_relevance_summary.json"
    samples_path = report_dir / "issue_relevance_samples.csv"
    for path in (selected_tmp, scored_tmp):
        if path.exists():
            path.unlink()
    if selected_path.exists() or scored_path.exists():
        raise SystemExit("출력 Parquet가 이미 있습니다. 다른 --output-dir을 사용하세요.")

    source = pq.ParquetFile(input_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in source.schema_arrow.names]
    if missing:
        raise SystemExit("입력 Parquet 필수 컬럼 누락: " + ", ".join(missing))
    duplicate = [field.name for field in SCORE_FIELDS if field.name in source.schema_arrow.names]
    if duplicate:
        raise SystemExit("이미 관련성 점수 컬럼이 있습니다: " + ", ".join(duplicate))

    output_schema = pa.schema([*source.schema_arrow, *SCORE_FIELDS])
    selected_writer = None
    scored_writer = None
    counters = Counter()
    selected_councils = Counter()
    selected_council_years = Counter()
    score_counts = Counter()
    sampler = HashSampler()
    started = time.time()
    selected_count = 0
    stop = False

    try:
        for batch in source.iter_batches(batch_size=args.read_batch_size):
            selected_rows = []
            scored_rows = []
            for row in batch.to_pylist():
                if args.limit and counters["input"] >= args.limit:
                    stop = True
                    break
                counters["input"] += 1
                diagnostics = score_row(row, args.min_score)
                reason = diagnostics["relevance_reason"]
                counters[reason] += 1
                score_counts[diagnostics["relevance_score"]] += 1

                original_embedding_row = int(row.get("embedding_row") or 0)
                scored_row = {
                    **row,
                    "source_embedding_row": original_embedding_row,
                    **diagnostics,
                }
                scored_rows.append(scored_row)

                council = safe_text(row.get("council")) or "unknown"
                year = safe_text(row.get("meeting_year")) or safe_text(row.get("meeting_date"))[:4]
                sample_row = {
                    **row,
                    **diagnostics,
                    "issue_text_preview": re.sub(r"\s+", " ", safe_text(row.get("issue_text")))[:500],
                }
                if reason == "selected":
                    sample_key = ("selected", council, year)
                    sample_limit = args.selected_samples_per_council_year
                else:
                    sample_key = (reason, council)
                    sample_limit = args.excluded_samples_per_council
                sampler.add(sample_key, sample_limit, safe_text(row.get("chunk_id")), sample_row)

                if reason != "selected":
                    continue
                selected_row = dict(scored_row)
                selected_row["embedding_row"] = selected_count
                selected_rows.append(selected_row)
                selected_count += 1
                selected_councils[council] += 1
                selected_council_years[(council, year)] += 1

            if scored_rows:
                scored_table = pa.Table.from_pylist(scored_rows, schema=output_schema)
                if scored_writer is None:
                    scored_writer = pq.ParquetWriter(scored_tmp, output_schema, compression="zstd")
                scored_writer.write_table(scored_table)
            if selected_rows:
                selected_table = pa.Table.from_pylist(selected_rows, schema=output_schema)
                if selected_writer is None:
                    selected_writer = pq.ParquetWriter(selected_tmp, output_schema, compression="zstd")
                selected_writer.write_table(selected_table)
            if counters["input"] % 10000 < len(batch):
                print(
                    f"관련성 평가: {counters['input']:,}/{source.metadata.num_rows:,} "
                    f"| 선택 {selected_count:,}",
                    flush=True,
                )
            if stop:
                break
    finally:
        if selected_writer is not None:
            selected_writer.close()
        if scored_writer is not None:
            scored_writer.close()

    if not scored_tmp.is_file():
        raise SystemExit("평가 결과가 생성되지 않았습니다.")
    if not selected_tmp.is_file() or selected_count == 0:
        os.replace(scored_tmp, scored_path)
        raise SystemExit("관련성 기준을 통과한 청크가 없습니다. --min-score를 확인하세요.")
    os.replace(scored_tmp, scored_path)
    os.replace(selected_tmp, selected_path)

    sample_rows = sampler.values()
    write_samples(samples_path, sample_rows)
    report = {
        "script_version": SCRIPT_VERSION,
        "input_path": str(input_path),
        "input_row_count": source.metadata.num_rows,
        "scanned_row_count": counters["input"],
        "min_score": args.min_score,
        "selected_count": selected_count,
        "selected_rate": round(selected_count / counters["input"], 6),
        "reason_counts": dict(counters),
        "score_distribution": {str(score): count for score, count in sorted(score_counts.items())},
        "score_percentiles": {
            "p10": percentile_from_counter(score_counts, 0.10),
            "p25": percentile_from_counter(score_counts, 0.25),
            "p50": percentile_from_counter(score_counts, 0.50),
            "p75": percentile_from_counter(score_counts, 0.75),
            "p90": percentile_from_counter(score_counts, 0.90),
        },
        "selected_councils": dict(selected_councils.most_common()),
        "selected_council_years": {
            f"{council}|{year}": count
            for (council, year), count in sorted(selected_council_years.items())
        },
        "sample_count": len(sample_rows),
        "elapsed_seconds": round(time.time() - started, 3),
        "generated_files": [
            str(selected_path), str(scored_path), str(summary_path), str(samples_path),
        ],
    }
    atomic_write_json(summary_path, report)

    print("\n완료")
    print(f"입력: {counters['input']:,}개")
    print(f"선택: {selected_count:,}개 ({selected_count / counters['input'] * 100:.2f}%)")
    print(f"선택 매니페스트: {selected_path}")
    print(f"전체 판정 결과: {scored_path}")
    print(f"요약: {summary_path}")
    print(f"검수 샘플: {samples_path}")


if __name__ == "__main__":
    main()
