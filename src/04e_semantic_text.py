#!/usr/bin/env python3
"""04c 정책 후보에서 군집화용 semantic_text 자식 청크를 만든다.

목적
    원문 근거와 메타데이터는 그대로 보존하면서, 임베딩에 불필요한
    기관·발화자 이름, 인사말, 회의 진행 문장, 보고서 제목 나열을 줄이고,
    여러 정책이 섞인 부모 청크를 짧은 정책 단위 자식 청크로 분할한다.

입력
    data/index/arctic_ko_issue_relevance_test_v11/chunk_metadata.parquet

출력
    <output-dir>/chunk_metadata.parquet          semantic_text 통과 행
    <output-dir>/semantic_scored_metadata.parquet 전체 판정 행
    <report-dir>/semantic_text_summary.json
    <report-dir>/semantic_text_samples.csv

주의
    원본 chunk_text와 issue_text는 삭제하거나 덮어쓰지 않는다. 다음 임베딩은
    이 스크립트가 생성한 semantic_text 컬럼을 사용한다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as error:
    raise SystemExit("pyarrow가 필요합니다. 실행 전: pip install pyarrow") from error


SCRIPT_VERSION = "2026-10-19-semantic-child-v2"

REQUIRED_COLUMNS = {
    "embedding_row", "chunk_id", "council", "meeting_date",
    "agenda_title", "issue_text", "chunk_text",
}

SEMANTIC_FIELDS = [
    pa.field("semantic_source_embedding_row", pa.int64()),
    pa.field("parent_chunk_id", pa.string()),
    pa.field("semantic_unit_index", pa.int16()),
    pa.field("semantic_unit_count", pa.int16()),
    pa.field("semantic_text", pa.string()),
    pa.field("semantic_topic_hint", pa.string()),
    pa.field("semantic_char_count", pa.int32()),
    pa.field("semantic_body_char_count", pa.int32()),
    pa.field("semantic_removed_line_count", pa.int16()),
    pa.field("semantic_removed_role_count", pa.int16()),
    pa.field("semantic_policy_signal_count", pa.int16()),
    pa.field("semantic_reason", pa.string()),
    pa.field("semantic_version", pa.string()),
]

GENERIC_AGENDA_RE = re.compile(
    r"^(?:심사된\s*안건|부의(?:된)?\s*안건|상정된\s*안건|보고사항|"
    r"업무보고(?:의\s*건)?|주요업무보고(?:의\s*건)?|의사일정|"
    r"휴회의\s*건|회의록\s*서명의원|질의[·ㆍ\s]*답변|가\.?\s*[^;]{0,50}\s*소관)$"
)
PROCEDURAL_AGENDA_RE = re.compile(
    r"회기\s*(?:결정|운영)|의사일정|휴회의\s*건|서명의원|위원장\s*선임|"
    r"부위원장\s*선임|의석\s*배정|회의규칙|심사된\s*안건|보고사항"
)
POLICY_TOKEN_RE = re.compile(
    r"인구|저출생|출산|난임|보육|아동|돌봄|청년|주거|주택|교육|학교|학생|"
    r"의료|병원|건강|보건|복지|장애|노인|교통|버스|철도|도로|안전|재난|"
    r"소방|기후|환경|탄소|산업|기업|경제|창업|일자리|고용|노동|농업|농촌|"
    r"축산|수산|관광|문화|체육|에너지|전력|도시|지역개발|균형발전|예산|"
    r"재정|조례|법률|제도|공공|감염병|정신건강|지원|개선|확대|강화|도입|"
    r"구축|예방|보호|감축|활성화|육성|전환|대책|계획|사업"
)
ACTION_TOKEN_RE = re.compile(
    r"지원|개선|확대|축소|강화|완화|도입|구축|확보|예방|보호|감축|"
    r"활성화|육성|전환|대책|해결|제안|요구|개정|폐지|추진|시행|운영|"
    r"조성|정비|발굴|제공|설치|지원하|추진하|시행하|필요하"
)
DOMAIN_TOKEN_RE = re.compile(
    r"인구|저출생|출산|난임|보육|아동|돌봄|청년|주거|주택|교육|학교|학생|"
    r"의료|병원|건강|보건|복지|장애|노인|교통|버스|철도|도로|안전|재난|"
    r"소방|기후|환경|탄소|산업|기업|경제|창업|일자리|고용|노동|농업|농촌|"
    r"축산|수산|관광|문화|체육|에너지|전력|도시|지역개발|균형발전|감염병|정신건강"
)

ROLE_WORDS = (
    "의장|부의장|위원장|부위원장|위원|의원|도지사|시장|교육감|부시장|부지사|"
    "실장|국장|본부장|과장|담당관|원장|소장|처장|청장|사장|대표이사|"
    "전문위원|직무대리|직무대행|의사관|주무관|팀장|정책관|기획관|"
    "관장|센터장|부장|단장|교장|교육장"
)
ROLE_NAME_RE = re.compile(
    rf"(?:{ROLE_WORDS})\s*(?:후보자\s*)?[가-힣]{{2,4}}"
)
ROLE_ONLY_RE = re.compile(rf"^(?:[가-힣A-Za-z0-9·ㆍ()\s]{{0,35}})?(?:{ROLE_WORDS})입니다[.!]?$")

STAGE_DIRECTION_RE = re.compile(
    r"\([^)]{0,80}(?:박수|웃음|장내|청취불능|자료|영상|마이크|의사봉|일동|기립|착석)[^)]*\)"
)
HTML_TAG_RE = re.compile(r"<[^>]+>")
SPONSOR_LIST_RE = re.compile(
    r"\([^)]{0,500}(?:대표발의|공동발의|의원\s*발의|의원\s*찬성)[^)]*\)"
)
ORG_NAME_RE = re.compile(
    r"[가-힣A-Za-z0-9·ㆍ]{2,30}(?:개발공사|도시공사|관광공사|교통공사|"
    r"시설공단|문화재단|복지재단|관광재단|연구원|진흥원|사회서비스원|"
    r"테크노파크|교육지원청|교육청|특별자치도의회|광역시의회|도의회|시의회)"
)

DROP_SENTENCE_RE = re.compile(
    r"^(?:"
    r"안녕하십니까[?.!]?|반갑습니다[.!]?|감사합니다[.!]?|수고하셨습니다[.!]?|"
    r"질의하실\s+(?:위원|의원)님|더\s+질의하실|다음은\s+.+(?:발언|질의|답변)|"
    r".+(?:나오셔서|발언하여|질의하여|답변하여)\s*주시기\s*바랍니다|"
    r".+자리로\s*돌아가|회의를\s*(?:시작|개회|속개|정회|산회)|"
    r"(?:정회|산회)를\s*선포|질의[·ㆍ\s]*답변을\s*(?:시작|종결)|"
    r"이상으로\s+.+(?:마치겠습니다|종결하겠습니다)|"
    r"의사일정\s*제?\s*\d+항|.+안건을\s*상정|이의가\s*없으므로|"
    r"원안대로\s*(?:가결|의결)|가결되었음을\s*선포"
    r"|다음은\s+질의.{0,80}(?:진행|순서)|질의.{0,80}(?:마치|종결)"
    r"|원활한\s+회의\s+진행|회의\s+준비와\s+(?:중식|자료\s*검토)"
    r"|간부\s*공무원을?\s*소개|보고를\s*받도록\s*하겠습니다"
    r"|본질의.{0,80}보충질의.{0,80}(?:제한|실시)"
    r"|거수로\s*신청|잠시.{0,40}정회"
    r")"
)
ACK_ONLY_RE = re.compile(
    r"^(?:예|네|아니요|그렇습니다|알겠습니다|맞습니다|그렇죠|그러네요|"
    r"입니다|이상입니다)[,.!?\s]*$"
)
GREETING_RE = re.compile(
    r"깊은\s*감사|감사의\s*말씀|진심으로\s*환영|노고에\s*감사|"
    r"새해\s*복\s*많이|건강과\s*행복|바쁘신\s*의정활동"
)
PROCEDURAL_RE = re.compile(
    r"질의하실|발언하여\s*주시|답변하여\s*주시|자리에\s*앉아|"
    r"회의를\s*(?:개회|속개|정회|산회)|상정하겠습니다|"
    r"가결되었음을|이의가\s*없으므로|수고하셨습니다"
)
REPORT_LIST_RE = re.compile(r"심사보고서|검토보고서|동의안|일부개정조례안|예산안")
MEETING_ADMIN_RE = re.compile(
    r"회기|의사일정|개의|개회|속개|정회|산회|휴회|회의규칙|"
    r"업무\s*(?:추진)?계획\s*보고|조례안\s*심사|안건을\s*상정|"
    r"질의하실|답변하여\s*주시|이의가\s*없으므로|가결되었음을"
)
MEETING_SCHEDULE_RE = re.compile(
    r"\d{1,2}월\s*\d{1,2}일|\d{1,2}일간의\s*일정|"
    r"보고받으시|심사하시|개의하여|세부\s*의사일정"
)
TRANSITION_RE = re.compile(
    r"^(?:그다음|다음으로|다음은|이어서|그리고|또한|한편|두\s*번째|세\s*번째|셋째|마지막으로|끝으로|아울러)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="군집화용 정책 semantic_text 생성")
    parser.add_argument(
        "--input",
        default="data/index/arctic_ko_issue_relevance_test_v11/chunk_metadata.parquet",
    )
    parser.add_argument("--output-dir", default="data/index/arctic_ko_semantic_test_v2")
    parser.add_argument("--report-dir", default="outputs/semantic_text_test_v2")
    parser.add_argument("--min-chars", type=int, default=120)
    parser.add_argument("--target-chars", type=int, default=600)
    parser.add_argument("--max-chars", type=int, default=900)
    parser.add_argument("--read-batch-size", type=int, default=4096)
    parser.add_argument("--selected-samples-per-council-year", type=int, default=2)
    parser.add_argument("--excluded-samples-per-reason", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0, help="진단용 입력 행 제한. 0이면 전체")
    return parser.parse_args()


def safe_text(value) -> str:
    return "" if value is None else str(value)


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", html.unescape(safe_text(value)))
    text = HTML_TAG_RE.sub(" ", text)
    text = text.replace("\u200b", " ").replace("\ufeff", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_issue_parts(issue_text: str, agenda_title: str) -> tuple[str, str]:
    text = normalize_text(issue_text)
    agenda = normalize_text(agenda_title)
    match = re.search(r"\[안건\]\s*(.*?)\s*\[정책본문\]\s*(.*)", text, re.DOTALL)
    if match:
        agenda = match.group(1).strip() or agenda
        body = match.group(2).strip()
    elif "[정책본문]" in text:
        body = text.split("[정책본문]", 1)[1].strip()
    else:
        body = text
    return agenda, body


def clean_topic_hint(agenda: str) -> str:
    text = SPONSOR_LIST_RE.sub(" ", normalize_text(agenda))
    text = re.sub(r"^\s*(?:제?\s*\d+\s*[.)항]?|[가-하]\.)\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip(" ;,.-")
    if not text or GENERIC_AGENDA_RE.fullmatch(text):
        return ""
    if len(re.findall(r"(?:^|;)\s*\d+[.)]?\s*", text)) >= 2 or text.count(";") >= 2:
        return ""
    if "?" in text or re.search(r"(?:알고|말씀|생각)\s*(?:있습니다|드립니다)", text):
        return ""
    if re.search(r"소관.{0,40}(?:주요)?업무.{0,30}(?:보고|추진계획)", text):
        return ""
    if re.match(r"^(?:예|네|그렇습니다|알겠습니다|맞습니다)[,.!\s]", text):
        return ""
    if PROCEDURAL_AGENDA_RE.search(text) and not POLICY_TOKEN_RE.search(text):
        return ""
    if len(text) < 6:
        return ""
    if text.count(";") >= 2 and not POLICY_TOKEN_RE.search(text):
        return ""
    if not re.search(r"조례|계획|정책|사업|예산|지원|대책|보고|동의안|건의안|결의안", text):
        return ""
    return text[:220].rstrip()


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\n+", "\n", text)
    parts = re.split(r"\n|(?<=[.!?])\s+|(?<=[다요죠까])\s+(?=[가-힣A-Z0-9\[])", text)
    return [re.sub(r"\s+", " ", part).strip() for part in parts if part.strip()]


def exact_speaker_values(row: dict) -> list[str]:
    values = []
    for column in ("speaker_label", "speaker_name"):
        value = normalize_text(row.get(column, ""))
        if len(value) >= 2:
            values.append(value)
    return sorted(set(values), key=len, reverse=True)


def clean_body(row: dict, body: str) -> tuple[str, int, int]:
    text = STAGE_DIRECTION_RE.sub(" ", normalize_text(body))
    exact_removed = 0
    for value in exact_speaker_values(row):
        exact_removed += text.count(value)
        text = text.replace(value, " ")

    role_matches = ROLE_NAME_RE.findall(text)
    text = ROLE_NAME_RE.sub(" ", text)
    text = ORG_NAME_RE.sub(" <기관> ", text)
    kept = []
    removed_lines = 0
    for sentence in split_sentences(text):
        sentence = re.sub(r"^[○●◎◇◆□■▷▶△▲※]+\s*", "", sentence).strip()
        sentence = re.sub(r"^(?:예|네)[,.]\s*", "", sentence).strip()
        sentence = re.sub(r"\s+", " ", sentence)
        if not sentence:
            continue
        if ACK_ONLY_RE.fullmatch(sentence) or (len(sentence) <= 180 and DROP_SENTENCE_RE.search(sentence)):
            removed_lines += 1
            continue
        if len(sentence) <= 140 and GREETING_RE.search(sentence) and not ACTION_TOKEN_RE.search(sentence):
            removed_lines += 1
            continue
        if len(sentence) <= 80 and ROLE_ONLY_RE.fullmatch(sentence):
            removed_lines += 1
            continue
        if len(sentence) <= 40 and not re.search(r"[가-힣A-Za-z]{3,}", sentence):
            removed_lines += 1
            continue
        kept.append(sentence)

    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ ]{2,}", " ", cleaned).strip()
    return cleaned, removed_lines, exact_removed + len(role_matches)


def policy_signal_count(text: str) -> int:
    tokens = set(POLICY_TOKEN_RE.findall(text))
    actions = set(ACTION_TOKEN_RE.findall(text))
    return len(tokens | actions)


def split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    """너무 긴 한 문장을 쉼표 경계 우선으로 안전하게 나눈다."""
    if len(sentence) <= max_chars:
        return [sentence]

    clauses = [part.strip() for part in re.split(r"(?<=[,;:])\s+", sentence) if part.strip()]
    if len(clauses) == 1:
        return [sentence[start:start + max_chars].strip() for start in range(0, len(sentence), max_chars)]

    pieces: list[str] = []
    current = ""
    for clause in clauses:
        if len(clause) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(
                clause[start:start + max_chars].strip()
                for start in range(0, len(clause), max_chars)
            )
            continue
        candidate = f"{current} {clause}".strip()
        if current and len(candidate) > max_chars:
            pieces.append(current)
            current = clause
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def split_semantic_units(
    cleaned_body: str,
    min_chars: int,
    target_chars: int,
    max_chars: int,
) -> list[str]:
    """정책 문장을 약 target_chars 크기의 비중첩 자식 청크로 묶는다."""
    sentences: list[str] = []
    for sentence in split_sentences(cleaned_body):
        sentences.extend(split_long_sentence(sentence, max_chars))
    if not sentences:
        return [""]

    units: list[str] = []
    current: list[str] = []
    current_chars = 0
    for sentence in sentences:
        projected = current_chars + (1 if current else 0) + len(sentence)
        transition_break = bool(
            current
            and current_chars >= max(100, min_chars)
            and TRANSITION_RE.match(sentence)
        )
        size_break = bool(current and projected > max_chars)
        target_break = bool(
            current
            and current_chars >= target_chars
            and len(sentence) >= 80
        )
        if transition_break or size_break or target_break:
            units.append("\n".join(current).strip())
            current = []
            current_chars = 0
        current.append(sentence)
        current_chars += (1 if current_chars else 0) + len(sentence)
    if current:
        units.append("\n".join(current).strip())

    # 마지막 조각이 지나치게 짧으면 앞 조각과 합친다. max_chars는 약간 초과할 수 있다.
    if len(units) >= 2 and len(units[-1]) < max(120, target_chars // 3):
        combined = f"{units[-2]}\n{units[-1]}".strip()
        if len(combined) <= max_chars + 150:
            units[-2:] = [combined]
    return units


def child_chunk_id(parent_chunk_id: str, unit_index: int, unit_text: str) -> str:
    key = f"{parent_chunk_id}|{unit_index}|{unit_text}".encode("utf-8")
    return "sem_" + hashlib.sha1(key).hexdigest()[:24]


def semanticize_units(
    row: dict,
    min_chars: int,
    target_chars: int,
    max_chars: int,
) -> list[dict]:
    agenda, body = extract_issue_parts(row.get("issue_text", ""), row.get("agenda_title", ""))
    topic_hint = clean_topic_hint(agenda)
    cleaned_body, removed_lines, removed_roles = clean_body(row, body)

    numbered_items = len(re.findall(r"(?:^|\n|\s)\d+[.)]\s*[가-힣A-Za-z]", cleaned_body))
    report_titles = len(REPORT_LIST_RE.findall(cleaned_body))
    parent_reason = "excluded_document_list" if numbered_items >= 4 and report_titles >= 4 else ""
    units = split_semantic_units(cleaned_body, min_chars, target_chars, max_chars)
    results = []
    for unit_index, unit in enumerate(units):
        signals = policy_signal_count(f"{topic_hint} {unit}")
        domain_count = len(set(DOMAIN_TOKEN_RE.findall(f"{topic_hint} {unit}")))
        procedural_count = len(PROCEDURAL_RE.findall(unit))
        meeting_admin_count = len(MEETING_ADMIN_RE.findall(unit))
        meeting_schedule_count = len(MEETING_SCHEDULE_RE.findall(unit))
        greeting_count = len(GREETING_RE.findall(unit))
        role_count_remaining = len(ROLE_NAME_RE.findall(unit))
        if parent_reason:
            reason = parent_reason
        elif len(unit) < min_chars:
            reason = "excluded_short_semantic_text"
        elif signals == 0 and procedural_count >= 2:
            reason = "excluded_procedural_semantic_text"
        elif meeting_schedule_count >= 3:
            reason = "excluded_meeting_schedule"
        elif meeting_admin_count >= 3 or (meeting_admin_count >= 2 and domain_count <= 1):
            reason = "excluded_meeting_administration"
        elif greeting_count >= 2 and domain_count == 0:
            reason = "excluded_greeting_semantic_text"
        elif role_count_remaining >= 8 and signals <= 1:
            reason = "excluded_roster_semantic_text"
        else:
            reason = "selected"

        semantic_text = (
            f"[주제] {topic_hint}\n[내용] {unit}" if topic_hint else f"[내용] {unit}"
        ).strip()
        results.append({
            "semantic_unit_index": unit_index,
            "semantic_unit_count": len(units),
            "semantic_text": semantic_text,
            "semantic_topic_hint": topic_hint,
            "semantic_char_count": len(semantic_text),
            "semantic_body_char_count": len(unit),
            "semantic_removed_line_count": removed_lines,
            "semantic_removed_role_count": removed_roles,
            "semantic_policy_signal_count": signals,
            "semantic_reason": reason,
            "semantic_version": SCRIPT_VERSION,
        })
    return results


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


def atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def write_samples(path: Path, rows: list[dict]) -> None:
    fields = [
        "semantic_reason", "chunk_id", "parent_chunk_id",
        "semantic_unit_index", "semantic_unit_count",
        "council", "meeting_date", "meeting_year",
        "chunk_type", "relevance_score", "semantic_char_count",
        "semantic_body_char_count", "semantic_removed_line_count",
        "semantic_removed_role_count", "semantic_policy_signal_count",
        "semantic_topic_hint", "before_preview", "after_preview",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if (
        args.min_chars < 1
        or args.target_chars < args.min_chars
        or args.max_chars < args.target_chars
        or args.read_batch_size < 1
        or args.limit < 0
    ):
        raise SystemExit(
            "길이는 1 <= --min-chars <= --target-chars <= --max-chars, "
            "배치 크기는 1 이상, --limit은 0 이상이어야 합니다."
        )

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"입력 파일이 없습니다: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    selected_path = output_dir / "chunk_metadata.parquet"
    scored_path = output_dir / "semantic_scored_metadata.parquet"
    selected_tmp = output_dir / "chunk_metadata.tmp.parquet"
    scored_tmp = output_dir / "semantic_scored_metadata.tmp.parquet"
    summary_path = report_dir / "semantic_text_summary.json"
    samples_path = report_dir / "semantic_text_samples.csv"
    for path in (selected_tmp, scored_tmp):
        if path.exists():
            path.unlink()
    if selected_path.exists() or scored_path.exists():
        raise SystemExit("출력 Parquet가 이미 있습니다. 다른 --output-dir을 사용하세요.")

    source = pq.ParquetFile(input_path)
    missing = REQUIRED_COLUMNS - set(source.schema_arrow.names)
    if missing:
        raise SystemExit("입력 Parquet 필수 컬럼 누락: " + ", ".join(sorted(missing)))
    duplicates = [field.name for field in SEMANTIC_FIELDS if field.name in source.schema_arrow.names]
    if duplicates:
        raise SystemExit("이미 semantic 컬럼이 있습니다: " + ", ".join(duplicates))

    output_schema = pa.schema([*source.schema_arrow, *SEMANTIC_FIELDS])
    selected_writer = None
    scored_writer = None
    counters = Counter()
    selected_councils = Counter()
    selected_council_years = Counter()
    selected_parent_ids = set()
    sampler = HashSampler()
    selected_count = 0
    scored_count = 0
    input_chars = 0
    semantic_chars = 0
    removed_lines_total = 0
    removed_roles_total = 0
    started = time.time()
    stop = False

    try:
        for batch in source.iter_batches(batch_size=args.read_batch_size):
            selected_rows = []
            scored_rows = []
            for row in batch.to_pylist():
                if args.limit and counters["input_parent"] >= args.limit:
                    stop = True
                    break
                counters["input_parent"] += 1
                source_value = row.get("embedding_row")
                source_row = int(source_value) if source_value is not None else -1
                parent_chunk_id = safe_text(row.get("chunk_id"))
                diagnostics_list = semanticize_units(
                    row, args.min_chars, args.target_chars, args.max_chars
                )
                counters["generated_unit"] += len(diagnostics_list)
                input_chars += len(safe_text(row.get("issue_text")))
                removed_lines_total += diagnostics_list[0]["semantic_removed_line_count"]
                removed_roles_total += diagnostics_list[0]["semantic_removed_role_count"]

                council = safe_text(row.get("council")) or "unknown"
                year = safe_text(row.get("meeting_year")) or safe_text(row.get("meeting_date"))[:4]
                for diagnostics in diagnostics_list:
                    reason = diagnostics["semantic_reason"]
                    counters[reason] += 1
                    semantic_chars += diagnostics["semantic_char_count"]
                    child_id = child_chunk_id(
                        parent_chunk_id,
                        diagnostics["semantic_unit_index"],
                        diagnostics["semantic_text"],
                    )
                    scored_row = {
                        **row,
                        "embedding_row": scored_count,
                        "chunk_id": child_id,
                        "semantic_source_embedding_row": source_row,
                        "parent_chunk_id": parent_chunk_id,
                        **diagnostics,
                    }
                    scored_rows.append(scored_row)
                    scored_count += 1

                    sample_row = {
                        **scored_row,
                        "before_preview": re.sub(
                            r"\s+", " ", safe_text(row.get("issue_text"))
                        )[:600],
                        "after_preview": re.sub(
                            r"\s+", " ", diagnostics["semantic_text"]
                        )[:600],
                    }
                    if reason == "selected":
                        key = ("selected", council, year)
                        sample_limit = args.selected_samples_per_council_year
                    else:
                        key = (reason,)
                        sample_limit = args.excluded_samples_per_reason
                    sampler.add(key, sample_limit, child_id, sample_row)

                    if reason != "selected":
                        continue
                    selected_row = dict(scored_row)
                    selected_row["embedding_row"] = selected_count
                    selected_rows.append(selected_row)
                    selected_count += 1
                    selected_parent_ids.add(parent_chunk_id)
                    selected_councils[council] += 1
                    selected_council_years[(council, year)] += 1

            if scored_rows:
                table = pa.Table.from_pylist(scored_rows, schema=output_schema)
                if scored_writer is None:
                    scored_writer = pq.ParquetWriter(scored_tmp, output_schema, compression="zstd")
                scored_writer.write_table(table)
            if selected_rows:
                table = pa.Table.from_pylist(selected_rows, schema=output_schema)
                if selected_writer is None:
                    selected_writer = pq.ParquetWriter(selected_tmp, output_schema, compression="zstd")
                selected_writer.write_table(table)

            if counters["input_parent"] % 10000 < len(batch):
                print(
                    f"semantic 자식 생성: {counters['input_parent']:,}/{source.metadata.num_rows:,} "
                    f"| 생성 {scored_count:,} | 선택 {selected_count:,}",
                    flush=True,
                )
            if stop:
                break
    finally:
        if selected_writer is not None:
            selected_writer.close()
        if scored_writer is not None:
            scored_writer.close()

    if not scored_tmp.exists():
        raise SystemExit("semantic 판정 결과가 생성되지 않았습니다.")
    os.replace(scored_tmp, scored_path)
    if not selected_tmp.exists() or selected_count == 0:
        raise SystemExit("semantic_text 기준을 통과한 청크가 없습니다. --min-chars를 확인하세요.")
    os.replace(selected_tmp, selected_path)

    sample_rows = sampler.values()
    write_samples(samples_path, sample_rows)
    scanned = counters["input_parent"]
    generated_units = counters["generated_unit"]
    summary = {
        "script_version": SCRIPT_VERSION,
        "security": "로컬 전처리만 수행하며 회의록 데이터 외부 전송 없음",
        "input_path": str(input_path),
        "input_row_count": source.metadata.num_rows,
        "scanned_parent_count": scanned,
        "min_chars": args.min_chars,
        "target_chars": args.target_chars,
        "max_chars": args.max_chars,
        "generated_unit_count": generated_units,
        "selected_unit_count": selected_count,
        "selected_parent_count": len(selected_parent_ids),
        "average_units_per_parent": round(generated_units / scanned, 6) if scanned else 0.0,
        "selected_rate": round(selected_count / generated_units, 6) if generated_units else 0.0,
        "reason_counts": {
            key: value
            for key, value in counters.items()
            if key == "selected" or key.startswith("excluded_")
        },
        "input_issue_text_chars": input_chars,
        "output_semantic_text_chars": semantic_chars,
        "semantic_to_input_char_ratio": round(semantic_chars / input_chars, 6) if input_chars else 0.0,
        "removed_line_count": removed_lines_total,
        "removed_role_name_count": removed_roles_total,
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
    atomic_write_json(summary_path, summary)

    print("\n완료")
    print(f"입력 부모 청크: {scanned:,}개")
    print(f"생성 자식 청크: {generated_units:,}개")
    print(f"선택 자식 청크: {selected_count:,}개 ({selected_count / generated_units * 100:.2f}%)")
    print(f"semantic 매니페스트: {selected_path}")
    print(f"전체 판정: {scored_path}")
    print(f"요약: {summary_path}")
    print(f"검수 샘플: {samples_path}")


if __name__ == "__main__":
    main()
