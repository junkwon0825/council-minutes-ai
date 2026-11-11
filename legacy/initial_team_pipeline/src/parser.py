"""
회의록 TSV 데이터 파서 및 메타데이터 추출기
"""
import re
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

# 지역명 매핑 테이블
REGION_MAP = {
    "강원": "강원",
    "경기": "경기",
    "충청남": "충남",
    "충청북": "충북",
}

# 주요 직함 키워드 목록
PREFIX_POSITIONS = ["위원장", "부위원장", "의장", "부의장", "도지사", "교육감", "부지사", "시장", "군수", "구청장"]
SUFFIX_POSITIONS = ["위원", "의원", "과장", "국장", "실장", "팀장", "원장", "총장"]


def clean_text(text: Optional[str]) -> str:
    """비표준 공백(\xa0 등) 및 불필요한 연속 공백 정제"""
    if not text:
        return ""
    # non-breaking space 및 공백 정리
    cleaned = text.replace("\xa0", " ")
    cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()
    return cleaned


def parse_speaker(raw_speaker: str) -> Tuple[str, str]:
    """
    화자 문자열에서 직함(position)과 순수 이름(clean_name) 분리
    예:
      '위원장이상정' -> ('위원장', '이상정')
      '노금식의원'   -> ('의원', '노금식')
      '김동연'       -> ('일반', '김동연')
    """
    speaker = clean_text(raw_speaker)
    if not speaker or speaker == "미상":
        return "미상", "미상"

    # 1. 접두사 직함 검사 (예: 위원장이상정)
    for pos in PREFIX_POSITIONS:
        if speaker.startswith(pos):
            clean_name = speaker[len(pos):].strip()
            return pos, clean_name if clean_name else pos

    # 2. 접미사 직함 검사 (예: 노금식의원)
    for pos in SUFFIX_POSITIONS:
        if speaker.endswith(pos):
            clean_name = speaker[:-len(pos)].strip()
            return pos, clean_name if clean_name else pos

    return "일반", speaker


def get_region(assembly: str) -> str:
    """의회명에서 시도 광역 자치단체 약칭 추출"""
    for key, val in REGION_MAP.items():
        if key in assembly:
            return val
    return "기타"


def parse_date(raw_date: str) -> Tuple[str, int, int, int]:
    """
    날짜 문자열 파싱 (YYYYMMDD -> 'YYYY-MM-DD', year, month, day)
    """
    cleaned_date = clean_text(raw_date)
    try:
        dt = datetime.strptime(cleaned_date, "%Y%m%d")
        return dt.strftime("%Y-%m-%d"), dt.year, dt.month, dt.day
    except Exception:
        # 형식 예외 시 안전 처리
        digits = re.sub(r"\D", "", cleaned_date)
        if len(digits) >= 8:
            y, m, d = int(digits[:4]), int(digits[4:6]), int(digits[6:8])
            return f"{y:04d}-{m:02d}-{d:02d}", y, m, d
        elif len(digits) >= 4:
            y = int(digits[:4])
            return f"{y:04d}-01-01", y, 1, 1
        return "1970-01-01", 1970, 1, 1


def parse_tsv_row(line: str, row_id: int, file_name: str) -> Optional[Dict[str, Any]]:
    """
    TSV 한 줄을 파싱하여 기본 메타데이터 및 발언 본문 딕셔너리로 반환
    """
    parts = line.rstrip("\r\n").split("\t")
    if len(parts) < 7:
        return None

    raw_date = parts[0]
    assembly = clean_text(parts[1])
    raw_agenda = parts[2]
    speaker_role = clean_text(parts[3])
    raw_speaker = parts[4]
    speaker_type = clean_text(parts[5])
    utterance = clean_text(parts[6])

    # 날짜 분해
    formatted_date, year, month, day = parse_date(raw_date)

    # 안건명 정제 (너무 긴 경우 250자로 제한)
    clean_agenda = clean_text(raw_agenda)[:250]

    # 화자 직함/이름 분리
    speaker_pos, clean_speaker_name = parse_speaker(raw_speaker)
    region = get_region(assembly)

    return {
        "source_file": file_name,
        "row_id": int(row_id),
        "date": formatted_date,
        "year": int(year),
        "month": int(month),
        "day": int(day),
        "assembly": assembly,
        "region": region,
        "agenda_title": clean_agenda,
        "speaker_name": clean_speaker_name,
        "speaker_raw": clean_text(raw_speaker),
        "speaker_position": speaker_pos,
        "speaker_role": speaker_role if speaker_role else "미상",
        "speaker_type": speaker_type if speaker_type else "미상",
        "utterance": utterance,
    }
