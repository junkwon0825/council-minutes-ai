"""
회의록 발언 단위 및 서브 청킹 모듈
"""
from typing import Iterator, Tuple, Dict, Any, List
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHAR_LEN


class CouncilMinuteChunker:
    """
    회의록 발언 청킹 처리기
    - 발언(Utterance)을 1차 기본 단위로 유지
    - chunk_size를 초과하는 장문 발언에 한해 RecursiveCharacterTextSplitter 적용
    - ChromaDB 제약 조건(None 불가, scalar 타입)에 맞춘 메타데이터 자동 생성
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        min_char_len: int = MIN_CHAR_LEN,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_char_len = min_char_len

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_parsed_record(
        self, record: Dict[str, Any]
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        단일 파싱 레코드를 청킹하여 (chunk_id, document_text, metadata) 리스트 반환
        """
        utterance = record.get("utterance", "")
        if len(utterance) < self.min_char_len:
            # 단순 진행 멘트 등 너무 짧은 발언은 건너뜀
            return []

        # 장문 발언 여부에 따라 분할
        if len(utterance) > self.chunk_size:
            chunk_texts = self.splitter.split_text(utterance)
        else:
            chunk_texts = [utterance]

        total_chunks = len(chunk_texts)
        results = []

        region = record.get("region", "UNK")
        date_str = record.get("date", "NODATE").replace("-", "")
        row_id = record.get("row_id", 0)

        for chunk_idx, chunk_text in enumerate(chunk_texts):
            # 고유 chunk_id: e.g. 충북_20240123_r142_c0
            chunk_id = f"{region}_{date_str}_r{row_id}_c{chunk_idx}"

            # ChromaDB 호환 메타데이터 생성 (모든 값은 str, int, float, bool)
            metadata: Dict[str, Any] = {
                "source_file": str(record.get("source_file", "")),
                "row_id": int(row_id),
                "chunk_index": int(chunk_idx),
                "total_chunks": int(total_chunks),
                "date": str(record.get("date", "")),
                "year": int(record.get("year", 0)),
                "month": int(record.get("month", 0)),
                "day": int(record.get("day", 0)),
                "assembly": str(record.get("assembly", "")),
                "region": str(region),
                "agenda_title": str(record.get("agenda_title", "")),
                "speaker_name": str(record.get("speaker_name", "")),
                "speaker_raw": str(record.get("speaker_raw", "")),
                "speaker_position": str(record.get("speaker_position", "")),
                "speaker_role": str(record.get("speaker_role", "")),
                "speaker_type": str(record.get("speaker_type", "")),
                "char_length": int(len(chunk_text)),
                "is_continuation": bool(chunk_idx > 0),
            }

            results.append((chunk_id, chunk_text, metadata))

        return results
