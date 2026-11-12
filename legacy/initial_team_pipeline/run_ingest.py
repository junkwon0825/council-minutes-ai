"""
회의록 텍스트 파일들을 청킹하여 메타데이터와 함께 ChromaDB에 적재하는 CLI 스크립트
(전체 예상 소요 시간 ETA, 실시간 처리 속도 표시, 임베딩 모델/컬렉션 변경 및 LLM 정책 메타데이터 자동 병합 지원)
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from tqdm import tqdm

from src.config import (
    DATA_DIR,
    BASE_DIR,
    BATCH_SIZE,
    MIN_CHAR_LEN,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_PRESETS,
    resolve_embedding_model,
)
from src.parser import parse_tsv_row
from src.chunker import CouncilMinuteChunker
from src.indexer import ChromaIndexer


def format_seconds(seconds: float) -> str:
    """초 단위를 보기 쉬운 시:분:초 형식으로 변환"""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}시간 {m:02d}분 {s:02d}초"
    elif m > 0:
        return f"{m}분 {s:02d}초"
    else:
        return f"{s}초"


def load_llm_metadata_map(meta_paths: Optional[List[Path]]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """
    LLM 추출 메타데이터 파일들(JSONL)을 읽어 (assembly, row_id) -> 메타데이터 딕셔너리로 매핑
    """
    meta_map = {}
    valid_paths = [p for p in (meta_paths or []) if p and p.exists()]
    if not valid_paths:
        return meta_map

    file_names = ", ".join(p.name for p in valid_paths)
    print(f">> [메타데이터 로드] LLM 정책 메타데이터 파일 감지 ({len(valid_paths)}개): {file_names}")
    count = 0
    for meta_path in valid_paths:
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    row_id = int(data.get("row_id", 0))
                    assembly = data.get("assembly", "")
                    if row_id > 0:
                        meta_map[(assembly, row_id)] = data
                        if ("", row_id) not in meta_map:
                            meta_map[("", row_id)] = data
                        count += 1
                except Exception:
                    pass

    print(f"   -> 총 {count:,}건의 정밀 정책 메타데이터(키워드/쟁점/분야)를 ChromaDB 색인용으로 매핑했습니다.\n")
    return meta_map


def ingest_data(
    data_dir: Path = DATA_DIR,
    sample_lines: int = 0,
    batch_size: int = BATCH_SIZE,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    min_char_len: int = MIN_CHAR_LEN,
    model_name: Optional[str] = None,
    collection_name: Optional[str] = None,
    meta_file: Optional[Path] = None,
    reset: bool = False,
):
    start_time = time.time()
    resolved_model = resolve_embedding_model(model_name)
    target_collection = collection_name or COLLECTION_NAME

    # LLM 정책 메타데이터 파일 자동 감지 (지정되지 않은 경우 data/*.jsonl 전체 로드)
    if meta_file:
        resolved_meta_paths = [meta_file]
    else:
        # data 폴더 내 모든 jsonl 파일 자동 감지 (중복 제외)
        data_jsonls = sorted(list((BASE_DIR / "data").glob("*.jsonl")))
        resolved_meta_paths = data_jsonls

    llm_meta_map = load_llm_metadata_map(resolved_meta_paths)

    print("=" * 65)
    print(" [지방의회 회의록 ChromaDB 적재 파이프라인] ")
    print("=" * 65)
    print(f" * 데이터 경로      : {data_dir}")
    print(f" * 임베딩 모델      : {resolved_model}")
    print(f" * 대상 컬렉션      : {target_collection}")
    print(f" * Chroma 배치 크기 : {batch_size}")
    print(f" * 청크 크기 / 중첩 : {chunk_size}자 / {chunk_overlap}자")
    print(f" * 최소 발언 필터링 : {min_char_len}자 미만 제외")
    print(f" * LLM 메타데이터   : {'적용 중 (' + str(len(llm_meta_map)//2) + '건)' if llm_meta_map else '미적용 (일반 TSV 메타데이터만 사용)'}")
    if sample_lines > 0:
        print(f" * [테스트 모드]     : 파일당 최대 {sample_lines:,}행만 적재")
    print("=" * 65)

    # 1. 텍스트 파일 목록 수집 및 사전 라인 수 집계
    txt_files = sorted(glob.glob(str(data_dir / "*.txt")))
    if not txt_files:
        print(f"[오류] '{data_dir}' 경로에 .txt 파일이 없습니다.")
        return

    print(">> 파일 목록 및 행 수 집계 중...")
    file_info = []
    grand_total_target_lines = 0

    for fpath in txt_files:
        fname = os.path.basename(fpath)
        with open(fpath, "r", encoding="utf-8-sig") as f:
            total_lines = sum(1 for _ in f) - 1  # 헤더 제외

        target = min(total_lines, sample_lines) if sample_lines > 0 else total_lines
        file_info.append((fpath, fname, total_lines, target))
        grand_total_target_lines += target

    print(f">> 총 {len(txt_files)}개 파일 | 처리 대상: 총 {grand_total_target_lines:,}행\n")

    # 2. 인덱서 및 청커 초기화
    indexer = ChromaIndexer(
        collection_name=target_collection,
        embedding_model_name=resolved_model,
        reset=reset,
    )
    chunker = CouncilMinuteChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_char_len=min_char_len,
    )

    total_chunks_added = 0

    print("\n" + "-" * 65)
    print(" [적재 진행 상황] (남은 예상 시간: remaining)")
    print("-" * 65)

    with tqdm(
        total=grand_total_target_lines,
        desc="[Total Ingest]",
        unit="rows",
        bar_format="{desc}: {percentage:3.0f}%|{bar:25}| {n_fmt}/{total_fmt} rows [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        dynamic_ncols=True,
    ) as total_pbar:

        for fpath, fname, total_lines, target_lines in file_info:
            batch_ids = []
            batch_docs = []
            batch_metas = []
            file_chunks = 0

            # 파일명 앞부분 약칭 추출 (예: 2024_강원)
            short_name = fname.replace("cleaned_", "").replace(".txt", "")[:12]

            with open(fpath, "r", encoding="utf-8-sig") as f:
                header = f.readline()  # 헤더 스킵

                with tqdm(
                    total=target_lines,
                    desc=f" [{short_name}]",
                    unit="rows",
                    leave=False,
                    bar_format="{desc}: {percentage:3.0f}%|{bar:15}| {n_fmt}/{total_fmt} rows [{elapsed}<{remaining}]",
                    dynamic_ncols=True,
                ) as file_pbar:

                    for row_id, line in enumerate(f, start=1):
                        if sample_lines > 0 and row_id > sample_lines:
                            break

                        record = parse_tsv_row(line, row_id=row_id, file_name=fname)
                        if record:
                            chunk_items = chunker.chunk_parsed_record(record)
                            for chunk_id, chunk_text, meta in chunk_items:
                                # LLM 추출 정책 메타데이터가 있으면 1:1 병합
                                assembly = str(meta.get("assembly", ""))
                                llm_info = llm_meta_map.get((assembly, row_id)) or llm_meta_map.get(("", row_id))
                                if llm_info and llm_info.get("is_substantive"):
                                    meta["policy_category"] = str(llm_info.get("policy_category", "기타"))
                                    meta["core_issue"] = str(llm_info.get("core_issue", ""))
                                    meta["keywords"] = ", ".join(llm_info.get("keywords", []))
                                    meta["summary"] = str(llm_info.get("summary", ""))
                                else:
                                    meta["policy_category"] = "기타"
                                    meta["core_issue"] = ""
                                    meta["keywords"] = ""
                                    meta["summary"] = ""

                                batch_ids.append(chunk_id)
                                batch_docs.append(chunk_text)
                                batch_metas.append(meta)

                        # 배치가 채워지면 ChromaDB에 임베딩 및 색인
                        if len(batch_ids) >= batch_size:
                            indexer.add_batch(batch_ids, batch_docs, batch_metas)
                            batch_len = len(batch_ids)
                            file_chunks += batch_len
                            total_chunks_added += batch_len
                            batch_ids.clear()
                            batch_docs.clear()
                            batch_metas.clear()

                            total_pbar.set_postfix({
                                "chunks": f"{total_chunks_added:,}",
                                "last_batch": f"{batch_len}",
                            })

                        total_pbar.update(1)
                        file_pbar.update(1)

            # 파일 내 남은 잔여 청크 적재
            if batch_ids:
                indexer.add_batch(batch_ids, batch_docs, batch_metas)
                batch_len = len(batch_ids)
                file_chunks += batch_len
                total_chunks_added += batch_len
                batch_ids.clear()
                batch_docs.clear()
                batch_metas.clear()

                total_pbar.set_postfix({
                    "chunks": f"{total_chunks_added:,}",
                    "last_batch": f"{batch_len}",
                })

    elapsed_total = time.time() - start_time
    print("-" * 65)
    print(" [최종 적재 결과 보고서]")
    print(f" * 총 소요 시간     : {format_seconds(elapsed_total)} ({elapsed_total:.1f}초)")
    print(f" * 처리된 원본 행   : {grand_total_target_lines:,}행")
    print(f" * 생성/적재된 청크 : {total_chunks_added:,}개")
    print(f" * 평균 처리 속도   : {grand_total_target_lines / elapsed_total:.1f}행/초")
    print(f" * Chroma 전체 청크 : {indexer.count():,}개 (컬렉션: {target_collection})")
    print("=" * 65)


if __name__ == "__main__":
    preset_help = ", ".join([f"'{k}' ({v})" for k, v in EMBEDDING_PRESETS.items()])
    parser = argparse.ArgumentParser(description="회의록 ChromaDB 적재 도구")
    parser.add_argument(
        "--sample-lines",
        type=int,
        default=0,
        help="테스트용: 각 파일당 읽을 최대 행 수 (0이면 전체 처리)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"ChromaDB 배치 적재 단위 (기본: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=MIN_CHAR_LEN,
        help=f"최소 발언 길이 필터링 (기본: {MIN_CHAR_LEN})",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=EMBEDDING_MODEL_NAME,
        help=f"임베딩 모델명 또는 프리셋 별칭. (기본: {EMBEDDING_MODEL_NAME})",
    )
    parser.add_argument(
        "-c", "--collection",
        type=str,
        default=COLLECTION_NAME,
        help=f"ChromaDB 컬렉션 이름 (기본: {COLLECTION_NAME})",
    )
    parser.add_argument(
        "--meta-file",
        type=str,
        default="",
        help="LLM 정책 메타데이터(JSONL) 경로 (기본: data/gangwon_policy_metadata.jsonl 자동 감지)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="적재 전 대상 Chroma 컬렉션 초기화 여부",
    )

    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    meta_file_path = Path(args.meta_file) if args.meta_file else None

    ingest_data(
        sample_lines=args.sample_lines,
        batch_size=args.batch_size,
        min_char_len=args.min_len,
        model_name=args.model,
        collection_name=args.collection,
        meta_file=meta_file_path,
        reset=args.reset,
    )
