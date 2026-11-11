"""
지방의회 회의록에서 로컬 LLM(Ollama gemma4:e2b)을 활용하여 구체적 정책 의제 및 복합 키워드(메타데이터용)를 추출하는 스크립트
(체크포인트 이어하기, 멀티스레드 병렬 가속, 실시간 flush 지원)
"""
import argparse
import concurrent.futures
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
import requests
from tqdm import tqdm


# 시스템 프롬프트: 포괄적 단어 엄격 배제 및 2~4어절 구체적 정책 복합키워드 추출 강제
EXTRACTION_SYSTEM_PROMPT = """당신은 대한민국 지방의회 정책 분석 및 메타데이터 구축 전문가입니다.
주어진 지방의회 회의록 발언을 정밀하게 분석하여, 향후 검색 및 필터링 메타데이터로 활용할 수 있도록 JSON 포맷으로만 응답하세요.

[⚠️ 절대 금지 키워드 (Negative Constraints)]
아래와 같이 모든 회의록에 흔히 등장하는 1단어짜리 포괄적 명사, 기관명, 직함은 'keywords'에 절대로 포함하지 마세요:
- 금지 단어: 사업, 예산, 지원, 조례, 조례안, 강원특별자치도, 의견, 업무, 행정, 강원도, 의회, 의원, 위원, 발언, 질의, 답변, 감사, 생각, 회의, 내용, 부분

[✅ 권장 키워드 형식]
2~4개 단어가 결합된 '구체적이고 실질적인 정책 의제 복합명사'로 추출하세요:
- 좋은 예시: "농지제도 혁신", "지역 소멸 대응", "미래 인구정책", "일자리 산업 육성", "학령인구 감소", "교육복지 강화", "반도체 클러스터", "공공의료 확충", "해안침식 방지", "정주여건 개선"

[출력 포맷 (반드시 아래 순수 JSON 형식으로만 응답할 것, 마크다운 코드블록 없이 순수 JSON만 출력)]
{
  "is_substantive": true,
  "policy_category": "경제/산업",
  "core_issue": "농지 규제 완화를 통한 농가 소득 증대 및 농지제도 혁신 방안",
  "keywords": ["농지제도 혁신", "농업 규제 완화", "경자유전 원칙 개편"],
  "summary": "해방 이후 지속된 농지 소유 규제를 완화하여 청년농 유입과 농업 경쟁력을 강화해야 한다고 주장함."
}

※ policy_category 후보: "경제/산업", "농림/수산", "보건/복지", "교육/문화", "환경/안전", "도시/교통", "자치/행정", "기타" 중 정확히 1개 선택.
※ 만약 단순 의전 멘트(개의 선포, 방청객 환영, 단순 이석 보고 등)로 실질적 정책 논의가 없다면 is_substantive를 false로 하고 keywords는 []로 비워두세요.
"""


def clean_and_parse_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """LLM 출력 문자열에서 유연하고 안전하게 JSON을 파싱"""
    if not raw_text:
        return None

    text = raw_text.strip()
    if "```" in text:
        text = re.sub(r'```(?:json)?\s*', '', text)
        text = text.replace('```', '').strip()

    match = re.search(r'(\{[\s\S]*\})', text)
    if match:
        json_candidate = match.group(1).strip()
    else:
        json_candidate = text

    try:
        return json.loads(json_candidate)
    except Exception:
        pass

    try:
        fixed = re.sub(r',\s*\}', '}', json_candidate)
        fixed = re.sub(r',\s*\]', ']', fixed)
        return json.loads(fixed)
    except Exception:
        return None


def get_available_ollama_models(base_url: str) -> List[str]:
    """Ollama 서버의 가용한 모델 목록 반환"""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=3)
        if resp.status_code == 200:
            return [m.get("name", "") for m in resp.json().get("models", [])]
    except Exception:
        pass
    return []


def check_ollama_model(base_url: str, model_name: str) -> bool:
    """Ollama 서버에서 해당 모델이 사용 가능한지 확인"""
    models = get_available_ollama_models(base_url)
    return any(model_name.lower() in m.lower() for m in models)


def extract_metadata_with_llm(
    text: str,
    speaker_info: str,
    model_name: str,
    base_url: str = "http://localhost:11434",
    timeout: int = 60,
) -> Optional[Dict[str, Any]]:
    """단일 발언 텍스트에서 로컬 LLM을 호출하여 JSON 메타데이터 추출"""
    
    user_prompt = f"""[발언자 정보]: {speaker_info}
[발언 내용]:
{text}

위 발언을 분석하여 시스템 지침에 따라 JSON으로만 응답하세요."""

    payload = {
        "model": model_name,
        "prompt": f"{EXTRACTION_SYSTEM_PROMPT}\n\n{user_prompt}",
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
        }
    }

    try:
        resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=timeout)
        resp.raise_for_status()
        raw_output = resp.json().get("response", "").strip()
        return clean_and_parse_json(raw_output)
    except Exception:
        return None


def load_processed_row_ids(output_file: str) -> Set[int]:
    """이미 처리되어 저장된 row_id 집합을 로드 (체크포인트 이어하기용)"""
    processed = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        if "row_id" in data:
                            processed.add(int(data["row_id"]))
                    except Exception:
                        pass
    return processed


def process_single_record(
    record: Dict[str, Any],
    model_name: str,
    base_url: str,
) -> Dict[str, Any]:
    """단일 레코드에 대해 LLM 호출 및 결과 딕셔너리 구성"""
    speaker_info = f"{record['assembly']} {record['speaker_role']} {record['speaker_name']} ({record['speaker_type']}) - 안건: {record['agenda_title']}"
    meta = extract_metadata_with_llm(
        text=record["utterance"][:1200],
        speaker_info=speaker_info,
        model_name=model_name,
        base_url=base_url,
    )

    if meta and isinstance(meta, dict):
        cat = meta.get("policy_category", "기타")
        if "|" in cat:
            cat = cat.split("|")[0].strip()

        return {
            "row_id": record["row_id"],
            "date": record["date"],
            "assembly": record["assembly"],
            "speaker": f"{record['speaker_role']} {record['speaker_name']}",
            "agenda": record["agenda_title"],
            "is_substantive": meta.get("is_substantive", True),
            "policy_category": cat,
            "core_issue": meta.get("core_issue", ""),
            "keywords": meta.get("keywords", []),
            "summary": meta.get("summary", ""),
            "utterance_preview": record["utterance"][:100] + "...",
        }
    else:
        return {
            "row_id": record["row_id"],
            "date": record["date"],
            "assembly": record["assembly"],
            "speaker": f"{record['speaker_role']} {record['speaker_name']}",
            "agenda": record["agenda_title"],
            "is_substantive": False,
            "policy_category": "미상",
            "core_issue": "추출 실패",
            "keywords": [],
            "summary": "",
            "utterance_preview": record["utterance"][:100] + "...",
        }


def process_council_file(
    input_file: str,
    output_file: str,
    model_name: str = "gemma4:e2b",
    base_url: str = "http://localhost:11434",
    sample_limit: int = 0,
    min_char_len: int = 80,
    max_workers: int = 2,
):
    print("=" * 65)
    print(" [로컬 LLM 기반 의회 회의록 정책 키워드 & 메타데이터 추출기] ")
    print("=" * 65)
    print(f" * 입력 데이터   : {input_file}")
    print(f" * 출력 파일     : {output_file}")
    print(f" * 요청 모델     : {model_name}")
    print(f" * 동시 워커 수   : {max_workers}개 스레드 (병렬 가속)")
    print(f" * 처리 대상     : {'전체 처리' if sample_limit <= 0 else f'최대 {sample_limit}건 샘플'}")
    print(f" * 최소 글자수   : {min_char_len}자 미만 제외")
    print("=" * 65)

    # 1. 모델 가용성 확인
    available_models = get_available_ollama_models(base_url)
    if not available_models:
        print(f"[오류] Ollama 서버({base_url})에 연결할 수 없습니다. 'ollama serve'를 실행하세요.")
        return

    selected_model = model_name
    if not check_ollama_model(base_url, model_name):
        print(f"[경고] 요청 모델 '{model_name}'이 로컬에 없습니다. 'ollama pull {model_name}'을 권장합니다.")
        fallback = available_models[0]
        for m in available_models:
            if "gemma" in m.lower():
                fallback = m
                break
        print(f"▶ 로컬 대체 모델 '{fallback}'으로 진행합니다.\n")
        selected_model = fallback

    # 2. 체크포인트 확인 (이미 처리된 행은 스킵)
    processed_ids = load_processed_row_ids(output_file)
    if processed_ids:
        print(f"[체크포인트] 기존 결과 파일에서 이미 처리된 {len(processed_ids):,}건을 감지했습니다. (자동 이어서 진행)")

    # 3. 대상 데이터 로드
    records_to_process = []
    with open(input_file, "r", encoding="utf-8-sig") as f:
        f.readline()  # 헤더
        for row_id, line in enumerate(f, start=1):
            if row_id in processed_ids:
                continue

            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 7:
                utterance = parts[6].strip()
                if len(utterance) >= min_char_len:
                    records_to_process.append({
                        "row_id": row_id,
                        "date": parts[0],
                        "assembly": parts[1],
                        "agenda_title": parts[2],
                        "speaker_role": parts[3],
                        "speaker_name": parts[4],
                        "speaker_type": parts[5],
                        "utterance": utterance,
                    })
                    if 0 < sample_limit <= len(records_to_process):
                        break

    if not records_to_process:
        print("[알림] 모든 대상 발언이 이미 처리 완료되었습니다.")
        return

    print(f">> 신규 처리 대상: {len(records_to_process):,}건 (동시 {max_workers}개 병렬 워커 가동)\n")

    # 4. 멀티스레드 병렬 실행 및 실시간 파일 기록 (안전한 Ctrl+C 중단 지원)
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    file_lock = threading.Lock()
    success_count = 0
    processed_count = 0
    interrupted = False

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    future_to_rec = {}

    with open(output_file, "a", encoding="utf-8") as out_f:
        with tqdm(total=len(records_to_process), desc=f"[{selected_model}] 메타데이터 추출", unit="건") as pbar:
            try:
                # 작업 제출
                future_to_rec = {
                    executor.submit(process_single_record, rec, selected_model, base_url): rec
                    for rec in records_to_process
                }

                for future in concurrent.futures.as_completed(future_to_rec):
                    try:
                        res = future.result()
                        if res.get("is_substantive") and res.get("keywords"):
                            success_count += 1

                        with file_lock:
                            out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                            out_f.flush()
                        processed_count += 1
                    except Exception:
                        pass
                    finally:
                        pbar.update(1)

            except (KeyboardInterrupt, SystemExit):
                interrupted = True
                print("\n\n[중단 알림] Ctrl + C 가 감지되었습니다.")
                print(">> 대기 중인 작업을 취소하고 현재까지의 진행 상황을 안전하게 저장합니다...")
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    executor.shutdown(wait=False)
                for f in future_to_rec:
                    f.cancel()
            finally:
                if not interrupted:
                    executor.shutdown(wait=True)

    status_title = " [중단 처리 완료 결과 보고서] " if interrupted else " [추출 완료 결과 보고서] "
    print("\n" + "=" * 65)
    print(status_title)
    print("=" * 65)
    print(f" * 이번 세션 처리 : {processed_count:,}건{' (중단됨)' if interrupted else ''}")
    print(f" * 유효 정책 발언 : {success_count:,}건")
    print(f" * 총 누적 데이터 : {len(load_processed_row_ids(output_file)):,}건")
    print(f" * 저장된 파일    : {output_file}")
    if interrupted:
        print(" * 이어하기 안내  : 동일한 명령어를 다시 실행하시면 중단된 지점부터 이어서 진행됩니다.")
    print("=" * 65)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ollama 로컬 LLM 회의록 메타데이터 추출기")
    parser.add_argument(
        "--input",
        type=str,
        default="text/cleaned_2024_강원특별자치도의회.txt",
        help="입력 TSV 회의록 파일",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="출력 JSONL 파일 경로 (미지정 시 data/{입력파일명}.jsonl 자동 생성)",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="gemma4:e2b",
        help="사용할 Ollama 모델명 (기본: gemma4:e2b, 고성능 모델 필요 시 gemma4:e4b 선택 가능)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="처리할 발언 개수 (0이면 전체 처리, 기본: 0)",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=80,
        help="최소 발언 길이 필터링 (기본: 80자)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="동시 병렬 요청 워커 수 (기본: 2)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:11434",
        help="Ollama API 서버 주소 (기본: http://localhost:11434)",
    )

    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    input_path = Path(args.input)
    if not args.output:
        # 입력 파일의 stem(파일명에서 확장자 제외)을 기본 jsonl 이름으로 사용
        output_file = str(Path("data") / f"{input_path.stem}.jsonl")
    else:
        output_file = args.output

    process_council_file(
        input_file=str(input_path),
        output_file=output_file,
        model_name=args.model,
        base_url=args.url,
        sample_limit=args.sample,
        min_char_len=args.min_len,
        max_workers=args.workers,
    )

