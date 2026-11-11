# Legacy Initial Team Pipeline

이 디렉터리는 2026 DATA·AI 분석 경진대회 초기에 사용한 팀 저장소
`team-ggeodeureok`의 코드 스냅샷을 보존한 영역입니다.

## Why it is preserved

최종 공개 저장소의 `src/` 파이프라인은 대회 종료 후 통합·정리된 코드이기 때문에,
초기 단계에서 수행한 전처리, ChromaDB 적재, 임베딩 모델 비교, 검색 실험의 개발 흐름이
Git history에서 충분히 드러나지 않았습니다.

따라서 초기 파이프라인을 삭제하지 않고 legacy 영역에 보존해
**초기 실험 → 최종 파이프라인**의 발전 과정을 확인할 수 있도록 했습니다.

## Main contents

- JSON/HTML 전처리 및 TSV 변환
- 발언 파싱 및 metadata 구조화
- 발언 단위 chunking
- Hugging Face embedding model preset
- ChromaDB batch ingestion
- semantic search + metadata filtering
- local LLM policy metadata extraction
- embedding model comparison experiments

## Provenance

이 코드는 기존 팀 저장소에서 이관한 스냅샷입니다.
기존 저장소의 commit author와 실제 코드 작성자가 항상 일치하지 않았기 때문에
commit history를 재작성하지 않았습니다.

개인 기여 범위는 저장소 루트의
[CONTRIBUTIONS.md](../../CONTRIBUTIONS.md)를 참고하세요.

## Current version

현재 포트폴리오에서 사용하는 최종 파이프라인은
루트의 `src/`와 `demo/`를 기준으로 합니다.
이 디렉터리는 개발 과정과 초기 실험을 보존하기 위한 자료입니다.
