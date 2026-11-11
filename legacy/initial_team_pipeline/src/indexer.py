"""
LangChain Community의 Chroma VectorStore 및 배치 적재 파이프라인
"""
import os
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
try:
    import torch
except ImportError:
    raise ImportError(
        "\n" + "=" * 65 + "\n"
        "[오류] 'torch' 패키지를 찾을 수 없습니다.\n"
        "현재 Conda 가상환경('project01')이 활성화되지 않은 상태입니다.\n\n"
        "▶ 해결 방법 (터미널에서 가상환경 활성화 후 재실행):\n"
        "   conda activate project01\n"
        "   python run_ingest.py --batch-size 500\n\n"
        "▶ 만약 가상환경 활성화 후에도 동일하다면 PyTorch를 설치하세요:\n"
        "   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121\n"
        + "=" * 65
    )

# LangChain Community 모듈 import
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    resolve_embedding_model,
)

# LangChain deprecation 경고 필터링 (콘솔 가독성 유지)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class ChromaIndexer:
    """
    LangChain Community의 Chroma VectorStore 관리 클래스
    - HuggingFaceEmbeddings (CUDA GPU 지원)
    - 모델명 또는 별칭(alias) 자동 해석
    - LangChain 호환 Document 및 메타데이터 적재
    - 배치 단위 add_texts 및 시맨틱 필터 검색 지원
    """

    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        collection_name: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        reset: bool = False,
    ):
        self.persist_directory = persist_directory or CHROMA_DIR
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name or COLLECTION_NAME

        # 모델 별칭 해석 (예: 'kosimcse' -> 'BM-K/KoSimCSE-roberta-multitask')
        resolved_model = resolve_embedding_model(embedding_model_name)
        self.model_name = resolved_model

        # CUDA GPU 가속 자동 감지
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        alias_info = f" (별칭/입력: {embedding_model_name})" if embedding_model_name and embedding_model_name != resolved_model else ""
        print(f"[임베딩] HuggingFace '{resolved_model}'{alias_info} 모델 로드 중 (디바이스: {self.device.upper()})...")
        print(f"[ChromaDB] 컬렉션: '{self.collection_name}' | 저장소: '{self.persist_directory}'")

        # LangChain Community HuggingFaceEmbeddings 초기화
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": False},
        )

        # LangChain Community Chroma VectorStore 초기화
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_directory),
        )

        if reset:
            try:
                self.vectorstore.delete_collection()
                print(f"[알림] 기존 컬렉션 '{self.collection_name}' 초기화 완료.")
                # 삭제 후 새 컬렉션 재생성
                self.vectorstore = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=str(self.persist_directory),
                )
            except Exception as e:
                print(f"[경고] 컬렉션 초기화 중 예외: {e}")

    def add_batch(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """
        LangChain Community Chroma의 add_texts 메서드를 통한 배치 적재
        """
        if not ids:
            return

        try:
            self.vectorstore.add_texts(
                texts=documents,
                metadatas=metadatas,
                ids=ids,
            )
        except Exception as e:
            if "dimension" in str(e).lower():
                raise RuntimeError(
                    f"임베딩 벡터 차원 불일치 오류가 발생했습니다!\n"
                    f"원인: 기존 컬렉션('{self.collection_name}')에 다른 모델의 차원으로 데이터가 적재되어 있습니다.\n"
                    f"해결: 다른 컬렉션명을 지정하거나(--collection <NAME>), 컬렉션을 초기화(--reset)한 후 다시 적재하세요."
                ) from e
            raise

    def count(self) -> int:
        """현재 Chroma 컬렉션에 적재된 청크 개수 반환"""
        try:
            return self.vectorstore._collection.count()
        except Exception:
            return 0

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        LangChain similarity_search_with_score 메서드를 통한 검색
        반환값: List of (Document, score)
        """
        try:
            return self.vectorstore.similarity_search_with_score(
                query=query_text,
                k=n_results,
                filter=where,
            )
        except Exception as e:
            if "dimension" in str(e).lower():
                raise RuntimeError(
                    f"임베딩 벡터 차원 불일치 오류가 발생했습니다!\n"
                    f"원인: 검색에 지정된 임베딩 모델('{self.model_name}')과 기존 컬렉션('{self.collection_name}')의 차원이 다릅니다.\n"
                    f"해결: 인덱싱할 때 사용한 동일한 임베딩 모델을 지정하세요."
                ) from e
            raise
