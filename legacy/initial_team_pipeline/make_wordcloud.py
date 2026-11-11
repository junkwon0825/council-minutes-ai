"""
강원특별자치도의회 회의록 핵심 키워드 추출 및 워드클라우드 생성 스크립트
(Kiwi 고속 형태소 분석기 기반 + 정책 복합키워드 및 불용어 엄격 정제)
"""
import argparse
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from kiwipiepy import Kiwi
from wordcloud import WordCloud

# 의회 회의록 특화 불용어 목록 (상투적 행정/직함/의전/기관명 완벽 배제)
COUNCIL_STOPWORDS = {
    # 1. 사용자가 특별히 요청한 제외 키워드 (변별력 없는 포괄적 단어)
    "사업", "예산", "지원", "조례", "조례안", "강원특별자치도", "의견", "업무",
    "행정", "강원도", "강원", "의회", "도정", "시군", "지자체", "정부", "국가",
    # 2. 직함 및 호칭
    "의원", "위원", "위원장", "부위원장", "의장", "부의장", "도지사", "교육감",
    "부지사", "시장", "군수", "구청장", "국장", "과장", "실장", "팀장", "원장",
    "총장", "위원님", "의원님", "도민", "군민", "시민", "선배", "동료", "존경",
    "관계자", "관계관", "공무원", "총괄", "소관", "부서",
    # 3. 회의 진행, 의전 및 형식 멘트
    "발언", "질의", "답변", "회의", "본회의", "임시회", "정례회", "감사",
    "개의", "산회", "정회", "선포", "의사", "일정", "위원회", "수고", "인사",
    "말씀", "보고", "설명", "자료", "서류", "제출", "다음", "질문", "회의록",
    "얘기", "이야기", "청취", "퇴장", "출석", "부탁", "요청", "고생", "답변석",
    # 4. 일반 지시/대명사 및 무의미한 형식 명사
    "우리", "저희", "본인", "자신", "그것", "이것", "저것", "여기", "저기",
    "부분", "때문", "생각", "정도", "사항", "관련", "내용", "경우", "사실",
    "확인", "판단", "입장", "이유", "이후", "이전", "지금", "현재", "당시",
    "오늘", "내일", "어제", "하나", "이상", "이하", "가운데", "자체", "각종",
    "주요", "전체", "모든", "서로", "단계", "추진", "진행", "노력", "필요",
    "준비", "검토", "상황", "결과", "시간", "추가", "개정", "신청", "이용",
    "사용", "방법", "대상", "기준", "시작", "존경하는", "감사합니다", "제시",
    "마련", "반영", "해결", "조치", "발생", "확보", "기회",
}


def extract_keywords_and_generate_wordcloud(
    input_file: str = "text/cleaned_2024_강원특별자치도의회.txt",
    output_image: str = "gangwon_wordcloud.png",
    top_n: int = 100,
    font_path: str = "C:/Windows/Fonts/malgun.ttf",
    include_bigrams: bool = True,
):
    print("=" * 65)
    print(" [강원특별자치도의회 핵심 정책 의제 & 워드클라우드 추출] ")
    print("=" * 65)
    print(f" * 입력 데이터   : {input_file}")
    print(f" * 상투어 필터링 : '사업, 예산, 지원, 조례, 행정, 강원도' 등 배제 완료")
    print(f" * 복합 키워드   : {'포함 (Bigram 결합)' if include_bigrams else '단일 명사만'}")
    print(f" * 상위 키워드   : {top_n}개")

    if not os.path.exists(input_file):
        print(f"[오류] 입력 파일이 존재하지 않습니다: {input_file}")
        return []

    # 1. Kiwi 형태소 분석기 로드
    kiwi = Kiwi(num_workers=4)

    # 2. 발언 수집
    start_time = time.time()
    utterances = []
    with open(input_file, "r", encoding="utf-8-sig") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 7:
                utt = parts[6].strip()
                if len(utt) >= 20:
                    utterances.append(utt)

    print(f">> 총 {len(utterances):,}건 유효 발언 형태소 분석 진행 중...")

    # 3. 명사 및 복합어 추출
    keyword_counter = Counter()

    for res in kiwi.tokenize(utterances):
        nouns_in_sent = []
        for token in res:
            if token.tag in ("NNG", "NNP") and len(token.form) >= 2:
                word = token.form
                if word not in COUNCIL_STOPWORDS:
                    nouns_in_sent.append(word)
                    keyword_counter[word] += 1

        # 복합 명사구 (연속된 유의미 명사 2개 결합, 예: "지역 소멸", "농지 혁신")
        if include_bigrams and len(nouns_in_sent) >= 2:
            for i in range(len(nouns_in_sent) - 1):
                w1, w2 = nouns_in_sent[i], nouns_in_sent[i + 1]
                if w1 != w2:
                    bigram = f"{w1} {w2}"
                    keyword_counter[bigram] += 2  # 복합 키워드에 가중치 부여

    elapsed_nlp = time.time() - start_time
    print(f">> 키워드 집계 완료: {len(keyword_counter):,}종 (소요 시간: {elapsed_nlp:.2f}초)")

    top_keywords = keyword_counter.most_common(top_n)

    # 상위 30개 출력
    print("\n" + "=" * 65)
    print(f"{'순위':<6}{'정제된 정책 키워드':<20}{'중요도(빈도)':<12}")
    print("-" * 45)
    for rank, (word, count) in enumerate(top_keywords[:30], 1):
        print(f"{rank:<6}{word:<20}{count:<12,}")
    print("=" * 65)

    # 4. 워드클라우드 렌더링
    wc = WordCloud(
        font_path=font_path,
        width=1600,
        height=900,
        background_color="white",
        colormap="ocean",
        max_words=top_n,
        prefer_horizontal=0.85,
        min_font_size=16,
        max_font_size=210,
        relative_scaling=0.45,
        random_state=42,
    )
    wc.generate_from_frequencies(dict(top_keywords))
    wc.to_file(output_image)
    print(f"[성공] 정제된 워드클라우드 저장 완료: {output_image}")

    return top_keywords


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="정책 중심 워드클라우드 생성기")
    parser.add_argument("--input", type=str, default="text/cleaned_2024_강원특별자치도의회.txt")
    parser.add_argument("--output", type=str, default="gangwon_wordcloud.png")
    parser.add_argument("--top", type=int, default=100)

    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    extract_keywords_and_generate_wordcloud(
        input_file=args.input,
        output_image=args.output,
        top_n=args.top,
    )
