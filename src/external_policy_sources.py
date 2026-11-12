"""공공 Open API(법제처 자치법규·열린국회정보 의안·통계청 KOSIS) 조회 모듈.

회의록 원문은 외부로 보내지 않는다. 외부로 나가는 값은 check_keyword를 통과한
짧은 정책어뿐이며, 허용 호스트·접근 기록·로컬 캐시를 강제한다.
KOSIS 통계표 조회는 regional_stats.py가 이 모듈의 _get·캐시를 써서 한다.
결과는 화면 참고용(reference_only)이며 회의록 기반 예측 모델 입력에는 쓰지 않는다.
"""

from __future__ import annotations

import hashlib
import html.entities
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:  # 의회 통합 규칙(전남광주통합특별시의회 = 광주 + 전남). 없으면 단일 의회로만 처리
    sys.path.insert(0, str(PROJECT_ROOT.parent / "llm_baseline"))
    import assembly_groups as AG  # noqa: E402
except ImportError:
    AG = None


def _load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    """프로젝트 .env(KEY=값)를 환경 변수로. 이미 설정된 값은 덮어쓰지 않고 값은 출력하지 않음"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and not os.getenv(key):
            os.environ[key] = value


_load_dotenv()

CACHE = Path(os.getenv("COUNCIL_EXTERNAL_CACHE") or PROJECT_ROOT / "data" / "external" / "cache")
UA = {"User-Agent": "local-council-minutes-research/1.0"}
ALLOWED_HOSTS = {"www.law.go.kr", "open.assembly.go.kr", "kosis.kr"}

# 명칭·버전·이용 조건 — 결과 JSON에 그대로 기록됨. 이용 전 각 약관 원문을 확인할 것
DATA_SOURCES = {
    "law_ordin": {
        "name": "국가법령정보 공동활용 Open API — 자치법규 목록 조회",
        "provider": "법제처 (국가법령정보센터)",
        "api": "https://www.law.go.kr/DRF/lawSearch.do?target=ordin (DRF 목록 조회 서비스, type=XML)",
        "version": "DRF 목록 조회 서비스 (법제처가 버전 번호를 따로 부여하지 않아 서비스명·조회일로 표기)",
        "data_type": "공공데이터 — 자치법규 이름·지자체기관명·공포일자·제개정 구분 (본문 미수집)",
        "copyright": "조례·규칙은 저작권법 제7조(보호받지 못하는 저작물)에 해당. Open API 이용은 법제처 공동활용 이용약관을 따름",
        "terms": "https://open.law.go.kr",
        "attribution": "출처: 법제처 국가법령정보센터(www.law.go.kr) 자치법규 Open API",
        "key_env": "LAW_OC",
    },
    "assembly_bills": {
        "name": "열린국회정보 Open API — 국회의원 발의법률안 (서비스 ID nzmimeepazxkubdpn)",
        "provider": "국회사무처 (열린국회정보)",
        "api": "https://open.assembly.go.kr/portal/openapi/nzmimeepazxkubdpn",
        "version": "Open API 서비스 nzmimeepazxkubdpn (제21·22대 국회 조회)",
        "data_type": "공공데이터 — 의안명·대표발의자·제안일·처리결과",
        "copyright": "열린국회정보 이용약관 및 해당 데이터의 공공누리 표시를 따름 (이용 전 확인 필요)",
        "terms": "https://open.assembly.go.kr",
        "attribution": "출처: 국회사무처 열린국회정보(open.assembly.go.kr)",
        "key_env": "ASSEMBLY_API_KEY",
    },
    "kosis": {
        "name": "KOSIS 국가통계포털 공유서비스 Open API — 통계자료(statisticsParameterData)·통계표 검색(statisticsSearch)",
        "provider": "통계청 (각 통계표의 작성기관은 결과에 따로 표시)",
        "api": "https://kosis.kr/openapi/Param/statisticsParameterData.do , https://kosis.kr/openapi/statisticsSearch.do",
        "version": "KOSIS 공유서비스 Open API — 통계표별 버전은 '통계표ID + 수록시점 + 조회일'로 결과마다 표시",
        "data_type": "공공데이터 — 17개 시도별 집계 통계값(개인 단위 자료 아님)",
        "copyright": "KOSIS 이용약관 및 각 통계표의 작성기관 이용 조건(공공누리 표시 등)을 따름 — 이용 전 확인 필요",
        "terms": "https://kosis.kr/openapi",
        "attribution": "출처: 통계청 KOSIS 국가통계포털(kosis.kr) — 통계표명·작성기관·수록시점은 결과별로 표시",
        "key_env": "KOSIS_API_KEY",
    },
}

# 광역의회 ↔ 자치법규 지자체기관명(정식 명칭)
REGION = {
    "서울특별시의회": ["서울특별시"], "부산광역시의회": ["부산광역시"], "대구광역시의회": ["대구광역시"],
    "인천광역시의회": ["인천광역시"], "광주광역시의회": ["광주광역시"], "대전광역시의회": ["대전광역시"],
    "울산광역시의회": ["울산광역시"], "세종특별자치시의회": ["세종특별자치시"], "경기도의회": ["경기도"],
    "강원특별자치도의회": ["강원특별자치도", "강원도"], "충청북도의회": ["충청북도"], "충청남도의회": ["충청남도"],
    "전북특별자치도의회": ["전북특별자치도", "전라북도"], "전라남도의회": ["전라남도"], "경상북도의회": ["경상북도"],
    "경상남도의회": ["경상남도"], "제주특별자치도의회": ["제주특별자치도"],
}
# 회의록 메타데이터의 옛 명칭 → REGION 기준 명칭
COUNCIL_ALIASES = {"전라북도의회": "전북특별자치도의회", "강원도의회": "강원특별자치도의회"}

_KEYWORD_OK = re.compile(r"^[가-힣A-Za-z0-9][가-힣A-Za-z0-9 ·\-]{0,29}$")
# 회의록 발언자 표기(이름+직함)가 검색어로 새 나가지 않도록
_PERSON = re.compile(r"[가-힣]{2,4}\s*(의원|위원장|위원|의장|국장|실장|과장|시장|도지사|지사|교육감|교육장|단장|본부장|청장)(님)?$"
                     r"|^(위원장|부위원장|부의장|의원|위원|의장|국장|실장|과장|시장|도지사|교육감)\s*[가-힣]{2,4}$")
_XML_STANDARD_ENTITIES = {b"amp", b"lt", b"gt", b"apos", b"quot"}
_XML_NAMED_ENTITY = re.compile(rb"&([A-Za-z][A-Za-z0-9]+);")
_XML_BARE_AMPERSAND = re.compile(
    rb"&(?!#\d+;|#x[0-9A-Fa-f]+;|amp;|lt;|gt;|apos;|quot;)"
)


class PolicyError(ValueError):
    """외부 데이터 이용 원칙에 어긋나는 요청"""


def check_keyword(keyword: str) -> str:
    """외부로 보낼 검색어 검사: 짧은 정책어만 허용 (회의록 문장·이름 목록 등 유출 방지)"""
    k = re.sub(r"\s+", " ", (keyword or "").strip())
    if not _KEYWORD_OK.match(k) or len(k.split()) > 4:
        raise PolicyError(f"외부로 보낼 수 없는 검색어입니다(짧은 정책어 1~30자, 4단어 이하만 허용): {k[:40]!r}")
    if _PERSON.search(k):
        raise PolicyError(f"사람(발언자) 이름으로 보이는 검색어는 외부로 보내지 않습니다: {k!r}")
    return k


def _xml_safe_bytes(payload: bytes) -> bytes:
    """법제처 XML에 섞인 HTML 엔티티와 잘못된 `&`를 XML-safe 형태로 바꾼다.

    XML은 amp/lt/gt/apos/quot 다섯 이름만 기본 지원한다. 법제처 응답의
    `&nbsp;`, `&middot;` 등은 Unicode 숫자 참조로 바꾸고, 알 수 없는
    엔티티는 원문 문자열로 보존되도록 `&amp;name;`으로 이스케이프한다.
    """

    def replace_named(match: re.Match[bytes]) -> bytes:
        name_bytes = match.group(1)
        if name_bytes in _XML_STANDARD_ENTITIES:
            return match.group(0)
        name = name_bytes.decode("ascii")
        value = html.entities.html5.get(name + ";")
        if value is None:
            return b"&amp;" + name_bytes + b";"
        return "".join(f"&#{ord(character)};" for character in value).encode("ascii")

    normalized = _XML_NAMED_ENTITY.sub(replace_named, payload)
    return _XML_BARE_AMPERSAND.sub(b"&amp;", normalized)


def _parse_law_xml(payload: bytes) -> ET.Element:
    """법제처 응답을 정규화해 파싱하고 HTML 오류 응답을 구분한다."""
    root = ET.fromstring(_xml_safe_bytes(payload))
    if str(root.tag).lower().endswith("html"):
        raise RuntimeError("법제처 API가 XML 대신 HTML 오류 페이지를 반환했습니다")
    return root


def normalize_council(council: str) -> str:
    name = (council or "").strip()
    return COUNCIL_ALIASES.get(name, name)


def sources_used(keys) -> list:
    return [{"id": k, **{f: DATA_SOURCES[k][f] for f in ("name", "provider", "version", "copyright", "terms", "attribution")}}
            for k in keys if k in DATA_SOURCES]


def key_available(source_id: str) -> bool:
    return _key(DATA_SOURCES[source_id]["key_env"]) is not None


def _cache_get(key: str, max_age_days: int = 7):
    f = CACHE / (hashlib.md5(key.encode("utf-8")).hexdigest() + ".json")
    if f.exists() and (time.time() - f.stat().st_mtime) < max_age_days * 86400:
        return json.loads(f.read_text(encoding="utf-8"))
    return None


def _cache_put(key: str, value):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / (hashlib.md5(key.encode("utf-8")).hexdigest() + ".json")
    f.write_text(json.dumps({"key": key, "saved_at": datetime.now().isoformat(timespec="seconds"), "value": value},
                            ensure_ascii=False), encoding="utf-8")


def _get(url: str, source: str, keyword: str, timeout: int = 20) -> bytes:
    host = urllib.parse.urlparse(url).hostname
    if host not in ALLOWED_HOSTS:
        raise PolicyError(f"허용되지 않은 외부 주소: {host}")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    CACHE.mkdir(parents=True, exist_ok=True)
    with (CACHE / "access_log.jsonl").open("a", encoding="utf-8") as f:     # 인증값은 기록하지 않음
        f.write(json.dumps({"at": datetime.now().isoformat(timespec="seconds"), "source": source, "host": host,
                            "keyword": keyword, "bytes": len(body)}, ensure_ascii=False) + "\n")
    return body


def _key(env: str) -> str | None:
    v = (os.getenv(env) or "").strip()
    return v or None


def group_regions() -> dict:
    """통합 의회 → 통합 지자체 표기 목록 (assembly_groups가 없으면 빈 dict)"""
    if AG is None:
        return {}
    out = {}
    for g in AG._groups():
        names = [g["name"]] + g.get("aliases", [])
        forms = {n[:-2] if n.endswith("의회") else n for n in names}
        out[g["name"]] = sorted(forms, key=len, reverse=True)
    return out


def members_of(council: str) -> list:
    """통합 의회면 전신 의회 목록, 아니면 [자기 자신]"""
    g = AG.find_group(council or "") if AG is not None else None
    return [m["assembly"] for m in g["members"]] if g else [council]


def region_of_org(org: str) -> tuple:
    """자치법규 지자체기관명 → (광역의회, 단계)   단계: '광역'(도·시 본청/교육청) / '기초'(시·군·구)"""
    o = (org or "").strip()
    for council, forms in [*group_regions().items(), *REGION.items()]:
        for f in forms:
            if o.startswith(f):
                rest = o[len(f):].strip()
                return council, ("광역" if rest in ("", "교육청") else "기초")
    return None, None


# ----------------------------------------------------------------------------- 자치법규 (법제처)
def fetch_local_ordinances(keyword: str, max_pages: int = 5) -> dict:
    """자치법규 '이름' 검색 전체 목록(기준일 필터 전). 목록 정보만 받고 본문은 받지 않음"""
    src = DATA_SOURCES["law_ordin"]
    oc = _key(src["key_env"])
    if not oc:
        return {"source": src["name"], "error": f"{src['key_env']} 가 설정되지 않아 사용할 수 없음 "
                                                 f"(법제처 Open API 활용 신청 후 받은 인증값을 환경 변수로 설정: {src['terms']})"}
    keyword = check_keyword(keyword)
    ck = f"ordin|{keyword}|{hashlib.sha256(oc.encode()).hexdigest()[:8]}"
    items = (_cache_get(ck) or {}).get("value")
    if items is None:
        items = []
        for page in range(1, max_pages + 1):
            q = urllib.parse.urlencode({"OC": oc, "target": "ordin", "type": "XML", "query": keyword,
                                        "display": 100, "page": page})
            response = _get(
                f"https://www.law.go.kr/DRF/lawSearch.do?{q}",
                "law_ordin", keyword,
            )
            root = _parse_law_xml(response)
            if (root.findtext("resultCode") or "00") != "00":
                return {"source": src["name"], "error": f"법제처 API 오류: {root.findtext('resultMsg')}"}
            rows = root.findall("law")
            for x in rows:
                g = lambda tag: (x.findtext(tag) or "").strip()
                d = g("공포일자")
                items.append({"name": g("자치법규명"), "org": g("지자체기관명"), "kind": g("자치법규종류"),
                              "type": g("제개정구분명"), "promulgated": f"{d[:4]}-{d[4:6]}-{d[6:8]}" if len(d) == 8 else None,
                              "law_id": g("자치법규ID")})
            total = int(root.findtext("totalCnt") or 0)
            if page * 100 >= total or not rows:
                break
            time.sleep(0.5)
        _cache_put(ck, items)
    out = []
    for it in items:
        council, level = region_of_org(it["org"])
        out.append({**it, "council": council, "level": level})
    return {"source": src["name"], "source_id": "law_ordin", "attribution": src["attribution"],
            "keyword": keyword, "count": len(out), "items": out}


def search_local_ordinances(keyword: str, as_of: str | None = None, max_pages: int = 5) -> dict:
    """공포일 ≤ as_of 자치법규만 의회별로 집계 (기존 Agent 도구와 같은 출력 형식)"""
    fetched = fetch_local_ordinances(keyword, max_pages=max_pages)
    if "error" in fetched:
        return fetched
    out = [it for it in fetched["items"]
           if not as_of or (it["promulgated"] and it["promulgated"] <= as_of)]
    by = {}
    for it in out:
        if it["council"]:
            b = by.setdefault(it["council"], {"광역": 0, "기초": 0, "first": None})
            b[it["level"]] += 1
            b["first"] = min(filter(None, [b["first"], it["promulgated"]]), default=None)
    return {**{k: v for k, v in fetched.items() if k != "items"}, "as_of": as_of,
            "count": len(out), "by_council": by, "items": out[:60]}


# ----------------------------------------------------------------------------- 국회 의안 (국회사무처)
def fetch_national_bills(keyword: str) -> dict:
    src = DATA_SOURCES["assembly_bills"]
    key_ = _key(src["key_env"])
    if not key_:
        return {"source": src["name"], "error": f"{src['key_env']} 가 설정되지 않아 사용할 수 없음 ({src['terms']})"}
    keyword = check_keyword(keyword)
    out = []
    for age in (22, 21):
        q = urllib.parse.urlencode({"KEY": key_, "Type": "json", "pIndex": 1, "pSize": 100, "AGE": age, "BILL_NAME": keyword})
        ck = f"bills|{age}|{keyword}"
        data = (_cache_get(ck) or {}).get("value")
        if data is None:
            data = json.loads(_get(f"https://open.assembly.go.kr/portal/openapi/nzmimeepazxkubdpn?{q}", "assembly_bills", keyword))
            _cache_put(ck, data)
        blocks = data.get("nzmimeepazxkubdpn") if isinstance(data, dict) else None
        rows = blocks[-1].get("row", []) if blocks else []
        for r in rows:
            out.append({"bill": r.get("BILL_NAME"), "proposer": r.get("PROPOSER"),
                        "date": (r.get("PROPOSE_DT") or "")[:10], "result": r.get("PROC_RESULT"), "age": age})
    return {"source": src["name"], "source_id": "assembly_bills", "attribution": src["attribution"],
            "keyword": keyword, "count": len(out), "items": out}


def search_national_bills(keyword: str, as_of: str | None = None) -> dict:
    fetched = fetch_national_bills(keyword)
    if "error" in fetched:
        return fetched
    out = [it for it in fetched["items"] if not as_of or (it["date"] and it["date"] <= as_of)]
    return {**{k: v for k, v in fetched.items() if k != "items"}, "as_of": as_of,
            "count": len(out), "items": out[:30]}


def safe_call(func, *args: Any, **kwargs: Any) -> dict:
    """정책 위반·외부 서버 오류는 파이프라인을 멈추지 않고 error 결과로"""
    try:
        return func(*args, **kwargs)
    except PolicyError as e:
        return {"error": f"이용 원칙 위반으로 차단: {e}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
