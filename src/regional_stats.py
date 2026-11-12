#!/usr/bin/env python3
"""지역별(17개 시도) 공공 통계 지표 — 통계청 KOSIS 국가통계포털 Open API.

Postgre_DB/llm_external/regional_stats.py를 이 프로젝트 구조로 옮긴 것이다.
  · 지표 목록: config/kosis_indicators.json (인구·출산·재정·교통사고·고용·복지 …)
  · 이슈 → 지표 고르기는 로컬에서만(외부 전송 없음) → KOSIS에서 17개 시도 값 → 지역별 '정책 수요' 점수(0~1)
  · 표 고르기: 사람이 확인해 고정한 org_id/tbl_id가 있으면 그 표, 없으면 검색어로 KOSIS 통계표를 찾아
    17개 시도 값이 나오는 첫 표를 자동 선택(결과에 '자동 선택'으로 표시)
  · 시점: live 모드는 조회일까지 공표된 값 전부, backtest 모드는 기준일 시점에 이미 공표된 값만
    (수록 연도 말 + publication_lag_months ≤ 기준일)

  python src/regional_stats.py --list
  python src/regional_stats.py --match 저출생
  python src/regional_stats.py --discover tfr            # 후보 통계표 검색 (사람이 확인 후 --pin)
  python src/regional_stats.py --pin tfr 101 DT_XXXX
  python src/regional_stats.py --check                   # 모든 지표가 17개 시도 값을 내는지 점검
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.parse
from datetime import date
from pathlib import Path
from typing import Any

import external_policy_sources as EXT


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG = PROJECT_ROOT / "config" / "kosis_indicators.json"
KOSIS = "https://kosis.kr/openapi"
DEFAULT_LAG_MONTHS = 9
TOTAL_WORDS = {"계", "합계", "전체", "총계", "소계", "전국", "총인구", "남녀전체", "남녀계", "전체연령", "합", "계(남+여)", "총",
               "종합", "총합", "전체가구", "총계(전체)"}
SHORT = {"서울": "서울특별시의회", "부산": "부산광역시의회", "대구": "대구광역시의회", "인천": "인천광역시의회",
         "광주": "광주광역시의회", "대전": "대전광역시의회", "울산": "울산광역시의회", "세종": "세종특별자치시의회",
         "경기": "경기도의회", "강원": "강원특별자치도의회", "충북": "충청북도의회", "충남": "충청남도의회",
         "전북": "전북특별자치도의회", "전남": "전라남도의회", "경북": "경상북도의회", "경남": "경상남도의회",
         "제주": "제주특별자치도의회"}


def catalog() -> list[dict[str, Any]]:
    return json.loads(CATALOG.read_text(encoding="utf-8"))["indicators"]


def indicator_text(ind: dict[str, Any]) -> str:
    """의미 유사도 연결용 설명 문장"""
    return ind.get("description") or f"{ind['name']} ({ind['category']}): {', '.join(ind['keywords'])}"


def match_indicators(keyword: str, limit: int = 4) -> list[dict[str, Any]]:
    """이슈어와 지표 keywords를 비교 (로컬 처리, 외부 전송 없음)"""
    k = re.sub(r"\s+", "", keyword or "")
    out = []
    for n, ind in enumerate(catalog()):
        hit = [w for w in ind["keywords"] if re.sub(r"\s+", "", w) in k or k in re.sub(r"\s+", "", w)]
        if hit:
            best = max(len(re.sub(r"\s+", "", w)) for w in hit)      # 더 길게(구체적으로) 일치한 지표를 앞에
            out.append((-best, -len(hit), n, {"id": ind["id"], "name": ind["name"], "category": ind["category"],
                                              "need": ind["need"], "matched": hit}))
    return [x[-1] for x in sorted(out)][:limit]


def council_of(name: str) -> str | None:
    n = re.sub(r"\s+", "", name or "")
    for council, forms in EXT.group_regions().items():      # 통합 이후 통계표의 통합 지자체 행
        if n in forms:
            return council
    for council, formal in EXT.REGION.items():
        if n in formal:
            return council
    return SHORT.get(n)


def latest_visible_year(as_of: str, mode: str, lag_months: int) -> int:
    """연간 수록 시점 중 사용할 수 있는 마지막 연도"""
    day = date.fromisoformat(as_of[:10])
    if mode != "backtest":
        return day.year
    # Y년 자료는 (Y년 12월 + lag)개월에 공표된다고 보고, 그 달이 기준일이 속한 달 이전이어야 함
    return (day.year * 12 + day.month - 1 - 11 - lag_months) // 12


class KosisAuthError(RuntimeError):
    """인증키 오류 — 표·분류 조합을 바꿔 다시 시도해도 소용없으므로 즉시 중단"""


def _api(path: str, params: dict, keyword: str):
    key = EXT._key("KOSIS_API_KEY")
    if not key:
        raise EXT.PolicyError("KOSIS_API_KEY 가 설정되지 않음")
    q = urllib.parse.urlencode({"method": "getList", "apiKey": key, "format": "json", "jsonVD": "Y", **params})
    data = json.loads(EXT._get(f"{KOSIS}/{path}?{q}", "kosis", keyword).decode("utf-8"))
    if isinstance(data, dict) and data.get("err"):
        error = KosisAuthError if str(data.get("err")) == "11" else RuntimeError
        raise error(f"KOSIS 오류 {data.get('err')}: {data.get('errMsg')}")
    return data


def kosis_search(term: str, n: int = 10) -> list[dict[str, Any]]:
    term = EXT.check_keyword(term)
    ck = f"kosis_search|{term}"
    hit = EXT._cache_get(ck, max_age_days=30)
    if hit:
        return hit["value"]
    rows = _api("statisticsSearch.do", {"searchNm": term, "resultCount": n}, term)
    out = [{"org_id": r.get("ORG_ID"), "org_nm": r.get("ORG_NM"), "tbl_id": r.get("TBL_ID"), "tbl_nm": r.get("TBL_NM"),
            "period": f"{r.get('STRT_PRD_DE', '')}~{r.get('END_PRD_DE', '')}"} for r in rows if r.get("TBL_ID")]
    EXT._cache_put(ck, out)
    return out


def kosis_fetch(org_id: str, tbl_id: str, periods: int = 5) -> list[dict[str, Any]]:
    """통계표의 최근 periods년(월별 표면 periods×12개월) 전체 — 분류 단계 수를 몰라 objL1..n을 늘려 가며 시도"""
    ck = f"kosis_data|{org_id}|{tbl_id}|{periods}"
    hit = EXT._cache_get(ck, max_age_days=30)
    if hit:
        return hit["value"]
    last = None
    for prd_se, n in (("Y", periods), ("M", periods * 12)):     # 연간 자료가 없으면 월별(→ 연평균)
        for depth in range(1, 5):
            params = {"orgId": org_id, "tblId": tbl_id, "itmId": "ALL", "prdSe": prd_se, "newEstPrdCnt": n}
            params.update({f"objL{i}": "ALL" for i in range(1, depth + 1)})
            try:
                rows = _api("Param/statisticsParameterData.do", params, tbl_id)
                if isinstance(rows, list) and rows:
                    EXT._cache_put(ck, rows)
                    return rows
            except KosisAuthError:
                raise
            except RuntimeError as e:
                last = e
    raise RuntimeError(f"{tbl_id} 자료를 받지 못함: {last}")


def extract_regions(rows: list, item_hint: str | None, year_max: int) -> dict | None:
    """KOSIS 행 목록 → {의회: 값}  (지역 분류가 시도이고 나머지 분류는 '계'인 행만, year_max 이하 최신 시점)"""
    items = sorted({r.get("ITM_NM", "") for r in rows})
    item = next((i for i in items if item_hint and item_hint.replace(" ", "") in i.replace(" ", "")), items[0] if items else "")
    cand: dict[str, dict] = {}
    for r in rows:
        if r.get("ITM_NM", "") != item:
            continue
        prd = str(r.get("PRD_DE", ""))[:4]
        if not prd.isdigit() or int(prd) > year_max:
            continue
        dims = [r.get(f"C{i}_NM") for i in range(1, 9) if r.get(f"C{i}_NM")]
        regions = [council_of(d) for d in dims]
        reg = [c for c in regions if c]
        if len(reg) != 1:
            continue
        others = [d for d, c in zip(dims, regions) if not c]
        if any(re.sub(r"\s+", "", o) not in TOTAL_WORDS for o in others):
            continue
        try:
            v = float(str(r.get("DT", "")).replace(",", ""))
        except ValueError:
            continue
        cand.setdefault(prd, {}).setdefault(reg[0], ([], r.get("UNIT_NM", "")))[0].append(v)   # 월별이면 연평균
    cand = {p: {c: (round(sum(vs) / len(vs), 4), u) for c, (vs, u) in d.items()} for p, d in cand.items()}
    for prd in sorted(cand, reverse=True):          # 17개(통합 후 16개) 중 15개 이상 채워진 가장 최근 시점
        if len(cand[prd]) >= 15:
            vals = cand[prd]
            return {"period": prd, "item": item, "unit": next(iter(vals.values()))[1],
                    "values": {c: v for c, (v, _) in vals.items()}}
    return None


def need_scores(values: dict, need: str) -> dict:
    """값 순위 → 0~1 (1 = 그 정책 수요가 가장 큰 지역). need='high'면 큰 값, 'low'면 작은 값이 1"""
    order = sorted(values, key=lambda c: values[c], reverse=(need == "high"))
    n = max(len(order) - 1, 1)
    return {c: round(1 - i / n, 3) for i, c in enumerate(order)}


def get_regional_indicator(ind_id: str, as_of: str | None = None, mode: str = "live") -> dict:
    src = EXT.DATA_SOURCES["kosis"]
    ind = next((i for i in catalog() if i["id"] == ind_id), None)
    if not ind:
        return {"error": f"없는 지표 id: {ind_id} (사용 가능: {', '.join(i['id'] for i in catalog())})"}
    if not EXT._key("KOSIS_API_KEY"):
        return {"source": src["name"], "error": f"KOSIS_API_KEY 가 설정되지 않아 사용할 수 없음 ({src['terms']})"}
    as_of = as_of or date.today().isoformat()
    year_max = latest_visible_year(as_of, mode, int(ind.get("publication_lag_months") or DEFAULT_LAG_MONTHS))
    k = ind["kosis"]
    try:
        tables = ([{"org_id": k["org_id"], "tbl_id": k["tbl_id"], "tbl_nm": k.get("tbl_nm") or k["tbl_id"], "org_nm": k.get("org_nm", "")}]
                  if k.get("org_id") and k.get("tbl_id") else kosis_search(k["search"]))
    except (EXT.PolicyError, RuntimeError, ValueError, OSError) as e:
        return {"source": src["name"], "indicator": ind_id, "error": f"통계표 검색 실패: {type(e).__name__}: {e}"}
    pinned = bool(k.get("org_id") and k.get("tbl_id") and k.get("verified"))      # 사람이 확인한 표만 '확정'
    errors = []
    for t in tables[:5]:
        try:
            got = extract_regions(kosis_fetch(t["org_id"], t["tbl_id"]), k.get("item_hint"), year_max)
        except KosisAuthError as e:
            return {"source": src["name"], "indicator": ind_id,
                    "error": f"{e} — .env의 KOSIS_API_KEY 값을 확인하세요 (kosis.kr/openapi 마이페이지)"}
        except (RuntimeError, ValueError, OSError) as e:          # OSError: 네트워크 시간 초과·연결 실패
            errors.append(f"{t['tbl_id']}: {type(e).__name__}: {e}")
            continue
        if not got:
            errors.append(f"{t['tbl_id']}: {year_max}년 이하에서 17개 시도 값이 나오는 행을 찾지 못함")
            continue
        sc = need_scores(got["values"], ind["need"])
        return {"source": src["name"], "source_id": "kosis", "indicator": ind["id"], "name": ind["name"],
                "need": ind["need"], "table": {**t, "pinned": pinned}, "period": got["period"], "item": got["item"],
                "unit": got["unit"], "values": got["values"], "need_score": sc, "as_of": as_of, "mode": mode,
                "attribution": f"출처: 통계청 KOSIS 국가통계포털, 「{t.get('tbl_nm')}」({t.get('org_nm') or '작성기관'}; "
                               f"통계표 {t['org_id']}/{t['tbl_id']}), 수록시점 {got['period']}"
                               + ("" if pinned else " — 자동 선택된 표, 확인 필요")}
    return {"source": src["name"], "indicator": ind_id, "error": "알맞은 통계표를 찾지 못함: " + " | ".join(errors[:3])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--match")
    ap.add_argument("--discover", metavar="ID")
    ap.add_argument("--pin", nargs=3, metavar=("ID", "ORG_ID", "TBL_ID"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.list:
        for i in catalog():
            k = i["kosis"]
            state = "확인됨" if k.get("verified") else ("후보·확인 필요" if k.get("tbl_id") else "자동 선택")
            print(f"{i['id']:14} {i['category']:6} {i['name']:20} 수요={'클수록' if i['need'] == 'high' else '작을수록'}  "
                  f"표={(k['org_id'] + '/' + k['tbl_id']) if k.get('tbl_id') else k['search']} [{state}]")
    if a.match:
        print(json.dumps(match_indicators(a.match), ensure_ascii=False, indent=1))
    if a.discover:
        ind = next(i for i in catalog() if i["id"] == a.discover)
        for t in kosis_search(ind["kosis"]["search"]):
            print(f"  {t['org_id']:>5} {t['tbl_id']:<16} {t['tbl_nm']}  ({t['org_nm']}, {t['period']})")
        print(f"\n맞는 표를 확인했으면: python src/regional_stats.py --pin {a.discover} <ORG_ID> <TBL_ID>")
    if a.pin:
        doc = json.loads(CATALOG.read_text(encoding="utf-8"))
        for i in doc["indicators"]:
            if i["id"] == a.pin[0]:
                i["kosis"].update(org_id=a.pin[1], tbl_id=a.pin[2])
        CATALOG.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"고정: {a.pin[0]} → {a.pin[1]}/{a.pin[2]}")
    if a.check:
        for i in catalog():
            r = get_regional_indicator(i["id"])
            print(f"{i['id']:14} " + (f"오류: {r['error']}" if r.get("error") else
                                      f"OK {len(r['values'])}개 시도, {r['period']}, {r['item']} ({r['unit']}) — {r['attribution']}"))


if __name__ == "__main__":
    main()
