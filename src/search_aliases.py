#!/usr/bin/env python3
"""검색어와 지역·위원회 필터의 제한적인 별칭을 관리한다.

별칭은 제공된 회의록의 표기 차이를 함께 조회하기 위한 용도다. 외부 시점의
정보를 추가하지 않으며, 원본 council 값은 변경하지 않는다.
"""

from __future__ import annotations

from collections.abc import Iterable


COUNCIL_ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    ("전라남도", "전남", "전라남도의회", "전남도의회"),
    (
        "전라북도", "전북", "전라북도의회",
        "전북특별자치도", "전북특별자치도의회",
    ),
    (
        "강원도", "강원도의회",
        "강원특별자치도", "강원특별자치도의회",
    ),
)

COMMITTEE_ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    ("행정자치위원회", "기획행정위원회"),
)


def _normalize(value: str | None) -> str:
    return " ".join(str(value or "").split()).strip()


def _alias_values(
    value: str | None,
    groups: Iterable[tuple[str, ...]],
) -> tuple[str, ...]:
    normalized = _normalize(value)
    if not normalized:
        return ()
    for group in groups:
        if normalized in group:
            return tuple(dict.fromkeys(group))
    return (normalized,)


def council_alias_values(value: str | None) -> tuple[str, ...]:
    return _alias_values(value, COUNCIL_ALIAS_GROUPS)


def committee_alias_values(value: str | None) -> tuple[str, ...]:
    return _alias_values(value, COMMITTEE_ALIAS_GROUPS)


def expand_query_aliases(query: str) -> str:
    normalized = _normalize(query)
    additions: list[str] = []
    for group in COUNCIL_ALIAS_GROUPS:
        if any(alias in normalized for alias in group):
            additions.extend(alias for alias in group if alias not in normalized)
    for group in COMMITTEE_ALIAS_GROUPS:
        if any(alias in normalized for alias in group):
            additions.extend(alias for alias in group if alias not in normalized)
    return " ".join([normalized, *dict.fromkeys(additions)]).strip()
