"""Tiny filter predicate over :class:`PropertyRow` sequences.

Grammar (simple enough that a hand-written tokenizer beats a dep on
lark/pyparsing):

    expr   := atom ( ' AND ' atom )*
    atom   := field ( '.' subkey )? op value
    field  := property | property_name | structure_id
            | method | conditions
            | composition_formula | composition_prototype
    op     := = | != | > | < | >= | <= | in
    value  := 'string'   (single-quoted)
            | number
            | 'a','b','c' (for ``in``)

`property=bandgap` is sugar for `property_name='bandgap'`: matches the
roadmap CLI example. Unquoted alphabetic RHS is also treated as a
string so callers don't have to double-quote in shells.

Complexity that isn't here yet: OR, NOT, parentheses, LIKE. Callers
who need those can post-filter in Python or wait for 6.2b.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable, List, Tuple

from .schema import PropertyRow


class FilterParseError(ValueError):
    """Raised on malformed filter expressions."""


# ---------------------------------------------------------------------------
# Tokenizer / parser
# ---------------------------------------------------------------------------


_OPS = ("<=", ">=", "!=", "=", "<", ">", "in")

# Matches ``field``, ``field.subkey``, a value (quoted string, number,
# or unquoted identifier), or an operator / AND.
_TOKEN_RE = re.compile(
    r"""
    \s*
    (?:
        (?P<str>'([^']*)')                    # 'quoted'
      | (?P<num>-?\d+(?:\.\d+)?)              # 123 or -0.5
      | (?P<op>(?:<=|>=|!=|=|<|>|in))         # operator
      | (?P<and>AND)                           # conjunction
      | (?P<ident>[A-Za-z_][\w]*)             # ident / field / bare string
      | (?P<dot>\.)                            # field.subkey separator
      | (?P<comma>,)                           # in-list separator
    )
    \s*
    """,
    re.VERBOSE,
)


def _tokenize(expr: str) -> List[Tuple[str, str]]:
    tokens: List[Tuple[str, str]] = []
    pos = 0
    n = len(expr)
    while pos < n:
        m = _TOKEN_RE.match(expr, pos)
        if m is None or m.end() == pos:
            raise FilterParseError(
                f"could not tokenize from position {pos}: {expr[pos:pos+24]!r}"
            )
        for name in ("str", "num", "op", "and", "ident", "dot", "comma"):
            tok = m.group(name)
            if tok is not None:
                if name == "str":
                    tokens.append(("STR", m.group(2) or ""))
                elif name == "and":
                    tokens.append(("AND", "AND"))
                else:
                    tokens.append((name.upper(), tok))
                break
        pos = m.end()
    return tokens


# Field aliases — what the user types → the PropertyRow attribute.
_FIELD_ALIASES = {
    "property": "property_name",
    "property_name": "property_name",
    "property_value": "property_value",
    "structure_id": "structure_id",
    "method": "method",
    "conditions": "conditions",
    "composition_formula": "composition_formula",
    "composition_prototype": "composition_prototype",
}


def _parse_atom(tokens: List[Tuple[str, str]], i: int) -> Tuple[Callable[[PropertyRow], bool], int]:
    # field or field.subkey
    if i >= len(tokens) or tokens[i][0] != "IDENT":
        raise FilterParseError(
            f"expected field name at token {i}: got {tokens[i:i+2]}"
        )
    field_raw = tokens[i][1]
    if field_raw not in _FIELD_ALIASES:
        raise FilterParseError(
            f"unknown field {field_raw!r}. Known: {sorted(_FIELD_ALIASES)}"
        )
    field = _FIELD_ALIASES[field_raw]
    subkey = None
    i += 1
    if i < len(tokens) and tokens[i][0] == "DOT":
        i += 1
        if i >= len(tokens) or tokens[i][0] != "IDENT":
            raise FilterParseError(
                f"expected subkey after '.'; got {tokens[i:i+1]}"
            )
        subkey = tokens[i][1]
        i += 1

    # operator
    if i >= len(tokens) or tokens[i][0] != "OP":
        raise FilterParseError(
            f"expected operator after field; got {tokens[i:i+1]}"
        )
    op = tokens[i][1]
    i += 1

    # value (or list for ``in``)
    values: List[Any] = []
    if op == "in":
        # comma-separated list
        while True:
            if i >= len(tokens):
                raise FilterParseError("unexpected end after 'in'")
            t, v = tokens[i]
            if t == "STR":
                values.append(v)
            elif t == "NUM":
                values.append(float(v) if "." in v else int(v))
            elif t == "IDENT":
                values.append(v)
            else:
                raise FilterParseError(f"bad value in 'in' list: {tokens[i]}")
            i += 1
            if i < len(tokens) and tokens[i][0] == "COMMA":
                i += 1
                continue
            break
    else:
        if i >= len(tokens):
            raise FilterParseError("unexpected end of expression")
        t, v = tokens[i]
        if t == "STR":
            values = [v]
        elif t == "NUM":
            values = [float(v) if "." in v else int(v)]
        elif t == "IDENT":
            # Bare identifier as string — ``property=bandgap`` form.
            values = [v]
        else:
            raise FilterParseError(f"bad RHS: {tokens[i]}")
        i += 1

    pred = _build_predicate(field, subkey, op, values)
    return pred, i


def _build_predicate(
    field: str, subkey: str | None, op: str, values: List[Any],
) -> Callable[[PropertyRow], bool]:
    def extract(row: PropertyRow) -> Any:
        base = getattr(row, field, None)
        if subkey is not None:
            if isinstance(base, dict):
                return base.get(subkey)
            return None
        return base

    if op == "in":
        allowed = set(values)
        return lambda r: extract(r) in allowed

    rhs = values[0]

    def cmp(r: PropertyRow) -> bool:
        lhs = extract(r)
        if op == "=":
            return lhs == rhs
        if op == "!=":
            return lhs != rhs
        # Numeric comparisons — coerce both sides.
        try:
            return _numeric_cmp(lhs, rhs, op)
        except (TypeError, ValueError):
            return False

    return cmp


def _numeric_cmp(lhs: Any, rhs: Any, op: str) -> bool:
    l = float(lhs)
    r = float(rhs)
    if op == ">":
        return l > r
    if op == "<":
        return l < r
    if op == ">=":
        return l >= r
    if op == "<=":
        return l <= r
    raise ValueError(f"non-numeric operator {op!r}")


def compile_filter(expression: str) -> Callable[[PropertyRow], bool]:
    """Turn a filter string into a ``(PropertyRow) -> bool`` predicate.

    Empty / whitespace-only expressions produce a universal predicate
    (match everything), which is useful for tests and for CLI callers
    who don't want to special-case "no filter".
    """
    if not expression or not expression.strip():
        return lambda _r: True
    tokens = _tokenize(expression)
    predicates: List[Callable[[PropertyRow], bool]] = []
    i = 0
    while i < len(tokens):
        pred, i = _parse_atom(tokens, i)
        predicates.append(pred)
        if i < len(tokens):
            if tokens[i][0] != "AND":
                raise FilterParseError(
                    f"expected 'AND' between clauses, got {tokens[i]}"
                )
            i += 1
    if not predicates:
        raise FilterParseError("empty expression after tokenization")
    return lambda r: all(p(r) for p in predicates)


def apply_filter(
    rows: Iterable[PropertyRow], expression: str,
) -> List[PropertyRow]:
    """Return rows matching the expression, order-preserving."""
    pred = compile_filter(expression)
    return [r for r in rows if pred(r)]
