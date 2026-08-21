"""Extract and compare boxed mathematical answers.

The equivalence rules are based on DeepScaleR's math reward:
https://github.com/agentica-project/deepscaler/blob/e6080ccd974eb64bd3430f0b36108244a6fee330/deepscaler/rewards/math_utils/utils.py
"""

from __future__ import annotations

import re

import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser


_UNSAFE_SUBSTRINGS = ("^{", "^(")
_UNSAFE_PATTERNS = (r"\^[0-9]+\^", r"\^[0-9][0-9]+")
_TUPLE_BRACKETS = "()[]"
_FRACTION_RE = re.compile(r"-?[0-9]+.?/0*[1-9][0-9]*.?")
_FORMATTED_COMMA_RE = re.compile(r"(\d),(\d\d\d)($|\D)")
_UNITS = (
    "degree",
    "cm",
    "centimeter",
    "meter",
    "mile",
    "second",
    "minute",
    "hour",
    "day",
    "week",
    "month",
    "year",
    "foot",
    "feet",
    "inch",
    "yard",
)


# Public API


def answers_match(solution: str, reference: str) -> bool:
    """Return whether the last boxed answer matches the reference answer."""

    answer = _extract_boxed_answer(solution)
    if answer is None:
        return False

    boxed_reference = _extract_boxed_answer(reference)
    if boxed_reference is not None:
        reference = boxed_reference

    normalized_answer = _normalize_answer_text(answer)
    normalized_reference = _normalize_answer_text(reference)
    if normalized_answer == normalized_reference:
        return True

    # Use symbolic comparison only when normalized text differs.
    return _match_sympy(answer, reference)


# Boxed answer extraction


def _extract_boxed_answer(text: str) -> str | None:
    r"""Extract the contents of the last ``\boxed{}`` or ``\fbox{}``."""

    command_start = max(text.rfind(r"\boxed"), text.rfind(r"\fbox"))
    if command_start < 0:
        return None

    opening_brace = text.find("{", command_start)
    if opening_brace < 0:
        return None

    depth = 0
    for index in range(opening_brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[opening_brace + 1 : index]

    return None


# String normalization


def _normalize_answer_text(expression: str) -> str:
    expression = expression.strip()
    text_match = re.fullmatch(r"\\text\{(.+?)\}", expression)
    if text_match:
        expression = text_match.group(1).strip()

    expression = expression.replace("\n", "")
    expression = expression.replace(r"\!", "")
    expression = expression.replace("\\\\", "\\")
    expression = expression.replace("tfrac", "frac")
    expression = expression.replace("dfrac", "frac")
    expression = expression.replace(r"\left", "")
    expression = expression.replace(r"\right", "")
    expression = expression.replace(r"^{\circ}", "")
    expression = expression.replace(r"^\circ", "")
    expression = expression.replace(r"\$", "")
    if r"\text{ " in expression:
        expression = expression.split(r"\text{ ", 1)[0]
    expression = expression.replace(r"\%", "")
    expression = expression.replace(" .", " 0.")
    expression = expression.replace("{.", "{0.")

    if not expression:
        return expression
    if expression.startswith("."):
        expression = "0" + expression

    equation = expression.split("=")
    if len(equation) == 2 and len(equation[0]) <= 2:
        expression = equation[1]

    expression = _fix_square_roots(expression)
    expression = expression.replace(" ", "")
    expression = _fix_fractions(expression)

    if expression == "0.5":
        expression = r"\frac{1}{2}"

    return _fix_simple_fraction(expression)


def _fix_square_roots(expression: str) -> str:
    parts = expression.split(r"\sqrt")
    if len(parts) == 1:
        return expression

    fixed = parts[0]
    for suffix in parts[1:]:
        if not suffix:
            return expression
        if suffix.startswith("{"):
            fixed += r"\sqrt" + suffix
        else:
            fixed += r"\sqrt{" + suffix[0] + "}" + suffix[1:]
    return fixed


def _fix_fractions(expression: str) -> str:
    parts = expression.split(r"\frac")
    if len(parts) == 1:
        return expression

    fixed = parts[0]
    for suffix in parts[1:]:
        fixed += r"\frac"
        if suffix.startswith("{"):
            fixed += suffix
        elif len(suffix) < 2:
            return expression
        elif suffix[1] == "{":
            fixed += "{" + suffix[0] + "}" + suffix[1:]
        else:
            fixed += "{" + suffix[0] + "}{" + suffix[1] + "}" + suffix[2:]
    return fixed


def _fix_simple_fraction(expression: str) -> str:
    parts = expression.split("/")
    if len(parts) != 2:
        return expression

    try:
        numerator, denominator = map(int, parts)
    except ValueError:
        return expression

    if expression != f"{numerator}/{denominator}":
        return expression
    return rf"\frac{{{numerator}}}{{{denominator}}}"


# Symbolic comparison


def _match_sympy(answer: str, reference: str) -> bool:
    answer = _normalize_for_sympy(answer)
    reference = _normalize_for_sympy(reference)

    if reference == answer:
        return True
    if not answer:
        return False

    reference_parts = _split_tuple(reference)
    answer_parts = _split_tuple(answer)
    if len(reference_parts) != len(answer_parts):
        return False

    if len(reference_parts) > 1 and (
        reference[0] != answer[0] or reference[-1] != answer[-1]
    ):
        return False

    for expected, actual in zip(reference_parts, answer_parts):
        if _FRACTION_RE.fullmatch(expected) and _FRACTION_RE.fullmatch(actual):
            equal = expected == actual
        elif _is_integer_text(expected) != _is_integer_text(actual):
            equal = False
        else:
            equal = _are_sympy_equivalent(expected, actual)

        if not equal:
            return False

    return True


def _normalize_for_sympy(expression: str) -> str:
    text_match = re.fullmatch(r"\\text\{(.+?)\}", expression)
    if text_match:
        expression = text_match.group(1)

    expression = expression.replace(r"\%", "%")
    expression = expression.replace(r"\$", "$")
    expression = expression.replace("$", "")
    expression = expression.replace("%", "")
    expression = expression.replace(" or ", " , ")
    expression = expression.replace(" and ", " , ")
    expression = expression.replace("million", "*10^6")
    expression = expression.replace("billion", "*10^9")
    expression = expression.replace("trillion", "*10^12")

    for unit in _UNITS:
        expression = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", expression)
    expression = re.sub(r"\^ *\\circ", "", expression)

    if expression.startswith("{") and expression.endswith("}"):
        expression = expression[1:-1]

    expression = re.sub(r",\\! *", "", expression)
    try:
        number = float(expression)
    except ValueError:
        pass
    else:
        if _is_integer(number):
            expression = str(int(round(number)))

    if "\\" in expression:
        try:
            expression = _latex_to_text(expression)
        except Exception:
            pass

    expression = re.sub(r"- *", "-", expression)
    expression = re.sub(r"([0-9]) +([0-9])", r"\1+\2", expression)
    expression = expression.replace(" ", "")
    expression = expression.replace("{", "")
    expression = expression.replace("}", "")
    expression = expression.lower()

    if _is_integer_text(expression):
        expression = str(int(float(expression.replace(",", ""))))

    return expression


def _latex_to_text(expression: str) -> str:
    expression = expression.replace(r"\tfrac", r"\frac")
    expression = expression.replace(r"\dfrac", r"\frac")
    expression = expression.replace(r"\frac", r" \frac")
    expression = latex2text.LatexNodes2Text().latex_to_text(expression)

    replacements = {
        "√": "sqrt",
        "π": "pi",
        "∞": "inf",
        "∪": "U",
        "·": "*",
        "×": "*",
    }
    for source, target in replacements.items():
        expression = expression.replace(source, target)
    return expression.strip()


def _split_tuple(expression: str) -> list[str]:
    expression = _strip_formatted_commas(expression)
    if not expression:
        return []

    is_tuple = (
        len(expression) > 2
        and expression[0] in _TUPLE_BRACKETS
        and expression[-1] in _TUPLE_BRACKETS
        and all(char not in expression[1:-1] for char in _TUPLE_BRACKETS)
    )
    if not is_tuple:
        return [expression]
    return [item.strip() for item in expression[1:-1].split(",")]


def _are_sympy_equivalent(expected: str, actual: str) -> bool:
    difference = f"({expected})-({actual})"
    if not _is_safe_to_evaluate(difference):
        return False

    try:
        parsed = sympy_parser.parse_expr(
            difference.replace("^", "**"),
            transformations=(
                sympy_parser.standard_transformations
                + (sympy_parser.implicit_multiplication_application,)
            ),
        )
        return sympy.simplify(parsed) == 0
    except Exception:
        return False


def _is_safe_to_evaluate(expression: str) -> bool:
    letters = expression.replace("sqrt", "").replace("frac", "")
    if len({char for char in letters if char.isalpha()}) > 2:
        return False
    if any(text in expression for text in _UNSAFE_SUBSTRINGS):
        return False
    return not any(re.search(pattern, expression) for pattern in _UNSAFE_PATTERNS)


# Number helpers


def _is_integer(value: float) -> bool:
    try:
        return abs(value - int(round(value))) <= 1e-7
    except (OverflowError, ValueError):
        return False


def _is_integer_text(value: str) -> bool:
    try:
        number = float(_strip_formatted_commas(value))
    except ValueError:
        return False
    return _is_integer(number)


def _strip_formatted_commas(expression: str) -> str:
    while True:
        cleaned = _FORMATTED_COMMA_RE.sub(r"\1\2\3", expression)
        if cleaned == expression:
            return expression
        expression = cleaned
