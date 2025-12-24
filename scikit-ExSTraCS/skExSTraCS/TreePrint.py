# cf_tree.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Supported operators and their arity (number of operands).
DEFAULT_ARITY: Dict[str, int] = {
    "&": 2,
    "|": 2,
    "~": 1,
    "nand": 2,
    "nor": 2,
}


@dataclass
class Node:
    label: str
    children: List["Node"] = field(default_factory=list)

    def __str__(self) -> str:
        """Render the tree as an ASCII diagram (triggered by print(node))."""
        lines = [self.label]
        for i, ch in enumerate(self.children):
            last = (i == len(self.children) - 1)
            lines.extend(ch._to_lines(prefix="", is_last=last))
        return "\n".join(lines)

    def _to_lines(self, prefix: str, is_last: bool) -> List[str]:
        """Internal helper to build ASCII tree lines with proper connectors."""
        connector = "└── " if is_last else "├── "
        lines = [prefix + connector + self.label]

        next_prefix = prefix + ("    " if is_last else "│   ")
        for i, ch in enumerate(self.children):
            last = (i == len(self.children) - 1)
            lines.extend(ch._to_lines(prefix=next_prefix, is_last=last))
        return lines

    def to_parenthesized(self) -> str:
        """Render the tree in parenthesized (S-expression-like) form."""
        if not self.children:
            return self.label
        inner = " ".join(child.to_parenthesized() for child in self.children)
        return f"({self.label} {inner})"


def build_tree_from_rpn(expr: str, arity: Optional[Dict[str, int]] = None) -> Node:
    """
    Build an expression tree from a postfix (RPN) string.

    Example:
        "D16 D15 ~ D14 nor nand D16 D16 nor &"
    """
    arity = DEFAULT_ARITY if arity is None else arity

    stack: List[Node] = []
    for tok in expr.split():
        # Optional cleanup for tokens like "D3," (comma attached).
        tok = tok.rstrip(",")

        if tok in arity:
            k = arity[tok]
            if len(stack) < k:
                raise ValueError(f"Not enough operands for operator '{tok}' (need {k})")

            # In RPN, the right operand is popped first, then the left operand.
            if k == 1:
                a = stack.pop()
                stack.append(Node(tok, [a]))
            elif k == 2:
                b = stack.pop()
                a = stack.pop()
                stack.append(Node(tok, [a, b]))
            else:
                # Generic k-ary operator support (keeps left-to-right order).
                args = [stack.pop() for _ in range(k)][::-1]
                stack.append(Node(tok, args))
        else:
            stack.append(Node(tok))

    if len(stack) != 1:
        raise ValueError(f"Invalid RPN expression: stack size = {len(stack)}")
    return stack[0]


__all__ = ["Node", "build_tree_from_rpn", "DEFAULT_ARITY"]
