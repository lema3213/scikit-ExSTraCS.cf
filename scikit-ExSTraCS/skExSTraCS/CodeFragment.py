import math
import random
import re
import pandas as pd

from skExSTraCS.TreePrint import build_tree_from_rpn

class CodeFragment:
    OPERATOR_ARITY = {
        '+': 2,
        '-': 2,
        '*': 2,
        '/': 2,
        's': 1,
        'c': 1,

    }

    MAX_DEPTH = 2

    """
    GP tree node class:
    - value: node value (string), could be operator or variable name.
    - children: list of child nodes, unary operators (like sin) have 1 child, binary operators (like +, -, *, /) have 2 children.
    """
    def __init__(self,value, children=None, level=0,position = None):
        self.value = value
        self.children = children if children else []
        self.level = level
        self.position = position

    def __str__(self):
        """
        Directly call toPostfix(), display postfix expression when printing
        """
        return self.toPostfix()

    def printTree(self):
        expr = self.toPostfix()
        tree = build_tree_from_rpn(expr)
        print(tree)

    def toPostfix(self):
        """
        Return postfix expression string (RPN).
        Using post-order traversal: traverse child nodes first, then add current node's operator.
        """
        if not self.children:
            # No children, means it's a variable
            return str(self.value)

        # If it's a unary operator
        if len(self.children) == 1:
            return f"{self.children[0].toPostfix()} {self.value}"

        # If it's a binary operator
        if len(self.children) == 2:
            left_expr = self.children[0].toPostfix()
            right_expr = self.children[1].toPostfix()
            return f"{left_expr} {right_expr} {self.value}"

        # Theoretically shouldn't reach here
        return str(self.value)

    @staticmethod
    def createCodeFragment(variables, level=1):
        return CodeFragment._generateRandomTree(variables, max_level=level,current_level=level)

    @staticmethod
    def _generateRandomTree(variables,max_level,current_level=0,max_depth=2,current_depth=0):
        # Terminal stop condition:
        # - Stop if current_depth reaches MAX_DEPTH, OR
        # - Stop early with 50% probability (random.random() > 0.5)
        if current_depth == CodeFragment.MAX_DEPTH or (random.random() > 0.5):
            # Level-1 CF: terminal must be a raw feature D<idx>, except when current_depth == 0。
            if current_level == 1:
                if current_depth == 0:
                    pass # avoid creating one feature(without operator) condition
                else:
                    position = random.choice(variables)
                    return CodeFragment('D' + str(position), position=position)
            else:
                # For higher-level CFs:
                # - If we are generating at the top level (current_level == max_level),
                #   we allow directly choosing a raw feature with 50% probability.
                if max_level == current_level and random.random() > 0.5:
                    position = random.choice(variables)
                    return CodeFragment('D' + str(position), position=position)
                else:
                    # Otherwise, choose a lower-level CF (1 .. current_level-1) and reuse it
                    lower_level = random.choice(list(range(1,current_level)))

                    # no lower cf, just generate randomly.
                    child = CodeFragment._generateRandomTree(variables,max_level,current_level = lower_level)
                    return child

        # Non-terminal: randomly select an operator and its arity from OPERATOR_ARITY
        op, arity = random.choice(list(CodeFragment.OPERATOR_ARITY.items()))

        if arity == 1:
            # Unary operator: generate one child subtree
            child = CodeFragment._generateRandomTree(variables,max_level = max_level, current_level=current_level,current_depth= current_depth + 1)
            return CodeFragment(op, [child])
        elif arity == 2:
            # Binary operator: generate left and right child subtrees
            left_child = CodeFragment._generateRandomTree(variables,max_level = max_level, current_level=current_level, current_depth=current_depth + 1)
            right_child = CodeFragment._generateRandomTree(variables,max_level = max_level, current_level=current_level, current_depth=current_depth + 1)
            return CodeFragment(op, [left_child, right_child])
        else:
            # Guard: OPERATOR_ARITY should only contain unary/binary operators
            raise Exception('Invalid arity')

    @staticmethod
    def evaluate(cf, state):
        return CodeFragment.evaluateTree(cf, state)

    @staticmethod
    def evaluateTree(cf, state):
        # If it's a leaf node (variable), directly get value from context
        if not cf.children:
            return state[cf.position]

        # Otherwise it's an operator node
        op = cf.value
        arity = CodeFragment.OPERATOR_ARITY[op]

        if arity == 1:
            # Unary operator (sin)
            val = CodeFragment.evaluateTree(cf.children[0], state)
            if op == 's':
                return math.sin(val)
            elif op == 'c':
                return math.cos(val)
            elif op == 'a':
                return math.fabs(val)
            elif op == '~':
                return -val

        elif arity == 2:
            # Binary operators (+, -, *, /)
            left_val = CodeFragment.evaluateTree(cf.children[0], state)
            right_val = CodeFragment.evaluateTree(cf.children[1], state)

            if op == '+':
                return left_val + right_val
            elif op == '-':
                return left_val - right_val
            elif op == '*':
                return left_val * right_val
            elif op == '/':
                # Simple handling of division by 0
                if right_val == 0:
                    return float(0)
                return left_val / right_val
        else:
            raise Exception('Invalid arity')

    @staticmethod
    def fromPostfix(postfix: str):
        """
        Parse a postfix (RPN) string back into a CodeFragment tree.

        Terminals:
          - D<int> (e.g., D3, D16) -> leaf node, position=<int>

        Operators:
          - Must exist in CodeFragment.OPERATOR_ARITY
          - Unary operators pop 1 operand
          - Binary operators pop 2 operands (order matters: left then right)

        Returns:
          - The root CodeFragment node

        Raises:
          - ValueError for invalid postfix strings or unknown tokens
        """
        if postfix is None:
            raise ValueError("postfix is None")

        tokens = postfix.strip().split()
        if not tokens:
            raise ValueError("Empty postfix string")

        term_pat = re.compile(r"^D(\d+)$")
        stack = []

        for tok in tokens:
            # Operator token
            if tok in CodeFragment.OPERATOR_ARITY:
                arity = CodeFragment.OPERATOR_ARITY[tok]
                if len(stack) < arity:
                    raise ValueError(f"Invalid postfix: not enough operands for operator '{tok}'")

                if arity == 1:
                    child = stack.pop()
                    stack.append(CodeFragment(tok, [child]))

                elif arity == 2:
                    right = stack.pop()
                    left = stack.pop()
                    stack.append(CodeFragment(tok, [left, right]))

                else:
                    # Reserved for future multi-arity operators (e.g., arity=4)
                    children = [stack.pop() for _ in range(arity)][::-1]
                    stack.append(CodeFragment(tok, children))

                continue

            # Terminal token (leaf), e.g., D16
            m = term_pat.match(tok)
            if m:
                pos = int(m.group(1))
                stack.append(CodeFragment('D' + str(pos), position=pos))
                continue

            raise ValueError(f"Unknown token in postfix: '{tok}'")

        if len(stack) != 1:
            raise ValueError(f"Invalid postfix: stack has {len(stack)} items after parsing (expected 1)")

        return stack[0]