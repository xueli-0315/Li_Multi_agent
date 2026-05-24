from __future__ import annotations

from dataclasses import dataclass
from typing import Optional as Opt

from pyparsing import Combine
from pyparsing import Forward
from pyparsing import Literal
from pyparsing import Optional
from pyparsing import ParseException
from pyparsing import ParseResults
from pyparsing import ParserElement
from pyparsing import Regex
from pyparsing import Word
from pyparsing import alphanums
from pyparsing import alphas
from pyparsing import delimitedList
from pyparsing import infixNotation
from pyparsing import oneOf
from pyparsing import opAssoc

ParserElement.enablePackrat()


@dataclass
class Node:
    pass


@dataclass
class VarNode(Node):
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass
class NumberNode(Node):
    value: float

    def __str__(self) -> str:
        return str(self.value)


@dataclass
class FunctionNode(Node):
    name: str
    args: list[Node]

    def __str__(self) -> str:
        return f"{self.name}({', '.join(str(arg) for arg in self.args)})"


@dataclass
class BinaryOpNode(Node):
    op: str
    left: Node
    right: Node

    def __str__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


@dataclass
class ConditionalNode(Node):
    condition: Node
    true_expr: Node
    false_expr: Node

    def __str__(self) -> str:
        return f"({self.condition} ? {self.true_expr} : {self.false_expr})"


@dataclass
class UnaryOpNode(Node):
    op: str
    operand: Node

    def __str__(self) -> str:
        return f"({self.op}{self.operand})"


var = Combine(Optional(Literal("$")) + Word(alphas, alphanums + "_"))
number = Regex(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?")
mul_div = oneOf("* /")
add_sub = oneOf("+ -")
comparison = oneOf("> < >= <= == !=")
logical_and = oneOf("&& &")
logical_or = oneOf("|| |")
conditional = ("?", ":")


def _unwrap(arg: object) -> object:
    if isinstance(arg, (list, ParseResults)):
        if len(arg) == 1:
            return _unwrap(arg[0])
        return [_unwrap(item) for item in arg]
    return arg


def _create_var_node(tokens: ParseResults) -> VarNode:
    return VarNode(tokens[0])


def _create_number_node(tokens: ParseResults) -> NumberNode:
    return NumberNode(float(tokens[0]))


def _create_function_node(tokens: ParseResults) -> FunctionNode:
    name = tokens[0]
    args_raw = tokens[2:-1]
    processed_args = [_unwrap(arg) for arg in args_raw]
    if not all(isinstance(arg, Node) for arg in processed_args):
        raise ValueError(f"Invalid function args: {processed_args}")
    return FunctionNode(name=name, args=processed_args)


def _create_binary_op_node(tokens: ParseResults) -> Node:
    values = tokens[0]
    if len(values) == 3:
        left = _unwrap(values[0])
        right = _unwrap(values[2])
        if not isinstance(left, Node) or not isinstance(right, Node):
            raise ValueError("Invalid binary operand")
        return BinaryOpNode(op=values[1], left=left, right=right)

    first = _unwrap(values[0])
    if not isinstance(first, Node):
        raise ValueError("Invalid binary operand")
    result: Node = first
    for i in range(1, len(values) - 1, 2):
        right = _unwrap(values[i + 1])
        if not isinstance(right, Node):
            raise ValueError("Invalid binary operand")
        result = BinaryOpNode(op=values[i], left=result, right=right)
    return result


def _create_conditional_node(tokens: ParseResults) -> ConditionalNode:
    values = tokens[0]
    condition = _unwrap(values[0])
    true_expr = _unwrap(values[2])
    false_expr = _unwrap(values[4])
    if not isinstance(condition, Node) or not isinstance(true_expr, Node) or not isinstance(false_expr, Node):
        raise ValueError("Invalid conditional operand")
    return ConditionalNode(condition=condition, true_expr=true_expr, false_expr=false_expr)


def _create_unary_op_node(tokens: ParseResults) -> UnaryOpNode:
    values = tokens[0]
    operand = _unwrap(values[1])
    if not isinstance(operand, Node):
        raise ValueError("Invalid unary operand")
    return UnaryOpNode(op=values[0], operand=operand)


expr = Forward()
var.setParseAction(_create_var_node)
number.setParseAction(_create_number_node)
function_call = var + "(" + Optional(delimitedList(expr)) + ")"
function_call.setParseAction(_create_function_node)
operand = function_call | var | number | ("(" + expr + ")").setParseAction(lambda t: t[1])
unary_minus = Literal("-")
expr <<= infixNotation(
    operand,
    [
        (unary_minus, 1, opAssoc.RIGHT, _create_unary_op_node),
        (mul_div, 2, opAssoc.LEFT, _create_binary_op_node),
        (add_sub, 2, opAssoc.LEFT, _create_binary_op_node),
        (comparison, 2, opAssoc.LEFT, _create_binary_op_node),
        (logical_and, 2, opAssoc.LEFT, _create_binary_op_node),
        (logical_or, 2, opAssoc.LEFT, _create_binary_op_node),
        (conditional, 3, opAssoc.RIGHT, _create_conditional_node),
    ],
)


def parse_expression(text: str) -> Node:
    try:
        result = expr.parseString(text, parseAll=True)
        parsed = result[0]
        if not isinstance(parsed, Node):
            raise ValueError("Parsed result is not AST node")
        return parsed
    except ParseException as exc:
        raise ValueError(f"Invalid expression syntax: {exc}") from exc


def are_nodes_equal(node1: Node, node2: Node) -> bool:
    if type(node1) is not type(node2):
        return False
    if isinstance(node1, NumberNode):
        return node1.value == node2.value
    if isinstance(node1, VarNode):
        return node1.name == node2.name
    if isinstance(node1, FunctionNode):
        return node1.name == node2.name and len(node1.args) == len(node2.args)
    if isinstance(node1, BinaryOpNode):
        return node1.op == node2.op
    if isinstance(node1, ConditionalNode):
        return True
    if isinstance(node1, UnaryOpNode):
        return node1.op == node2.op
    return False


@dataclass
class SubtreeMatch:
    root1: Node
    root2: Node
    size: int


def find_largest_common_subtree(root1: Node, root2: Node) -> Opt[SubtreeMatch]:
    def get_subtree_size(node: Node) -> int:
        if isinstance(node, (NumberNode, VarNode)):
            return 1
        if isinstance(node, FunctionNode):
            return 1 + sum(get_subtree_size(arg) for arg in node.args)
        if isinstance(node, BinaryOpNode):
            return 1 + get_subtree_size(node.left) + get_subtree_size(node.right)
        if isinstance(node, ConditionalNode):
            return 1 + get_subtree_size(node.condition) + get_subtree_size(node.true_expr) + get_subtree_size(node.false_expr)
        if isinstance(node, UnaryOpNode):
            return 1 + get_subtree_size(node.operand)
        return 0

    def get_all_subtrees(node: Node) -> list[Node]:
        result = [node]
        if isinstance(node, FunctionNode):
            for arg in node.args:
                result.extend(get_all_subtrees(arg))
        elif isinstance(node, BinaryOpNode):
            result.extend(get_all_subtrees(node.left))
            result.extend(get_all_subtrees(node.right))
        elif isinstance(node, ConditionalNode):
            result.extend(get_all_subtrees(node.condition))
            result.extend(get_all_subtrees(node.true_expr))
            result.extend(get_all_subtrees(node.false_expr))
        elif isinstance(node, UnaryOpNode):
            result.extend(get_all_subtrees(node.operand))
        return result

    def is_commutative_op(op: str) -> bool:
        return op in {"+", "*", "==", "!=", "&", "&&", "|", "||"}

    def are_subtrees_equal(node1: Node, node2: Node) -> bool:
        if not are_nodes_equal(node1, node2):
            return False
        if isinstance(node1, (NumberNode, VarNode)):
            return True
        if isinstance(node1, FunctionNode):
            return all(are_subtrees_equal(a1, a2) for a1, a2 in zip(node1.args, node2.args))
        if isinstance(node1, BinaryOpNode):
            if is_commutative_op(node1.op):
                return (are_subtrees_equal(node1.left, node2.left) and are_subtrees_equal(node1.right, node2.right)) or (
                    are_subtrees_equal(node1.left, node2.right) and are_subtrees_equal(node1.right, node2.left)
                )
            return are_subtrees_equal(node1.left, node2.left) and are_subtrees_equal(node1.right, node2.right)
        if isinstance(node1, ConditionalNode):
            return (
                are_subtrees_equal(node1.condition, node2.condition)
                and are_subtrees_equal(node1.true_expr, node2.true_expr)
                and are_subtrees_equal(node1.false_expr, node2.false_expr)
            )
        if isinstance(node1, UnaryOpNode):
            return are_subtrees_equal(node1.operand, node2.operand)
        return False

    subtrees1 = get_all_subtrees(root1)
    subtrees2 = get_all_subtrees(root2)
    max_match: Opt[SubtreeMatch] = None
    max_size = 0

    for st1 in subtrees1:
        size1 = get_subtree_size(st1)
        if size1 <= max_size:
            continue
        for st2 in subtrees2:
            size2 = get_subtree_size(st2)
            if size1 != size2 or size2 <= max_size:
                continue
            if are_subtrees_equal(st1, st2):
                max_size = size1
                max_match = SubtreeMatch(st1, st2, size1)
    return max_match


def compare_expressions(expr1: str, expr2: str) -> Opt[SubtreeMatch]:
    tree1 = parse_expression(expr1)
    tree2 = parse_expression(expr2)
    return find_largest_common_subtree(tree1, tree2)


def match_expression_zoo(expression: str, expression_zoo: list[str]) -> tuple[int, str, str]:
    max_size = 0
    matched_subtree = ""
    matched_expression = ""
    for ref_expr in expression_zoo:
        try:
            match = compare_expressions(expression, ref_expr)
        except Exception:
            continue
        if match is not None and match.size > max_size:
            max_size = match.size
            matched_subtree = str(match.root1)
            matched_expression = ref_expr
    return max_size, matched_subtree, matched_expression


def count_free_args(expression: str) -> int:
    tree = parse_expression(expression)
    return _count_number_nodes(tree)


def _count_number_nodes(node: Node) -> int:
    if isinstance(node, NumberNode):
        return 1
    if isinstance(node, VarNode):
        return 0
    if isinstance(node, FunctionNode):
        return sum(_count_number_nodes(arg) for arg in node.args)
    if isinstance(node, BinaryOpNode):
        return _count_number_nodes(node.left) + _count_number_nodes(node.right)
    if isinstance(node, ConditionalNode):
        return _count_number_nodes(node.condition) + _count_number_nodes(node.true_expr) + _count_number_nodes(node.false_expr)
    if isinstance(node, UnaryOpNode):
        return _count_number_nodes(node.operand)
    return 0


def count_unique_vars(expression: str) -> int:
    tree = parse_expression(expression)
    unique_vars: set[str] = set()
    _collect_unique_vars(tree, unique_vars)
    return len(unique_vars)


def _collect_unique_vars(node: Node, unique_vars: set[str]) -> None:
    if isinstance(node, VarNode):
        if node.name.startswith("$"):
            unique_vars.add(node.name)
        return
    if isinstance(node, NumberNode):
        return
    if isinstance(node, FunctionNode):
        for arg in node.args:
            _collect_unique_vars(arg, unique_vars)
        return
    if isinstance(node, BinaryOpNode):
        _collect_unique_vars(node.left, unique_vars)
        _collect_unique_vars(node.right, unique_vars)
        return
    if isinstance(node, ConditionalNode):
        _collect_unique_vars(node.condition, unique_vars)
        _collect_unique_vars(node.true_expr, unique_vars)
        _collect_unique_vars(node.false_expr, unique_vars)
        return
    if isinstance(node, UnaryOpNode):
        _collect_unique_vars(node.operand, unique_vars)


def count_all_nodes(expression: str) -> int:
    tree = parse_expression(expression)
    return _count_nodes(tree)


def _count_nodes(node: Node) -> int:
    if isinstance(node, (NumberNode, VarNode)):
        return 1
    if isinstance(node, FunctionNode):
        return 1 + sum(_count_nodes(arg) for arg in node.args)
    if isinstance(node, BinaryOpNode):
        return 1 + _count_nodes(node.left) + _count_nodes(node.right)
    if isinstance(node, ConditionalNode):
        return 1 + _count_nodes(node.condition) + _count_nodes(node.true_expr) + _count_nodes(node.false_expr)
    if isinstance(node, UnaryOpNode):
        return 1 + _count_nodes(node.operand)
    return 0


def calculate_symbol_length(expression: str) -> int:
    return len(expression.strip())


def count_base_features(expression: str) -> int:
    tree = parse_expression(expression)
    base_features: set[str] = set()
    _collect_base_features(tree, base_features)
    return len(base_features)


def _collect_base_features(node: Node, base_features: set[str]) -> None:
    if isinstance(node, VarNode):
        if node.name.startswith("$"):
            base_features.add(node.name)
        return
    if isinstance(node, NumberNode):
        return
    if isinstance(node, FunctionNode):
        for arg in node.args:
            _collect_base_features(arg, base_features)
        return
    if isinstance(node, BinaryOpNode):
        _collect_base_features(node.left, base_features)
        _collect_base_features(node.right, base_features)
        return
    if isinstance(node, ConditionalNode):
        _collect_base_features(node.condition, base_features)
        _collect_base_features(node.true_expr, base_features)
        _collect_base_features(node.false_expr, base_features)
        return
    if isinstance(node, UnaryOpNode):
        _collect_base_features(node.operand, base_features)
