from enum import Enum


class NodeType(Enum):
    REAL = "REAL"
    HALLUCINATED = "HALLUCINATED"


class EdgeType(Enum):
    REAL = "REAL"
    HALLUCINATED = "HALLUCINATED"
    ILLEGAL = "ILLEGAL"  # Obtained from tool with no return value
    CONDITIONAL = "CONDITIONAL"
