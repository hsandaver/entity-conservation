"""Data models for graph structures."""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class Edge:
    source: str
    target: str
    relationship: str
    inferred: bool = False
    weight: Optional[float] = None


@dataclass
class Node:
    id: str
    label: str
    types: List[str]
    metadata: Any
    edges: List[Edge] = field(default_factory=list)


@dataclass
class GraphData:
    nodes: List[Node]
