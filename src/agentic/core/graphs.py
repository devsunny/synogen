from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from agentic.core.agents import Agent, START, END
from agentic.core.tools import function_to_openai_tool

# -------------------------------------------------------------------
# Adapter: allow either Agent or plain callable to be used as a node
# -------------------------------------------------------------------


class Runnable:
    """Minimal runner interface for graph execution."""

    def name(self) -> str: ...
    def description(self) -> Optional[str]: ...
    def metadata(self) -> Dict[str, Any]: ...
    def run(self, *args, context: Dict[str, Any] = None, **kwargs) -> Any: ...


class AgentAdapter(Runnable):
    def __init__(self, agent: Agent):
        self._agent = agent

    def name(self) -> str:
        return self._agent.get_agent_name()

    def description(self) -> Optional[str]:
        return self._agent.get_agent_description()

    def metadata(self) -> Dict[str, Any]:
        # Leverage Agent's OpenAI tool metadata for orchestration systems
        return self._agent.get_metadata()

    def run(self, *args, context: Dict[str, Any] = None, **kwargs) -> Any:
        if self._agent.stateless:
            # Stateless: do not pass context
            return self._agent.run(*args, **kwargs)
        else:
            # Stateful: pass context as first arg
            if context is None:
                context = {}
            kwargs["context"] = context  # inject context
        return self._agent.run(*args, **kwargs)


class CallableAdapter(Runnable):
    def __init__(
        self,
        fn: Callable[..., Any],
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        if not callable(fn):
            raise TypeError("fn must be callable")
        self._fn = fn
        self._name = name or getattr(fn, "__name__", "callable")
        self._description = description or (
            fn.__doc__.strip() if getattr(fn, "__doc__", None) else None
        )
        self._metadata = getattr(fn, "__openai_schema__", None)
        self._stateless = getattr(fn, "__tool_stateless__", False)

    def name(self) -> str:
        return self._name

    def description(self) -> Optional[str]:
        return self._description

    def metadata(self) -> Dict[str, Any]:
        if self._metadata is None:
            # Minimal metadata; expand to your tool schema as needed
            self._metadata, self._stateless = function_to_openai_tool(
                self._fn, name=self._name, description=self._description
            )
        return self._metadata

    def run(self, *args, context: Dict[str, Any] = None, **kwargs) -> Any:
        if self._stateless:
            # Stateless: do not pass context
            return self._fn(*args, **kwargs)
        else:
            # Stateful: pass context as first arg
            if context is None:
                context = {}
            kwargs["context"] = context  # inject context
        return self._fn(*args, **kwargs)


# -------------------------------------------------------------------
# Graph model
# -------------------------------------------------------------------

NodeLike = Union[Agent, Callable[..., Any], Runnable]


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    label: Optional[str] = None  # optional edge label (e.g., condition name)


@dataclass
class Node:
    id: str
    runner: Runnable
    kind: str  # "agent" | "callable"
    name: str
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Graph:
    """
    Executable directed acyclic graph.
    Execution model:
      - Kahn topological order.
      - Each node receives (context, inputs) where:
            context: shared mutable dict for orchestration state
            inputs:  dict[node_id -> output] from its immediate predecessors
      - Node output is stored in results[node_id].
    """

    def __init__(
        self,
        nodes: Dict[str, Node],
        edges: List[Edge],
        start_nodes: Optional[List[str]] = None,
    ):
        self.nodes = nodes
        self.edges = edges
        self.adj: Dict[str, List[str]] = {}
        self.rev: Dict[str, List[str]] = {}
        for n in nodes:
            self.adj[n] = []
            self.rev[n] = []
        for e in edges:
            self.adj[e.src].append(e.dst)
            self.rev[e.dst].append(e.src)
        self.start_nodes = start_nodes or self._infer_starts()

        self._validate_acyclic()

    # ---------- Topology utilities ----------
    def _infer_starts(self) -> List[str]:
        """Nodes with zero in-degree are starts."""
        indeg = {n: 0 for n in self.nodes}
        for e in self.edges:
            indeg[e.dst] += 1
        return [n for n, d in indeg.items() if d == 0]

    def get_adjacent_nodes(self, node_name: str) -> list[Node]:
        """
        Return all nodes directly connected *from* the given node name.
        Example: if node1 → node2 and node1 → node3,
                 get_adjacent_nodes("node1") -> [Node(node2), Node(node3)]
        """
        # Find the node ID by name
        node_id = None
        for nid, node in self.nodes.items():
            if node.name == node_name:
                node_id = nid
                break

        if node_id is None:
            raise KeyError(f"No node found with name '{node_name}'.")

        # Get all directly connected destination node IDs
        connected_ids = self.adj.get(node_id, [])

        # Return the Node objects
        return [self.nodes[nid] for nid in connected_ids]

    def get_start_nodes(self) -> List[Node]:
        """Return the list of start nodes as Node objects."""
        return [self.nodes[nid] for nid in self.start_nodes]

    def _topo_order(self) -> List[str]:
        indeg = {n: 0 for n in self.nodes}
        for e in self.edges:
            indeg[e.dst] += 1
        q = [n for n, d in indeg.items() if d == 0]
        order = []
        i = 0
        while i < len(q):
            u = q[i]
            i += 1
            order.append(u)
            for v in self.adj[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        if len(order) != len(self.nodes):
            raise ValueError(
                "Graph has at least one cycle; cannot produce topological order."
            )
        return order

    def _validate_acyclic(self) -> None:
        _ = self._topo_order()  # will raise on cycle

    # ---------- Export ----------
    def to_mermaid(self, direction: str = "TD") -> str:
        """
        Produce Mermaid flowchart text. Example:
            graph TD
              N1["StartAgent\n(agent)"] --> N2["MyFunc\n(callable)"]
        """
        lines = [f"graph {direction}"]
        for nid, n in self.nodes.items():
            label = n.name.replace('"', '\\"')
            sub = n.kind
            lines.append(f'  {nid}["{label}\\n({sub})"]')
        for e in self.edges:
            if e.label:
                lines.append(f'  {e.src} -- "{e.label}" --> {e.dst}')
            else:
                lines.append(f"  {e.src} --> {e.dst}")
        return "\n".join(lines)

    # ---------- Execution ----------
    def execute(
        self, initial_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        order = self._topo_order()
        context: Dict[str, Any] = initial_context or {}
        results: Dict[str, Any] = {}

        for nid in order:
            node = self.nodes[nid]
            inputs = {pid: results[pid] for pid in self.rev[nid] if pid in results}
            output = node.runner.run(context=context, inputs=inputs)
            results[nid] = output

        return results


# -------------------------------------------------------------------
# Builder
# -------------------------------------------------------------------


class GraphBuilder:

    def __init__(self):
        self._nodes: Dict[str, Node] = {}
        self._edges: List[Edge] = []
        self._explicit_starts: List[str] = []

    # ---- Adders ----
    def add_node(
        self,
        obj: NodeLike,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        id: Optional[str] = None,
    ) -> str:
        """Add an Agent instance, a plain callable, or a Runnable. Returns node id."""
        nid = id or str(uuid.uuid4())

        if isinstance(obj, Runnable):
            runner = obj
            kind = "agent" if isinstance(obj, AgentAdapter) else "callable"
        elif isinstance(obj, Agent):
            runner = AgentAdapter(obj)
            kind = "agent"
        elif callable(obj):
            runner = CallableAdapter(obj, name=name, description=description)
            kind = "callable"
        else:
            raise TypeError(
                "Unsupported node type. Provide Agent, callable, or Runnable."
            )

        nname = name or runner.name()
        ndesc = description or runner.description() or None
        meta = runner.metadata() or {}

        if nid in self._nodes:
            raise ValueError(f"Node id '{nid}' already exists.")
        self._nodes[nid] = Node(
            id=nid,
            runner=runner,
            kind=kind,
            name=nname,
            description=ndesc,
            metadata=meta,
        )
        return nid

    def mark_start(self, node_id: str) -> "GraphBuilder":
        if node_id not in self._nodes:
            raise KeyError(f"Unknown node id: {node_id}")
        if node_id not in self._explicit_starts:
            self._explicit_starts.append(node_id)
        return self

    def connect(
        self, src_id: str, dst_id: str, *, label: Optional[str] = None
    ) -> "GraphBuilder":
        if src_id not in self._nodes or dst_id not in self._nodes:
            raise KeyError("Both src and dst must be added before connecting.")
        self._edges.append(Edge(src=src_id, dst=dst_id, label=label))
        return self

    # ---- Compile targets ----
    def compile(self) -> Graph:
        if not self._nodes:
            raise ValueError("Cannot compile an empty graph.")
        g = Graph(
            nodes=dict(self._nodes),
            edges=list(self._edges),
            start_nodes=list(self._explicit_starts) or None,
        )
        return g

    def compile_mermaid(self, direction: str = "TD") -> str:
        return self.compile().to_mermaid(direction=direction)


# -------------------------------------------------------------------
# Example usage (you can delete this section in production)
# -------------------------------------------------------------------

if __name__ == "__main__":
    # Example callable step
    def summarize(
        context: Dict[str, Any], inputs: Dict[str, Any], payload: Optional[str] = None
    ) -> str:
        """Summarize text with an LLM (placeholder)."""
        text = payload or context.get("text") or ""
        result = f"[SUMMARY] {text[:50]}..."
        context["summary"] = result
        return result

    # Example callable that depends on previous outputs
    def route(context: Dict[str, Any], inputs: Dict[str, Any]) -> str:
        """Route based on summary length."""
        summary = inputs.get("N2") or inputs.get("summarize") or ""
        return "long" if len(summary) > 30 else "short"

    # Agents (from your base class)

    gb = GraphBuilder()
    n1 = gb.add_node(START, name="Start")
    n2 = gb.add_node(summarize, name="Summarize")
    n3 = gb.add_node(route, name="Router")
    n4 = gb.add_node(END, name="Finish")

    (gb.mark_start(n1).connect(n1, n2).connect(n2, n3).connect(n3, n4, label="done"))

    graph = gb.compile()

    # Execute
    print("MERMAID:")
    print(graph.to_mermaid())

    print(graph.get_adjacent_nodes("Start"))
