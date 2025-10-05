from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, List
import openai
from agentic.core.tools import function_to_openai_tool


class Agent(ABC):

    def __init__(
        self,
        *args,
        name: str = None,
        description: str = None,
        base_url: str = None,
        openai_client: openai.OpenAI = None,
        api_key: str = None,
        model: str = "qwen3-30b-a3b",
        stream_handler: Optional[Callable] = None,
        children: List[Agent] | None = None,
        **kwargs,
    ):
        self.args = args
        self.kwargs = kwargs
        # Initialize other necessary attributes here
        self.base_url = base_url
        self.api_key = api_key
        self.client = openai_client or openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.stream_handler = stream_handler
        self.children = children or []
        self.name = name
        self.description = description
        self._stateless = True
        self._metadata = None

    def add_agent(self, agent: Agent) -> Agent:
        self.children.append(agent)
        return self

    def get_children(self) -> List[Agent]:
        return self.children

    def get_agent_name(self) -> str:
        return self.name or self.__class__.__name__

    def get_agent_description(self) -> str:
        return self.description or self.__class__.__doc__ or None

    def get_metadata(self) -> dict[str, Any]:
        if self._metadata is None:
            self._metadata, self._stateless = function_to_openai_tool(
                self.run,
                name=self.get_agent_name(),
                description=self.get_agent_description(),
            )
        return self._metadata

    @property
    def stateless(self) -> bool:
        return self._stateless

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """Subclasses must implement this method."""
        pass


class StartAgent(Agent):

    def run(self, *args, context: Dict[str, Any] = None, **kwargs) -> str:
        return "Start Agent"


class EndAgent(Agent):
    def run(self, *args, context: Dict[str, Any] = None, **kwargs) -> str:
        return "End Agent"


START = StartAgent(name="START", description="This is the no-op start agent")
END = EndAgent(name="END", description="This is the no-op end agent")
