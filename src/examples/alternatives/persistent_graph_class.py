
from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Set,
    TypedDict,
    TypeVar,
    Generic,
    Self,
    Optional,
    Callable,
    Literal,
    AsyncGenerator,
    Generator,
    Type,
    Annotated,
    get_args,
    get_type_hints,
    get_origin,
)
from pydantic import BaseModel
from langgraph.graph.state import CompiledStateGraph
from langchain.schema.runnable import RunnableConfig 
from langchain.chat_models.base import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import StateSnapshot, StreamMode
from langgraph.config import get_stream_writer
from abc import abstractmethod

import asyncio
import inspect

from ... import run_in_parallel_event_loop, get_or_create_event_loop, BaseInterrupt

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, Interrupt, PregelTask

python_next = next  # An alias to the next function


# ============================================================================ STATE


class BaseState(BaseModel):

    def __init_subclass__(cls, **kwargs):
        """
        Add annotations to set a **default update behavior** 
        for langchain's for non annotated fields. 
        Default is to replace the current value by the new.
        """
        super().__init_subclass__(**kwargs)
        annotations = get_type_hints(cls, include_extras=True)
        for field, field_type in annotations.items():
            if get_origin(field_type) is not Annotated:
                # Default behavior : Replace the current value by the new.
                annotations[field] = Annotated[field_type, lambda _, new: new]
        cls.__annotations__ = annotations


State = TypeVar("State", bound=BaseState)


# ============================================================================= EDGES


class EdgeDescriptor(Dict[str, Set[str]]):
    """
    Describe the edges between nodes in the graph before compilation.
    """
    def __init__(self, start: str):
        """
        Parameters:
            start (str): The name of the nodes that starts the graph.
            end (str): The name of the nodes that ends the graph.
        """
        super().__init__()
        self.add(START, start)

    def add(self, start: str, end: str):
        """
        Add an edge between two nodes in the graph.

        Parameters:
            start (str): The name of the node the edge stems from.
            end (str): The name of the node the edge goes to.
        """
        if start not in self.keys():
            self[start] = set()
        self[start].add(end)

    def attach_edges(self, graph: StateGraph):
        """Attach the edges to the graph. The nodes must exist prior to this."""
        for start, ends in self.items():
            for end in ends:
                if end == "__end__":
                    graph.add_edge(start, END)
                else:
                    graph.add_edge(start, end)




# =========================================================================== GRAPH




class BaseGraph(Generic[State]):
    """
    An interface for creating LangGraph systems easily.
    """
 
    def __init__(
            self, 
            start: str, 
            state: Type[State], 
            checkpointer: Optional[BaseCheckpointSaver] = None,
            config: RunnableConfig = {"configurable": {"thread_id": "1"}},
        ):
        """
        Parameters:
            start (str): The name of the start node.
            state (BaseState): The state pydantic model for initializing the graph.
            checkpointer (BaseCheckpointSaver, optional): A checkpointer object to save the states of the graph.
            config: (RunnableConfig, optional): The config, namely the thread id to run the graph with.
        """
        self.__graph = StateGraph(state)
        self.__state_type = state
        self.__config: RunnableConfig = config
        self.__compiled_graph: Optional[CompiledStateGraph] = None
        self.__edges: EdgeDescriptor = EdgeDescriptor(start=start)

        self.__state: State = None
        self.__update_state_from_graph: bool = False  # Whether to update the state with the value from the graph
        self.__graph_state: StateSnapshot = None

        if checkpointer:
            self.__checkpointer = checkpointer
        else:
            self.__checkpointer: BaseCheckpointSaver = MemorySaver()

    @property
    def state(self) -> State:
        """
        Retrieve the state stored in the local thread. 
        Create a new state if the grpah has not run yet.
        """
        if self.__update_state_from_graph:
            snapshot: StateSnapshot = self.graph.get_state(self.__config)
            self.__state = self.__state_type.model_validate(snapshot.values)
            self.__update_state_from_graph = False

        elif self.__state is None:
            self.reset_state()
        
        return self.__state
    
    @property
    def graph_state(self) -> StateSnapshot:
        """
        Retrieve the state of the graph itself, 
        including the call metadata.
        """
        if self.__graph_state is None:
            self.__graph_state = self.graph.get_state(self.__config)
        return self.__graph_state
    
    @property
    def tasks(self) -> Tuple[PregelTask]:
        return self.graph_state.tasks
    
    @property
    def main_task(self) -> Optional[PregelTask]:
        tasks = self.tasks
        if len(tasks) > 0:
            return tasks[0]
    
    @property
    def interrupt(self) -> Optional[Any]:
        """
        Retrieve the value of the interrupt 
        associated with the main paused task of the graph, if there is any.
        """
        if self.main_task and self.main_task.interrupts:
            return self.main_task.interrupts[0].value
        
    @property
    def current_node(self) -> Optional[str]:
        if self.main_task:
            return self.main_task.name

    def reset_state(self, initialize: Dict[str, Any] = None):
        """
        Resets the state to the default value of its pydantic representation.
        
        Parameters:
            initilize (dict): Attributes to value mapping of non default values to set to the resetted state.
        """
        self.__state = self.__state_type()
        if initialize:
            for key, val in initialize:
                setattr(self.__state, key, val)
                
    @property
    def config(self) -> RunnableConfig:
        return self.__config

    @property
    def graph(self) -> CompiledStateGraph:
        if self.__compiled_graph is None:
            self.__edges.attach_edges(self.__graph)
            self.__compiled_graph = self.__graph.compile(
                checkpointer = self.__checkpointer
            )
        return self.__compiled_graph
    
    def reset_diagnostic_variables(self):
        """
        Reset variables that should be every time the graph is run or resumed.
        """
        self.__update_state_from_graph = True
        self.__graph_state = None
    



    # ======================================================================================= STREAMING



    
    async def astream(
            self, 
            resume: Any = None, 
            stream_mode: StreamMode = "custom"
        ) -> AsyncGenerator[Any, None]:
        """
        Stream the graph until the next interrupt.

        Parameters:
            resume (Any):
                The value to provide to the graph interrupt when resuming the call.
            stream_mode (StreamMode):
                A LangGraph stream mode to use instead of the default "custom" (yield values).
        """
        if self.interrupt:
            command = Command(resume=resume, update=self.state.model_dump()) if resume else Command(resume="", update=self.state.model_dump())
        else:
            if resume: raise ValueError(f"The resume value {resume} was provided while the graph has no current interruption.")
            command = Command(update=self.state.model_dump())
                              
        self.reset_diagnostic_variables()

        async for event in self.graph.astream(input=command, config=self.config, stream_mode=stream_mode):
            yield event

    def stream(
            self, 
            resume: Any = None, 
            stream_mode: StreamMode = "custom"
        ) -> Generator[Any, None, None]:
        """
        Stream the graph until the next interrupt.

        Parameters:
            resume (Any):
                The value to provide to the graph interrupt when resuming the call.
            stream_mode (StreamMode):
                A LangGraph stream mode to use instead of the default "custom" (yield values).
        """
        async_gen_instance = self.astream(resume=resume, stream_mode=stream_mode)
        try:
            loop = get_or_create_event_loop()
            if loop.is_running():
                while True:
                    yield run_in_parallel_event_loop(future=async_gen_instance.__anext__())
            else:
                while True:
                    yield loop.run_until_complete(future=async_gen_instance.__anext__())

        except BaseInterrupt as interrupt:
            raise interrupt
        
        except StopAsyncIteration:
            pass

    def invoke(self, resume: Any = None):
        """Execute the graph until the next interrupt."""
        for _ in self.stream(resume=resume): pass

    async def ainvoke(self, resume: Any = None):
        """Execute the graph until the next interrupt."""
        async for _ in self.astream(resume=resume): pass



    
    # =============================================================================== NODE DECORATOR




    def node(self, name: Optional[str] = None, update: Optional[str | List[str]] = None, next: str | List[str] = []) -> Callable[[State], Dict[str, Any]]:
        """
        Add a node to the graph based on the decorated function, which now produces a node update output. 
        Edges will be created based on "next".

        Parameters:
            name (Optional[str]): The name of the node. If not specified, use the name of the function.
            update (Optional[List[str]]): The keys of the State which should be updated. If unspecified, update the whole state.
            next (List[str]): The nodes this one connects to.
        """
        if isinstance(next, str): next = [next]
        if update is not None and isinstance(update, str): update = [update]

        if isinstance(name, Callable):  # Decorator mode
            func = name
            name = None
        else:
            func = None


        # ------------------------------------------- Decorator / Inspection


        def decorator(f: Callable):

            if name is None:
                local_name = f.__name__
            else:
                local_name = name

            # Inspect the signature to make sure the State is passed as an argument (or nothing)
            sig = inspect.signature(f)
            if len(sig.parameters) > 1:
                raise ValueError(
                    f"StateGraph nodes can only have one parameter. "
                    f"Got {len(sig.parameters)} params at node \"{local_name}\". Expected 1."
                )
            
            param = python_next(iter(sig.parameters.values()), None)

            if len(sig.parameters) == 0:
                raise ValueError(
                    f"Node \"{local_name}\" must have a \"{State.__name__}\" parameter, "
                    f",even if you don't want to modify the state in that node."
                )
            
            is_async = asyncio.iscoroutinefunction(f)
            is_gen = inspect.isgeneratorfunction(f)
            is_asyncgen = inspect.isasyncgenfunction(f)
            
            def extract_command(output: None | str | Command | dict) -> Optional[Command]:
                if isinstance(output, Command):
                    return output
                elif isinstance(output, str):
                    return Command(goto = output)
                else:
                    return None
                
            # ---------------------------------------------------- Node Function

            async def node_function(s: State):

                stream_writer = get_stream_writer()
                command = None

                if is_asyncgen:
                    async for event in f(s):
                        if isinstance(event, Command):
                            command = extract_command(event)
                            break
                        else:
                            stream_writer(event)

                elif is_gen:
                    for event in f(s):
                        if isinstance(event, Command):
                            command = extract_command(event)
                            break
                        else:
                            stream_writer(event)

                elif is_async:
                    result = await f(s)
                    command = extract_command(result)

                else:
                    command = extract_command(f(s))

                value_update = {key: getattr(s, key) for key in update} if update else s.model_dump()

                if command:
                    return command.__replace__(update=value_update)
                else:
                    return Command(update=value_update)
            

            # ------------------------------------------- Graph Update


            # Add the edge to the graph
            if len(next) == 1:
                self.__edges.add(local_name, next[0])

            # Add appropriate type hints if the node has many edges stemming from it.
            elif len(next) > 1:
                # Graph redirections are handled via Command outputs inside the node, 
                # but the graph schema is built using type hints which have to be added here.
                node_function.__annotations__["return"] = Command[Literal.__getitem__(tuple(next))]

            # Add the node to the graph
            self.__graph.add_node(node=local_name, action=node_function)
            
            return node_function
        

        # ---------------------------------------------------------------- Factory Logic

        
        if func:  # Decorator mode
            return decorator(func)
        else:  # Factory mode
            return decorator
