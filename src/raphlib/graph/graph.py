
from typing import (
    Any,
    Dict,
    List,
    TypeVar,
    Generic,
    Optional,
    Callable,
    Literal,
    Type,
)
from langgraph.graph.state import CompiledStateGraph
from langgraph.config import get_stream_writer

import asyncio, inspect, pydantic

from .state import BaseState
from .edges import EdgeDescriptor

from langgraph.graph import StateGraph
from langgraph.types import Command

python_next = next  # An alias to the next function

State = TypeVar("State", bound=BaseState)



class GraphBuilder(Generic[State]):
    """
    An interface for creating LangGraph systems easily.
    """

    def __init__(self, start: str, state: Type[State]):
        """
        Parameters:
            start (str): The name of the start node.
            state (BaseState): The state pydantic model for initializing the graph.
        """
        self.__graph = StateGraph(state)
        self.__edges: EdgeDescriptor = EdgeDescriptor(start=start)


    def compiled(self) -> CompiledStateGraph:
        self.__edges.attach_edges(self.__graph)
        compiled_graph = self.__graph.compile()
        return compiled_graph
    
    
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
