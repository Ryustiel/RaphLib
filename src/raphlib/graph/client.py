"""
Define streaming functions and state persistence for the graph.
"""

from typing import AsyncIterator, Iterator
from abc import ABC, abstractmethod

import httpx, json, pydantic

from .. import get_or_create_event_loop, run_in_parallel_event_loop

from langgraph.graph.state import CompiledStateGraph
from .graph import State


DEFAULT_RECURSION_LIMIT = 20


class GraphClient(ABC):
    """
    Client for the graph. It is used to define streaming functions and state persistence.
    """
    
    @abstractmethod
    def get_state(self) -> State:
        """
        Get the current state of the graph.
        """
        raise NotImplementedError("get_state() not implemented")
    
    @abstractmethod
    def process_new_state(self, new_state: State|dict) -> None:
        """
        Process the new state and update the current state.
        """
        raise NotImplementedError("process_new_state() not implemented")
        
    @abstractmethod
    async def astream_from(self, state: State|dict, assistant_id: str = None, config: dict = {}, recursion_limit: int = DEFAULT_RECURSION_LIMIT) -> AsyncIterator[str]:
        """
        Send the state to the graph and stream it.
        """
        raise NotImplementedError("astream_from() not implemented")
    
    async def astream(self, assistant_id: str = None, config: dict = {}, recursion_limit: int = DEFAULT_RECURSION_LIMIT) -> AsyncIterator[str]:
        """
        Stream from the current state.
        """
        async for chunk in self.astream_from(
            state=self.get_state(), 
            assistant_id=assistant_id, 
            config=config, 
            recursion_limit=recursion_limit
        ):
            yield chunk
        
    def stream_from(self, state: State|dict, assistant_id: str = None, config: dict = {}, recursion_limit: int = DEFAULT_RECURSION_LIMIT) -> Iterator[str]:
        """
        Send the state to the graph and stream it.
        """
        async_iter_instance = self.astream_from(
            state=state, 
            assistant_id=assistant_id, 
            config=config, 
            recursion_limit=recursion_limit
        )
        
        try:
            loop = get_or_create_event_loop()
            if loop.is_running():
                while True:
                    yield run_in_parallel_event_loop(future=async_iter_instance.__anext__())
            else:
                while True:
                    yield loop.run_until_complete(future=async_iter_instance.__anext__()) 
        
        except StopAsyncIteration:
            pass
        
    def stream(self, assistant_id: str = None, config: dict = {}, recursion_limit: int = 20) -> Iterator[str]:
        """
        Stream from the current state.
        """
        for chunk in self.stream_from(
            state=self.get_state(), 
            assistant_id=assistant_id, 
            config=config, 
            recursion_limit=recursion_limit
        ):
            yield chunk
        
    

class RemoteGraphClient(GraphClient):
    """
    Connects to a langgraph server and runs the graph.
    Store the state in RAM.
    """
    
    def __init__(self, initial_state: State|dict, url: str = "http://127.0.0.1:2024"):
        self.state = initial_state
        self.url = url
        
    def get_state(self) -> State:
        return self.state
    
    def process_new_state(self, new_state: str):
        self.state = new_state
        
    async def astream_from(self, state: State|dict, assistant_id: str, config: dict = {}, recursion_limit: int = DEFAULT_RECURSION_LIMIT) -> AsyncIterator[str]:
        
        if isinstance(state, pydantic.BaseModel):
            state = state.model_dump()
        
        async with httpx.AsyncClient() as client:

            response = await client.post(
                self.url + "/runs/stream", 
                json = {
                    "assistant_id": assistant_id,
                    "input": state,
                    "config": {
                        "recursion_limit": recursion_limit,
                        "configurable": config,
                    },
                    "stream_mode": ["custom", "values"],
                }, 
                headers = {"Content-Type": "application/json"}
            )

            output_state: str = ""
            async for line in response.aiter_lines():
                
                if line and line.startswith("data:"):
                        
                    if line[6] == "\"":  # Is a data and a custom event, not metadata
                    
                        chunk = line[7:-1]
                        yield chunk
                    
                    elif line[6] == "{":
                        output_state = line[6:]
                            
            if output_state is not None:
                self.process_new_state(json.loads(output_state))
            else:
                raise RuntimeError("RemoteGraphClient: no output state received.")
                            
            
            
class PersistentGraphClient(GraphClient):
    """
    Connects to a langgraph server and runs the graph.
    Store the state in RAM and persist it in a file.
    File is used to restore the state in case of a crash.
    """

    def __init__(self, storage_path: str, url: str = "http://127.0.0.1:2024"):
        self.storage_path = storage_path
        self.url = url
        
    def get_state(self) -> State:
        with open(self.storage_path, "r") as f:
            return json.load(f)
        
    def process_new_state(self, new_state: State|dict) -> None:
        with open(self.storage_path, "w") as f:
            json.dump(new_state, f, indent=4)



class PersistentRemoteGraphClient(PersistentGraphClient, RemoteGraphClient):
    """
    Connects to a langgraph server and runs the graph.
    Store the state in RAM and persist it in a file.
    File is used to restore the state in case of a crash.
    """
    pass
