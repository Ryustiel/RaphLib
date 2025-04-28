
from typing import Annotated, get_type_hints, get_origin
from pydantic import BaseModel

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
        