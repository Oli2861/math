from abc import ABC
from typing import Any

from pydantic import BaseModel


class Variable(BaseModel, ABC):
    """
    The discrete / continuous variable.
    """
    value: Any

    def __init__(self, value: Any):
        super().__init__(value=value)
