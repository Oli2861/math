from abc import ABC, abstractmethod
from typing import List


class Layer(ABC):

    @abstractmethod
    def forward(self, x: List[List[float]]) -> List[List[float]]:
        pass

    @abstractmethod
    def backprop(
            self,
            previous_activation: List[float],
            learning_rate: float,
            label: int | None = None,
            activation_cost_effect: List[float] | None = None,
            print_info: bool = False
    ) -> List[float]:
        pass
