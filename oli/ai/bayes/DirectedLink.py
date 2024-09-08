from pydantic import BaseModel

from oli.ai.bayes.Node import Node


class DirectedLink(BaseModel):
    source: Node
    destination: Node
    associated_probability: ConditionalProbability

    def __init__(self, source: Node, destination: Node, associated_probability: ConditionalProbability):
        super().__init__(source=source, destination=destination, associated_probability=associated_probability)
