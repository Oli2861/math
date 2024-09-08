from pydantic import BaseModel

from oli.ai.bayes.Variable import Variable


class Node(BaseModel):
    """
    Properties:
        variable: The discrete / continuous variable the node represents.
    """

    variable: Variable
    probability: float

    def __init__(self, variable: Variable):
        super().__init__(variable=variable)
