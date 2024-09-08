from typing import Optional

from pydantic import BaseModel


class ConditionalProbability(BaseModel):
    """
    Probability of an event occurring given evidence.
    P(x|y) = P(x,y) / P(y)
    If x and y are independent: P(x,y) = P(x) * P(y)
    :param probability_of_event_given_evidence: Probability of an event occurring given evidence. (What this class if about)
    :param evidence_probability: Probability of the evidence occurring (if specified)
    :param joint_probability: Probability of the evidence and the event occurring together.
    """
    probability_of_event_given_evidence: float

    evidence_probability: Optional[float]
    joint_probability: Optional[float]

    def __init__(
            self,
            probability_of_event_given_evidence: float,
            joint_probability: Optional[float] = None,
            evidence_probability: Optional[float] = None
    ):
        super().__init__(
            probability_of_event_given_evidence=probability_of_event_given_evidence,
            joint_probability=joint_probability,
            evidence_probability=evidence_probability
        )

    @staticmethod
    def from_joint_and_evidence(joint_probability: float, evidence_probability: float) -> "ConditionalProbability":
        """
        P(x|y) = P(x,y) / P(y)
        :param joint_probability: P(x,y)
        :param evidence_probability: P(y)
        :return: Conditional probability P(x|y)
        """
        return ConditionalProbability(joint_probability / evidence_probability)
