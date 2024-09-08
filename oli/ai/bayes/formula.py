import math
from typing import List, Optional
import unittest

from oli.ai.bayes.ConditionalProbability import ConditionalProbability


def is_independent(joint_probability: float, event_probability: float) -> bool:
    """
    If the joint probability is equal to the event probability, the RVs are independent.
    :param joint_probability: P(x|y)
    :param event_probability: P(x)
    :return: Whether the RVs are independent.
    """
    return math.isclose(joint_probability, event_probability)


def law_of_total_probability(conditional_probabilities: List[ConditionalProbability]) -> float:
    """
    We can calculate the probability a RV takes a certain value x, if all conditional probabilities are know.
    (Discrete case)
    P(x) = ∑_y P(x|y) * P(y)
    :param conditional_probabilities: List of the
    :return: Probability of RV taking x.
    """
    summed: float = 0

    for curr in conditional_probabilities:
        evidence_probability: Optional[float] = curr.evidence_probability
        assert evidence_probability is not None
        conditional_probability = curr.probability_of_event_given_evidence

        summed += conditional_probability * evidence_probability

    return summed


class TestLawOfTotalProbability(unittest.TestCase):

    def test_is_independent(self):
        joint_probability = 0.9
        event_probability = 0.9
        self.assertTrue(joint_probability, event_probability)

    def test_law_of_total_probability(self):
        cond1 = ConditionalProbability(
            probability_of_event_given_evidence=0.99,
            evidence_probability=0.6,
            joint_probability=None
        )
        cond2 = ConditionalProbability(
            probability_of_event_given_evidence=0.95,
            evidence_probability=0.4,
            joint_probability=None
        )

        result = law_of_total_probability([cond1, cond2])
        expected_result = 594 / 1000 + 380 / 1000

        self.assertAlmostEqual(result, expected_result, places=4)


if __name__ == '__main__':
    unittest.main()
