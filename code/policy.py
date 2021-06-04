from dataclasses import dataclass

from fulfillment_plan import FulfillmentPlan

@dataclass
class PolicyResults:
    fulfill_plan: FulfillmentPlan
    rewards: list[float]
