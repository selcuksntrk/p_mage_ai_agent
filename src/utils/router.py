# This file includes the router component definition.
# This component will be used to create agentic workflow loops based on risk score.

from langgraph.graph import END

from .agent_state import *


def router(state: AgentState):
    """
    LangGraph node to create workflow loop based on risk score.
    """
    max_iteration = state["max_iteration"]
    iteration_number = state["iteration_number"]

    if iteration_number < max_iteration:
        if len(state["project_risk_score_iterations"]) > 1:
            if state["project_risk_score_iterations"][-1] < state["project_risk_score_iterations"][0]:
                return END
            else:
                return "insight_generator"
        else:
            return "insight_generator"
    else:
        return END
