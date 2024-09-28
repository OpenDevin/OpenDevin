from agenthub.coact_agent.executor.executor_agent import (
    LocalExecutorAgent as CoActExecutorAgent,
)
from agenthub.coact_agent.planner.planner_agent import (
    GlobalPlannerAgent as CoActPlannerAgent,
)
from openhands.controller.agent import Agent

Agent.register('CoActPlannerAgent', CoActPlannerAgent)
Agent.register('CoActExecutorAgent', CoActExecutorAgent)
