# This file includes the workflow node definitions for agentic workflow.
# Through the agentic workflow these nodes will be used.

import io
from typing import Callable

import matplotlib.pyplot as plt
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END
from langgraph.graph import StateGraph

from .agent_state import AgentState
from .data_classes import *
from .router import router


def task_generation_node(llm_agent) -> Callable:
    """
    This node generates tasks and updates the state.
    """

    def node_fn(state: AgentState):
        description = state["project_description"]
        prompt = f"""
        You are a senior project management expert. Break down this project into actionable tasks:

        PROJECT: {description}

        Provide your analysis in these sections:

        1. CORE COMPONENTS
        - Key deliverables
        - Technical requirements
        - Dependencies and constraints
        - Critical success factors

        2. TASK BREAKDOWN
        For each major component, provide:
        - Task name (use action verbs)
        - Clear description (1-2 sentences)
        - Duration (in days)
        - Required skills
        - Dependencies
        - Deliverables
        - Acceptance criteria
        - Risk level (Low/Medium/High)

        Break down any task longer than 5 days into smaller subtasks.

        3. SEQUENCING & RELATIONSHIPS
        - Critical path tasks (mark with *)
        - Parallel execution opportunities
        - Key dependencies
        - Resource conflicts

        4. RISK & QUALITY
        - High-risk areas
        - Quality control points
        - Mitigation strategies
        - External dependencies

        Format all tasks using:
        - Action verbs
        - Specific outcomes
        - Clear timeframes
        - Measurable criteria

        Highlight tasks that:
        - Are on critical path (*)
        - Have external dependencies (#)
        - Require special skills (+)
        - Carry high risk (!)
        """

        structure_llm = llm_agent.with_structured_output(TaskList)
        tasks: TaskList = structure_llm.invoke(prompt)
        return {"tasks": tasks}

    return node_fn


def task_dependency_node(llm_agent) -> Callable:
    """Creates a callable node for dependency analysis"""

    def node_fn(state: AgentState):
        tasks = state["tasks"]
        print("Current State task_dependency_node", state, "\n\n")
        prompt = f"""
        You are an expert project scheduler specialized in creating dependency maps and critical path analysis. 
        Your task is to analyze the following project tasks and create a comprehensive dependency structure:

        Tasks to analyze: {tasks}

        Please follow these steps in order:

        1. First Pass - Individual Task Analysis:
           - Analyze each task's requirements and prerequisites
           - Identify any implicit technical or logical dependencies
           - Consider resource constraints and sequential requirements
           - List any assumptions you're making about task relationships

        2. Dependency Mapping (Upstream):
           - For each task, explicitly list all tasks that MUST be completed before it can begin
           - Classify dependencies as:
             * Mandatory (hard dependency)
             * Preferred (soft dependency)
             * Resource-based dependency

        3. Downstream Impact Analysis:
           - For each task, identify all tasks that:
             * Cannot start until this task completes
             * Are partially blocked by this task
             * May be impacted by this task's completion

        4. Validation Checks:
           - Verify there are no circular dependencies
           - Confirm all dependencies are logically sound
           - Flag any potential bottlenecks or risk areas
           - Highlight any tasks that could potentially be parallelized

        Output Format:
        For each task, provide:
        1. Task Name
        2. Upstream Dependencies: [List of tasks that must finish before this task]
        3. Downstream Dependents: [List of tasks waiting on this task]
        4. Critical Path Status: [Whether this task is on the critical path]
        5. Risk Level: [High/Medium/Low based on dependency complexity]

        Important Considerations:
        - If you're uncertain about a dependency, explain your reasoning
        - Highlight any assumptions that could affect the dependency structure
        - Flag any dependencies that seem unusual or require validation
        - Suggest any opportunities for optimizing the task sequence

        Please be explicit and thorough in your analysis, as this will be used for project planning and resource 
        allocation.
        """

        structure_llm = llm_agent.with_structured_output(DependencyList)
        dependencies: DependencyList = structure_llm.invoke(prompt)
        return {"dependencies": dependencies}

    return node_fn


def task_scheduler_node(llm_agent) -> Callable:
    """Creates a callable node for task scheduling"""

    def node_fn(state: AgentState):
        dependencies = state["dependencies"]
        tasks = state["tasks"]
        insights = state["insights"] if "insights" in state else ""

        prompt = f"""
        You are a scheduling optimization expert specializing in critical path management and parallel execution.

        INPUTS:
        Tasks: {tasks}
        Dependencies: {dependencies}
        Historical Data: {insights}
        Current Iteration: {state["schedule_iteration"]}

        OPTIMIZATION OBJECTIVES:
        - Minimize total duration
        - Maximize parallel execution
        - Balance resource utilization
        - Maintain appropriate risk buffers

        SCHEDULING RULES:
        1. Critical path tasks get priority
        2. Dependencies must be enforced
        3. Resource conflicts must be resolved
        4. Add buffers based on task risk level
        5. Favor parallel execution where possible

        OUTPUT FORMAT:
        For each task provide:
        1. Start Day (absolute)
        2. End Day (absolute)
        3. Duration (days)
        4. Float Available
        5. Parallel Tasks
        6. Critical Path [Yes/No]

        KEY METRICS:
        1. Total Duration: ___ days
        2. Parallel Task Groups: ___
        3. Critical Path Length: ___ days
        4. Total Float: ___ days
        5. Resource Utilization: ___%

        HANDLE THESE SCENARIOS:
        * Dependency Conflicts → Flag and propose solutions
        * Resource Overallocation → Suggest reallocation options  
        * Critical Path Bottlenecks → Recommend mitigation
        * Scheduling Conflicts → Provide alternatives

        Compare with previous iteration and explain key improvements made.

        Mark high-risk schedule segments with (!) and suggest contingency plans.
        """

        schedule_llm = llm_agent.with_structured_output(Schedule)
        schedule: Schedule = schedule_llm.invoke(prompt)
        new_state = state.copy()
        new_state["schedule"] = schedule
        new_state["schedule_iteration"].append(schedule)
        return new_state

    return node_fn


def task_allocation_node(llm_agent) -> Callable:
    """Creates a callable node for task allocation"""

    def node_fn(state: AgentState):
        tasks = state["tasks"]
        schedule = state["schedule"]
        team = state["team"]
        insights = state["insights"] if "insights" in state else ""

        prompt = f"""
        You are a resource management expert specializing in team optimization and workload balancing.

        INPUTS:
        Tasks: {tasks}
        Schedule: {schedule}
        Team: {team}
        Historical Data: {insights}
        Current Iteration: {state["task_allocations_iteration"]}

        OPTIMIZATION GOALS:
        - Balance workload across team
        - Match skills to requirements
        - Enable knowledge transfer
        - Maintain 80% max utilization
        - Support work-life balance

        TEAM ANALYSIS:
        For each member, consider:
        - Core skills and expertise
        - Current workload
        - Time zone
        - Development goals
        - Past performance

        ALLOCATION RULES:
        1. Critical tasks need experienced leads
        2. Complex tasks use buddy system
        3. Learning tasks pair junior/senior
        4. All critical tasks need backups
        5. Consider time zone overlaps

        OUTPUT FORMAT:
        For each task provide:
        1. Primary Assignee
        2. Backup Assignee
        3. Effort (hours/days)
        4. Start/End Dates
        5. Skill Match (1-5)
        6. Risk Level [Low/Medium/High]

        KEY METRICS:
        1. Team Balance Score: ___
        2. Skill Match Score: ___
        3. Knowledge Share Index: ___
        4. Risk Distribution: ___
        5. Time Zone Coverage: ___%

        HANDLE THESE SCENARIOS:
        * Skill Gaps → Recommend training/hiring
        * Overallocation → Suggest rebalancing
        * Timeline Conflicts → Propose alternatives
        * Single Points of Failure → Add backup coverage

        Mark tasks with:
        (!) High-risk assignments
        (*) Critical path tasks
        (+) Learning opportunities
        (#) Time zone challenges

        Compare with previous iteration and explain improvements in:
        - Workload balance
        - Skill matching
        - Knowledge sharing
        - Risk distribution
        """

        structure_llm = llm_agent.with_structured_output(TaskAllocationList)
        allocations: TaskAllocationList = structure_llm.invoke(prompt)
        new_state = state.copy()
        new_state["task_allocations"] = allocations
        new_state["task_allocations_iteration"].append(allocations)
        return new_state

    return node_fn


def risk_assessment_node(llm_agent) -> Callable:
    """Creates a callable node for risk assessment"""

    def node_fn(state: AgentState):
        schedule = state["schedule"]
        task_allocations = state["task_allocations"]

        prompt = f"""
        You are a risk analyst. Assess project risks and provide task and overall risk scores.

        PROJECT DATA:
        Task Allocations: {task_allocations}
        Schedule: {schedule}
        Previous Assessment: {state['risks_iteration']}

        RISK SCORING RULES:
        - Scale: 0 (no risk) to 10 (high risk)
        - Keep previous scores for unchanged assignments
        - Reduce score if:
          * More buffer time between tasks
          * More senior resource assigned
          * Previous successful completion
        - Increase score if:
          * Tight dependencies
          * New resource assignment
          * Complex task requirements

        ASSESSMENT CRITERIA:
        1. Technical Complexity
        2. Resource Experience
        3. Schedule Constraints
        4. Dependency Chain
        5. Buffer Availability

        REQUIRED OUTPUT:
        1. Per Task:
           - Risk Score (0-10)
           - Previous Score (if any)
        2. Overall Score:
           - Total Risk = Sum of all task scores
        """

        structure_llm = llm_agent.with_structured_output(RiskList)
        risks: RiskList = structure_llm.invoke(prompt)
        project_risk_score = sum(int(risk.score) for risk in risks.risks)

        new_state = state.copy()
        new_state["risks"] = risks
        new_state["project_risk_score"] = project_risk_score
        new_state["iteration_number"] += 1
        new_state["project_risk_score_iterations"].append(project_risk_score)
        new_state["risks_iteration"].append(risks)
        return new_state

    return node_fn


def insight_generation_node(llm_agent) -> Callable:
    """Creates a callable node for insight generation"""

    def node_fn(state: AgentState):
        schedule = state["schedule"]
        task_allocations = state["task_allocations"]
        risks = state["risks"]
        prompt = f"""
        You are a project optimization expert. Analyze plan and provide actionable improvements.

        INPUTS:
        Allocations: {task_allocations}
        Schedule: {schedule}
        Risks: {risks}

        ANALYZE FOR:
        1. Critical Issues
           • Resource conflicts
           • Timeline bottlenecks
           • High-risk tasks
           • Workflow inefficiencies

        2. Improvement Areas
           • Resource optimization
           • Schedule adjustments
           • Risk mitigation
           • Process enhancement

        RECOMMENDATIONS FORMAT:
        [Priority] [HIGH/MEDIUM/LOW]
        Issue: Current problem
        Fix: Specific solution
        Benefit: Risk reduction/improvement
        Steps: Implementation guide

        FLAG AS:
        ! = Must address (critical)
        ⚡ = Quick fix available
        🔄 = Process improvement
        ⚠️ = Risk reduction

        REQUIRED OUTPUT:

        1. Critical Improvements
           • Urgent fixes needed
           • Risk reduction steps
           • Resource conflicts
           • Timeline issues

        2. Optimization Steps
           • Resource balancing
           • Schedule adjustments
           • Process improvements
           • Efficiency gains

        Focus on specific, implementable changes that will reduce project risk score.
        """

        insights = llm_agent.invoke(prompt).content
        return {"insights": insights}

    return node_fn


def create_workflow(state: AgentState, llm_agent):
    # Create workflow
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("task_generation", task_generation_node(llm_agent))
    workflow.add_node("task_dependencies", task_dependency_node(llm_agent))
    workflow.add_node("task_scheduler", task_scheduler_node(llm_agent))
    workflow.add_node("task_allocator", task_allocation_node(llm_agent))
    workflow.add_node("risk_assessor", risk_assessment_node(llm_agent))
    workflow.add_node("insight_generator", insight_generation_node(llm_agent))

    # Add edges
    workflow.set_entry_point("task_generation")
    workflow.add_edge("task_generation", "task_dependencies")
    workflow.add_edge("task_dependencies", "task_scheduler")
    workflow.add_edge("task_scheduler", "task_allocator")
    workflow.add_edge("task_allocator", "risk_assessor")
    workflow.add_conditional_edges("risk_assessor", router, ["insight_generator", END])
    workflow.add_edge("insight_generator", "task_scheduler")

    return workflow


def compile_workflow(workflow, create_memory=True):
    memory = MemorySaver() if create_memory else None
    graph_plan = workflow.compile(checkpointer=memory)

    return graph_plan, memory


def display_plan(graph_plan):
    img = graph_plan.get_graph(xray=1).draw_mermaid_png()

    # Convert the binary image data to a numpy array that matplotlib can display
    img_array = plt.imread(io.BytesIO(img))

    # Display the image
    plt.imshow(img_array)
    plt.axis('off')  # Hide axes
    plt.show()
