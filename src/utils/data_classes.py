# This file includes the data classes that will be using through the agentic workflow.
# With pydantic, we can ensure the data model will be stable during the process.

import uuid
from typing import List

from pydantic import BaseModel, Field


class Task(BaseModel):
    """
    Our task model defines the properties that the task object will have.
    Task id is a unique identifier for the task.
    """
    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="Unique identifier for the task")
    task_name: str = Field(description="Name of the task")
    task_description: str = Field(description="Description of the task")
    estimated_day: int = Field(description="Estimated number of days to complete the task")


class TaskList(BaseModel):
    """
    List of tasks.
    """
    tasks: List[Task] = Field(description="List of tasks")


class TaskDependency(BaseModel):
    """
    Task dependency defines the relationships between tasks.
    """
    task: Task = Field(description="Task")
    dependent_tasks: List[Task] = Field(description="List of dependent tasks")


class TeamMember(BaseModel):
    """
    Team member model.
    It has name and description for that member.
    """
    name: str = Field(description="Name of the team member")
    profile: str = Field(description="Profile of the team member")


class Team(BaseModel):
    """
    List of team members.
    """
    team_members: List[TeamMember] = Field(description="List of team members")


# Iterative assessment
class TaskAllocation(BaseModel):
    """Task allocation class"""
    task: Task = Field(description="Task")
    team_member: TeamMember = Field(description="Team members assigned to the task")


class TaskSchedule(BaseModel):
    """
    Task schedules.
    It defines how many days to complete the task.
    """
    task: Task = Field(description="Task")
    start_day: int = Field(description="Start day of the task")
    end_day: int = Field(description="End day of the task")


# Lists
class DependencyList(BaseModel):
    """
    List of dependency lists.
    """
    dependencies: List[TaskDependency] = Field(description="List of task dependencies")


class Schedule(BaseModel):
    """List of task schedules"""
    schedule: List[TaskSchedule] = Field(description="List of task schedules")


class TaskAllocationList(BaseModel):
    """List of task allocations"""
    task_allocations: List[TaskAllocation] = Field(description="List of task allocations")


class TaskAllocationListIteration(BaseModel):
    """List of task allocations for each iteration"""
    task_allocations_iteration: List[TaskAllocationList] = Field(
        description="List of task allocations for each iteration")


class ScheduleIteration(BaseModel):
    """List of task schedules for each iteration"""
    schedule: List[Schedule] = Field(description="List of task schedules for each iteration")


class Risk(BaseModel):
    """Risk of a task"""
    task: Task = Field(description="Task")
    score: str = Field(description="Risk associated with the task")


class RiskList(BaseModel):
    """List of risks for each iteration"""
    risks: List[Risk] = Field(description="List of risks")


class RiskListIteration(BaseModel):
    """List of risks for each iteration"""
    risks_iteration: List[RiskList] = Field(description="List of risks for each iteration")
