import io
import logging
import time
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.agent_state import AgentState
from utils.agent_utils import select_llm_agent
from utils.api_handler import check_api_key, get_api_key, APIKeyError
from utils.workflow_nodes import (
    create_workflow,
    compile_workflow
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_ITERATIONS = 3  # Fixed number of iterations
MODEL_TYPE = "gpt-4o-mini"  # Fixed model type
MODEL_NAMES = {
    "gpt-4o-mini": "gpt-4o-mini"
}

ENV_VAR_MAPPING = {
    "gpt-4o-mini": "OPENAI_API_KEY"
}

# Initialize session states
if 'team_members' not in st.session_state:
    st.session_state.team_members = []

if 'project_data' not in st.session_state:
    st.session_state.project_data = {
        'description': '',
        'final_state': None,
        'results_df': None
    }

def delete_team_member(index):
    """Delete a team member from the session state"""
    if 'team_members' in st.session_state:
        st.session_state.team_members.pop(index)
        st.rerun()


def create_gantt_figure(final_state, number_of_iterations):
    """Create Gantt chart figure from state data with enhanced readability"""
    schedule = final_state['schedule_iteration'][-1].schedule
    allocations = final_state['task_allocations_iteration'][-1].task_allocations

    # Prepare schedule data
    schedule_data = []
    for ts in schedule:
        task_data = {
            'task_name': ts.task.task_name,
            'start_day': ts.start_day,
            'end_day': ts.end_day,
            'description': ts.task.task_description,
            'id': ts.task.id,
            'estimated_days': ts.task.estimated_day
        }
        schedule_data.append(task_data)

    # Prepare allocation data
    allocation_data = []
    for ta in allocations:
        allocation_data.append({
            'task_name': ta.task.task_name,
            'team_member': ta.team_member.name,
            'profile': ta.team_member.profile
        })

    df_schedule = pd.DataFrame(schedule_data)
    df_allocation = pd.DataFrame(allocation_data)
    df = df_allocation.merge(df_schedule, on='task_name')

    # Convert days to dates
    current_date = datetime.today()
    df['start'] = df['start_day'].apply(lambda x: current_date + timedelta(days=x))
    df['end'] = df['end_day'].apply(lambda x: current_date + timedelta(days=x))

    df.rename(columns={'team_member': 'Team Member'}, inplace=True)
    df.sort_values(by='Team Member', inplace=True)

    # Custom color palette
    custom_colors = [
        '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

    # Create visualization
    fig = px.timeline(
        df,
        x_start="start",
        x_end="end",
        y="task_name",
        color="Team Member",
        title=f"Project Timeline - Final Iteration",
        color_discrete_sequence=custom_colors,
        hover_data=['description', 'estimated_days', 'profile']
    )

    # Update layout
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        title={
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20, color='#2c3e50')
        },
        xaxis=dict(
            title="Timeline",
            title_font=dict(size=14, color='#2c3e50'),
            showgrid=True,
            showline=True,
            showticklabels=True,
            linecolor='#2c3e50',
            linewidth=2,
            ticks='outside',
            tickfont=dict(size=12, color='#2c3e50')
        ),
        yaxis=dict(
            title="Tasks",
            title_font=dict(size=14, color='#2c3e50'),
            showgrid=True,
            showline=True,
            showticklabels=True,
            linecolor='#2c3e50',
            linewidth=2,
            ticks='outside',
            tickfont=dict(size=12, color='#2c3e50'),
            autorange="reversed"
        ),
        height=max(400, len(df) * 40),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='#2c3e50',
            borderwidth=1,
            font=dict(size=12, color='#2c3e50'),
            title=dict(text="Team Members", font=dict(size=14, color='#2c3e50'))
        ),
        margin=dict(l=100, r=100, t=80, b=80)
    )

    # Add gridlines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(189, 195, 199, 0.4)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(189, 195, 199, 0.4)')

    return fig, df


def display_task_details(df):
    """Display detailed task information in an expandable section"""
    with st.expander("Task Details", expanded=True):
        for _, row in df.iterrows():
            st.markdown(f"### {row['task_name']}")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Team Member:** {row['Team Member']}")
                st.markdown(f"**Duration:** {row['estimated_days']} days")
            with col2:
                st.markdown(f"**Start Date:** {row['start'].strftime('%Y-%m-%d')}")
                st.markdown(f"**End Date:** {row['end'].strftime('%Y-%m-%d')}")
            st.markdown("**Description:**")
            st.markdown(row['description'])
            st.markdown("---")


def display_project_insights(final_state):
    """Display project insights in an expandable section"""
    if 'insights' in final_state and final_state['insights']:
        with st.expander("Project Insights", expanded=True):
            st.markdown(final_state['insights'])


def process_project(project_description, status_container, progress_bar):
    try:
        status_container.text("Checking API configuration...")
        progress_bar.progress(0.05)

        api_key_exists, error_message = check_api_key(MODEL_TYPE, ENV_VAR_MAPPING)
        if not api_key_exists:
            raise APIKeyError(error_message)

        status_container.text("Initializing agent and workflow...")
        progress_bar.progress(0.1)

        # Initialize workflow with the fixed model
        api_key = get_api_key(MODEL_TYPE, ENV_VAR_MAPPING)
        llm_agent = select_llm_agent(model=MODEL_TYPE, api_key=api_key, model_mapping=MODEL_NAMES)
        workflow = create_workflow(state=AgentState, llm_agent=llm_agent)
        graph_plan, memory = compile_workflow(workflow, create_memory=True)

        team_data = [
            {"name": member["name"], "role": member["role"]}
            for member in st.session_state.team_members
        ]

        state_input = {
            "project_description": project_description,
            "team": team_data,
            "insights": "",
            "iteration_number": 0,
            "max_iteration": MAX_ITERATIONS,
            "schedule_iteration": [],
            "task_allocations_iteration": [],
            "risks_iteration": [],
            "project_risk_score_iterations": []
        }

        config = {"configurable": {"thread_id": "1"}, "recursion_limit": 100}

        # Process steps with visual feedback
        processing_steps = [
            "Analyzing project requirements",
            "Generating tasks",
            "Allocating resources",
            "Assessing risks",
            "Creating timeline",
            "Finalizing plan"
        ]
        total_steps = len(processing_steps)

        # Process workflow with progress updates
        for event_idx, event in enumerate(graph_plan.stream(state_input, config, stream_mode=["updates"])):
            current_step = processing_steps[min(event_idx, total_steps - 1)]
            progress = min(1.0, (event_idx + 1) / total_steps)
            status_container.text(f"⚡ {current_step}")
            progress_bar.progress(progress)
            time.sleep(0.1)

        # Get results and convert back to DataFrame where needed
        final_state = graph_plan.get_state(config).values

        # If team data needs to be in DataFrame format, convert it back
        if isinstance(final_state.get('team', []), list):
            final_state['team'] = pd.DataFrame(final_state['team'])

        # Store results in session state
        st.session_state.project_data['description'] = project_description
        st.session_state.project_data['final_state'] = final_state

        # Create and store the dataframe
        fig, df = create_gantt_figure(final_state, final_state['iteration_number'])
        st.session_state.project_data['results_df'] = df

        return final_state, df

    except Exception as e:
        logger.error(f"Error in process_project: {str(e)}", exc_info=True)
        raise


def main():
    # App configuration
    st.set_page_config(
        page_title="AI Agent Project Planner",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Main app header with styling
    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 2rem;
        }
        .model-info {
            text-align: right;
            margin-bottom: 1rem;
            color: #666;
            font-size: 0.95em;
        }
        </style>
        <div class="main-header">
            <h1>Project Manager AI Agent</h1>
        </div>
        """, unsafe_allow_html=True)


    model_info = f"Model: {MODEL_TYPE}"
    st.markdown(f'<div class="model-info">{model_info}</div>', unsafe_allow_html=True)

    # Main content area
    # 1. Project Description Section
    st.header("Project Description")
    project_description = st.text_area(
        "Describe your project in detail",
        value=st.session_state.project_data.get('description', ''),
        height=150,
        placeholder="Enter your project description here...",
        help="Provide a detailed description of your project, including goals, requirements, and any specific constraints."
    )

    # Update project description in session state if changed
    if project_description != st.session_state.project_data.get('description', ''):
        st.session_state.project_data['description'] = project_description

    # 2. Team Setup Section
    st.header("Team Setup")

    # Create columns for the form and team list
    team_form_col, current_team_col = st.columns([1, 1])

    with team_form_col:
        st.subheader("Add Team Member")
        with st.form(key='team_member_form', clear_on_submit=True):
            member_name = st.text_input(
                "Name",
                key="name",
                placeholder="Enter team member's name..."
            )
            member_role = st.text_input(
                "Role",
                key="role",
                placeholder="Enter member's role, skills and experience..."
            )

            cols = st.columns([3, 1])
            with cols[1]:
                submit = st.form_submit_button(
                    "Add Member",
                    type="primary",
                    use_container_width=True
                )

            if submit:
                if not member_name or not member_role:
                    st.error("Please fill in both name and role")
                else:
                    st.session_state.team_members.append({
                        "name": member_name,
                        "role": member_role
                    })
                    st.success(f"Added {member_name} as {member_role}")
                    st.rerun()

    with current_team_col:
        st.subheader("Current Team")

        # Add CSS for vertical alignment
        st.markdown("""
            <style>
                .team-member-container {
                    display: flex;
                    align-items: center;
                    padding: 0.5rem 0;
                }
                .team-member-name {
                    flex: 2;
                    margin: 0;
                }
                .team-member-role {
                    flex: 2;
                    margin: 0;
                }
                .team-member-delete {
                    flex: 1;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }
                .stButton {
                    margin: auto;
                }
            </style>
        """, unsafe_allow_html=True)

        if not st.session_state.team_members:
            st.info("No team members added yet!")
        else:
            for idx, member in enumerate(st.session_state.team_members):
                # Create columns with custom HTML/CSS
                cols = st.columns([2, 2, 1])

                # Ensure all elements are in the same container for alignment
                with cols[0]:
                    st.markdown(
                        f"""
                        <div class="team-member-container">
                            <p class="team-member-name"><strong>{member['name']}</strong></p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with cols[1]:
                    st.markdown(
                        f"""
                        <div class="team-member-container">
                            <p class="team-member-role">{member['role']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with cols[2]:
                    if st.button("Delete", key=f"delete_{idx}", help="Remove team member"):
                        delete_team_member(idx)

                st.divider()

    # 3. Generate Plan Section
    st.header("Generate Plan")
    # Process Project button
    if st.button("Process Project", type="primary", use_container_width=True):
        if not project_description:
            st.error("Please enter a project description!")
            return
        if not st.session_state.team_members:
            st.error("Please add at least one team member!")
            return

        try:
            # Create progress containers
            status_container = st.empty()
            progress_bar = st.progress(0)

            # Process project with status updates
            with st.spinner("Processing project..."):
                final_state, df = process_project(
                    project_description,
                    status_container,
                    progress_bar
                )

            st.success("Processing completed successfully!")
            st.rerun()  # Rerun to show results

        except APIKeyError as e:
            logger.error(f"API key error: {str(e)}")
            st.error(f"API Configuration Error: {str(e)}")
        except Exception as e:
            logger.error(f"Error in process_project: {str(e)}", exc_info=True)
            st.error(f"An error occurred: {str(e)}")

    # Check if we have results in session state
    has_results = (
            st.session_state.project_data.get('final_state') is not None and
            st.session_state.project_data.get('results_df') is not None
    )

    # Display results if we have them
    if has_results:
        st.header("Project Results")
        final_state = st.session_state.project_data['final_state']
        df = st.session_state.project_data['results_df']

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Iterations", final_state['iteration_number'])
        with col2:
            risk_score = final_state.get('project_risk_score_iterations', [])[-1]
            st.metric("Risk Score", f"{risk_score:.2f}")
        with col3:
            total_tasks = len(final_state['schedule_iteration'][-1].schedule)
            st.metric("Total Tasks", total_tasks)

        # Create and display Gantt chart
        fig, _ = create_gantt_figure(final_state, final_state['iteration_number'])
        st.plotly_chart(fig, use_container_width=True)

        # Display task details
        display_task_details(df)

        # Display project insights
        display_project_insights(final_state)

        # Export options
        st.header("Export Project Data")
        export_col1, export_col2 = st.columns(2)

        with export_col1:
            # Excel Export
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                # Project Overview Sheet
                overview_data = {
                    'Metric': [
                        'Total Tasks',
                        'Project Duration (days)',
                        'Team Size',
                        'Risk Score',
                        'Number of Iterations'
                    ],
                    'Value': [
                        len(df),
                        df['end_day'].max(),
                        len(st.session_state.team_members),
                        final_state.get('project_risk_score_iterations', [0])[-1],
                        final_state.get('iteration_number', 0)
                    ]
                }
                pd.DataFrame(overview_data).to_excel(writer, sheet_name='Project Overview', index=False)

                # Timeline Sheet
                timeline_df = df.copy()
                timeline_df['Duration (days)'] = timeline_df['end_day'] - timeline_df['start_day']
                timeline_df = timeline_df[[
                    'task_name', 'Team Member', 'start', 'end', 'Duration (days)',
                    'description', 'estimated_days'
                ]]
                timeline_df.to_excel(writer, sheet_name='Timeline', index=False)

                # Team Members Sheet
                team_df = pd.DataFrame(st.session_state.team_members)
                team_df.to_excel(writer, sheet_name='Team', index=False)

                # Insights Sheet
                if 'insights' in final_state and final_state['insights']:
                    insights_data = {
                        'Project Insights': [final_state['insights']]
                    }
                    pd.DataFrame(insights_data).to_excel(writer, sheet_name='Insights', index=False)

            st.download_button(
                label="Download Excel Report",
                data=buffer.getvalue(),
                file_name=f"project_plan_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="excel_download"
            )

        with export_col2:
            # Generate markdown report
            report = []
            report.append("# Project Plan Report\n")
            report.append(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")
            report.append("## Project Overview")
            report.append(st.session_state.project_data.get('description', 'No description available'))
            report.append("\n")

            # Team Information
            report.append("## Team Composition")
            for member in st.session_state.team_members:
                report.append(f"- **{member['name']}**: {member['role']}")
            report.append("\n")

            # Timeline Summary
            report.append("## Project Timeline")
            report.append(f"- **Total Duration**: {df['end_day'].max()} days")
            report.append(f"- **Number of Tasks**: {len(df)}")
            report.append(f"- **Start Date**: {df['start'].min().strftime('%Y-%m-%d')}")
            report.append(f"- **End Date**: {df['end'].max().strftime('%Y-%m-%d')}\n")

            # Tasks Breakdown
            report.append("## Tasks Breakdown")
            for _, row in df.iterrows():
                report.append(f"### {row['task_name']}")
                report.append(f"- **Assigned to**: {row['Team Member']}")
                report.append(f"- **Duration**: {row['estimated_days']} days")
                report.append(f"- **Start**: {row['start'].strftime('%Y-%m-%d')}")
                report.append(f"- **End**: {row['end'].strftime('%Y-%m-%d')}")
                report.append(f"- **Description**: {row['description']}\n")

            # Project Insights
            if 'insights' in final_state and final_state['insights']:
                report.append("## Project Insights")
                report.append(final_state['insights'])

            markdown_content = '\n'.join(report)

            st.download_button(
                label="Download Markdown Report",
                data=markdown_content,
                file_name=f"project_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
                use_container_width=True,
                key="markdown_download"
            )
    # Add space after the Process Project button
    st.markdown("<div style='margin: 5rem 0;'></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
