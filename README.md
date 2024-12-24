# P-Mage: Project Manager AI Agent

An intelligent project planning agent application powered by GPT-4o-Mini that helps you organize, schedule, and manage your projects efficiently.

## Requirements

- Python 3.11.8
- Required packages (install via `pip install -r requirements.txt`)
- OpenAI API key (required for GPT-4 model access)

## Features

### Project Planning
- Interactive project description input
- Dynamic team member management
- Automated task generation and scheduling
- Resource allocation optimization
- Risk assessment and scoring
- Gantt chart visualization of project timeline

### Team Management
- Add/remove team members
- Define roles and responsibilities
- Track member assignments
- Resource allocation visualization

### Real-time Processing
- Step-by-step project analysis
- Live progress tracking
- Visual feedback during processing
- Iterative plan refinement

### Data Visualization
- Interactive Gantt charts
- Team allocation views
- Risk assessment metrics
- Project timeline visualization

### Export Options
- Excel reports with multiple sheets:
  - Project Overview
  - Timeline
  - Team Composition
  - Project Insights
- Markdown reports including:
  - Project Description
  - Team Information
  - Task Breakdown
  - Timeline Summary
  - Project Insights

## Setup

1. Clone the repository:
```bash
git clone https://github.com/selcuksntrk/p_mage_ai_agent.git
cd p_mage_ai_agent
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key (two options):

   Option 1: Environment Variable
   - Set the OPENAI_API_KEY environment variable in your system:
     ```bash
     export OPENAI_API_KEY=your_api_key_here  # For Linux/Mac
     set OPENAI_API_KEY=your_api_key_here     # For Windows
     ```

   Option 2: Configuration File
   - Create a `config.ini` file in the project root
   - Add your API key:
     ```ini
     [OPENAI]
     API_KEY=your_api_key_here (without quotes)
     ```

4. Run the application:
```bash
streamlit run src/app.py
```

## Usage

1. **Project Description**
   - Enter a detailed description of your project
   - Include goals, requirements, and constraints

2. **Team Setup**
   - Add team members with their names and roles
   - Define skills and experience levels
   - Manage team composition through the UI

3. **Generate Plan**
   - Click "Process Project" to start the analysis
   - Monitor real-time progress
   - Review generated timeline and assignments

4. **Review Results**
   - Examine the Gantt chart
   - Review task details and assignments
   - Check project insights and risk assessment

5. **Export Data**
   - Download Excel report for detailed analysis
   - Export Markdown report for documentation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.