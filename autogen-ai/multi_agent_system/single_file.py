"""
Production Multi-Agent System using AutoGen + Ollama

Features:
- Multiple specialized Agents
- Code execution
- Tool calling 
- Human in loop
- Error handling
"""

import autogen
from typing import Dict, List, Optional
import json
import os


##### 1. Ollama configuration

ollama_config = {
    "config_list": [
        {
            "model": "llama3.2",
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
            "temperature": 0.7,
        }
    ],
    "cache_seed": None,
}

#### 2 code specific configuration
code_llm_config = {
    "config_list": [
        {
            "model": "codellama",
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
            "temperature": 0.3,
        }
    ],
    "cache_seed": None
}

#### define custom tools/function

def search_web(query: str) -> str:
    return f"Search results for '{query}': [Mock results 1], [Mock results 2]"

def analyze_data(data: str) -> Dict:
    return {
        "summary": f"Analyzed {len(data)} characters",
        "insights": ["Insight 1", "Insight 2"],
        "recommendation": "Proceed with caution"
    }

def execute_code_safely(code: str) -> str:
    try:
        return f"Code execution simulated:\n{code}\n[Output: Success]"
    except Exception as e:
        return f"Error: {str(e)}"

### Function map for userProxyAgent

function_map = {
    "search_web": search_web,
    "analyze_data": analyze_data,
    "execute_code_safely": execute_code_safely 
}

### 3. Create agents ####

# UserProxyAgent- Execute code and tools (No llm)

user_proxy = autogen.UserProxyAgent(
    name="UserProxy",
    system_message="A Proxy for Human user. Execute code and tools.",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
        "last_n_messages": 3,
    },
    function_map=function_map,
)

# Planner Agent - Creates High-level plans
planner = autogen.AssistantAgent(
    name="Planner",
    system_message="""You are a strategic planner.
    your role:
    1. Understand user requirements
    2. Create step-by-step plans
    3. Identify which specialists are needed
    4. Delegate task to appropriate agents

    Be concise and structured. End with TERMINATE when done.
    """,
    llm_config=ollama_config,
)

# Task Decomposition Agent
task_decomposer = autogen.AssistantAgent(
    name="TaskDecomposer",
    system_message="""You break down task into smaller subtask.
    your role:
    1. Analyze the plan from Planner
    2. Create Numbered, actionable subtasks
    3. Estimate efforts for each task
    4. Identify dependencies
    output format:
    Task 1: [Description] (Effort: Low/Medium/High)
    Task 2: [Description] (Effort: Low/Medium/High)
    ...
    """,
    llm_config=ollama_config,
)

# Data Analyst Agent
data_analyst = autogen.AssistantAgent(
    name="DataAnalyst",
    system_message="""You are a data analysis expert.
    Your role:
    1. Analyze data requirements
    2. Design data pipelines
    3. Recommend data processing approaches
    4. Generate python code for data analysis

    Always include code examples using pandas and numpy. 
    """,
    llm_config=code_llm_config,
)

# ML Engineer Agent
ml_engineer = autogen.AssistantAgent(
    name="MLEngineer",
    system_message="""You are machine learning engineer.

    Your role:
    1. Design ML models
    2. Write training code
    3. Recommend ML frameworks (scikit-learn, PyTorch, TensorFlow)
    4. Optimize model performance

    Provide complete, executable code.""",
    llm_config=code_llm_config,
)

# Backend Engineer Agent
backend_engineer = autogen.AssistantAgent(
    name="BackendEngineer",
    system_message="""You are backend Software Engineer.

    Your role:
    1. Design APIs and services
    2. Write production-quality Python code
    3. handle error cases
    4. Write FastAPI/Flask code
    focus on clean, maintainable code.
    """,
    llm_config=code_llm_config,
)

### QA Agent
qa_agent = autogen.AssistantAgent(
    name="QATester",
    system_message="""You are QA Engineer focused on testing.
    your role:
    1. Review code for bugs
    2. Write test cases (pytest)
    3. Check edge cases
    4. Verify error handling

    Be thorough and critical.
    """,
    llm_config=ollama_config,
)

# Security Agent
security_agent = autogen.AssistantAgent(
    name="SecurityExpert",
    system_message="""You are a security expert.
    your role:
    1. review code for security vulnerabilities
    2. check for sql injection, XSS, etc.
    3. validate input handling
    4. Recommend security best practices

    Be paranoid but constructive.
    """,
    llm_config=ollama_config,
)

# Critic Agent
critic_agent = autogen.AssistantAgent(
    name="Critic",
    system_message="""You are critic reviewer.
    Your role:
    1. review all outputs
    2. Identify improvements
    3. Challenge assumptions
    4. Ensure quality standards

    Be constructive, not destructive
    """,
    llm_config=ollama_config,
)

## Writer Agent
writer_agent = autogen.AssistantAgent(
    name="TechnicalWriter", 
    system_message="""You are a technical document writer.
    your role:
    1. Create clean documentation
    2. Write README files
    3. Document APIs
    4. Summarize project deliverables

    Make it user-friendly.""",
    llm_config=ollama_config,
)


######  4. CREATE GROUP CHAT
## All agent that participate in discussion (No userProxyAgent here)
groupchat = autogen.GroupChat(
    agents=[
        planner,
        task_decomposer,
        data_analyst,
        ml_engineer,
        backend_engineer,
        qa_agent,
        security_agent,
        critic_agent,
        writer_agent,
    ],
    messages=[],
    max_round=20,
    speaker_selection_method="auto",
    allow_repeat_speaker=False
)

#### GroupChatManager orchestrates the conversation
manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=ollama_config,
)


def run_multi_agent_system(task: str):
    """
    Execute multi-agent workflow
    Args:
        task: User's task description
    """
    print("\n" + "="*60)
    print("Starting Multiagent system")
    print("="*60 + "\n")
    print(f"Task: {task}\n")

    try:
        # UserProxy initiates chat with Manager
        chat_result = user_proxy.initiate_chat(
            manager,
            message=task,
            clear_history=True,
        )
        print("\n" + "="*60)
        print("Conversation Completed")
        print("="*60)
        
        # Extract final summary
        print("\nSummary")
        print(f"Total messages: {len(chat_result.chat_history)}")
        print(f"Cost: NA (Using local LLM)")
        return chat_result
    except Exception as e:
        print(f"\nError: {str(e)}")
        return None
        
if __name__ == "__main__":
    ## Example 1: Group Chat (dynamic routing)
    task1 = """
    Build a machine learning pipeline that:
    1. loads customer data from csv
    2. perform data cleansing and feature engineering
    3. train a churn prediction model
    4. Expose prediction by REST API
    5. include comprehensive tests
    """
    result = run_multi_agent_system(task1)