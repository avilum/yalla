################################################################## PROMPTS #########################################################################
AVAILABLE_TOOLS = """
Tools Available:

1. **create_text_file:** Creates a text file inside the ubuntu terminal container. 
   *Example:* `create_text_file: my_file.txt, Hello World!`

2. **ubuntu_terminal:** Runs bash commands within a Ubuntu terminal environment.
   *Warning:* ALWAYS use /workdir and absolute paths. ALWAYS and double quotes. Don't use `sudo` or `docker`.
   *Example:* `git clone https://github.com/some/repo && ...`
   *Example:* `ubuntu_terminal: ls -l /workdir && ps aux`

3. **web_browser:** Accesses websites and retrieves their HTML content.
   *Warning:* Always start URLs with "https://".
   *Warning:* Use duckduckgo to search.
   *Example:* `web_browser: https://www.wikipedia.org/`
   *Example:* `web_browser: News about Israel`
   *Example:* `web_browser: Who is ...`

4. **llm_query:** Sends prompts to a large language model (LLM) and receives its responses.
   *Example:* `llm_query: What is the capital of France?` 
   *Example:* `llm_query: What are the top models based on the given context?` 
"""

####################################################################################################################################################
BREAK_DOWN_TO_STEPS_PROMPT = """
##
**Instructions:**
You are a helpful assistant who solves problems step-by-step using the tools provided. Your goal is to create a simple and concise plan with as few steps as possible. 

**Input:**
* **Tools:** {available_tools}
* **Task:** {user_query}

**Output:**
A numbered list of steps outlining your plan. Each step should be brief and specific. Separate each step with a new line.

**Example:**
1. Use the web_browser tool to search for "population of Tokyo".
2. Use the llm_query tool to summarize the information and answer the question "What is the population of Tokyo?".
3. done.
"""

####################################################################################################################################################
LLM_QUERY_TOOL_PROMPT = """
You are a helpful assistant with access to these tools: {available_tools} 

Here's what you've done so far:
* Tool history: 
```{tool_call_history}```
* Last input:
```{last_step}```
* Last output:
```{last_output}```

Your task is:
```{task}```
Provide a concise answer to the task using the information from the tools.

Concise Answer:
"""

####################################################################################################################################################
NEXT_STEP_PROMPT = """
You are a problem-solving assistant that chooses the next step.
Your Progress:
* Steps taken: {steps_done}/{max_steps}
* Previous steps and tool outputs: {call_history}
* Plan: {llm_plan}

You have access to these tools: {available_tools}
WARNING: NEVER repeat previous steps. NEVER make up things.

* Your TOP LEVEL TASK: {user_query}
Now, call exactly ONE of the available tools, using JSON format.
You MUST return ONLY ONE JSON object: {{"tool_name": "...", "tool_arguments": "..."}}
If you have enough context to complete the task, you MUST reply "done".
"""

####################################################################################################################################################
PREPARE_FINAL_OUTPUT_PROMPT = """
The last tool output was: {last_output}
You have finished solving the task: {user_query}
Based on this, provide a clear and concise final answer to the user's request.
Final Output:
"""
