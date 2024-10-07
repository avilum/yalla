class PromptBase:
    @property
    def available_tools(self) -> dict:
        raise NotImplementedError

    def break_down_to_steps_prompt(self, user_query):
        raise NotImplementedError

    def llm_query_tool_prompt(self, tool_call_history, task):
        raise NotImplementedError

    def next_step_prompt(self, steps_done, max_steps, call_history, llm_plan, user_query):
        raise NotImplementedError

    def prepare_final_output_prompt(self, tool_call_history, user_query):
        raise NotImplementedError


class LlamaPrompts(PromptBase):
    def __init__(self):
        self._available_tools = [
            {
                "type": "function",
                "function": {
                    "name": "create_text_file",
                    "description": "Creates a text file inside the ubuntu terminal container.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_name": {
                                "type": "string",
                                "description": "The name of the text file to be created.",
                            },
                            "file_content": {
                                "type": "string",
                                "description": "The content of the text file to be created.",
                            },
                        },
                        "required": ["file_name", "file_content"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "ubuntu_terminal",
                    "description": "Runs bash commands within a Ubuntu terminal environment. Use package managers to install dependencies.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The bash command to be executed.",
                            },
                        },
                        "required": ["command"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_browser",
                    "description": "Accesses the internet to extend your knowledge, perform web search, surf websites and URLs, and retrieve their HTML content as plain text.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The URL to be accessed. Use 'https://duckduckgo.com?q=' to search the query if you are not sure.",
                            },
                        },
                        "required": ["url"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "llm_query",
                    "description": "Communicates with an offine LLM and returns the LLM output. Use web_browser prior to this tool, to extend your context about realtime topics, if needed.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The query to be sent to the LLM.",
                            },
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            },
        ]

    @property
    def available_tools(self) -> dict:
        return self._available_tools

    def break_down_to_steps_prompt(self, user_query):
        return f"""
        ##
        **Instructions:**
        You are a helpful assistant who solves problems step-by-step using the tools provided. Your goal is to create a simple and concise plan with as few steps as possible.

        **Input:**
        * **Tools:** {self.available_tools}
        * **Task:** {user_query}

        **Output:**
        A numbered list of steps outlining your plan. Each step should be brief and specific. Separate each step with a new line.

        **Example:**
        1. Use the web_browser tool to search for "population of Tokyo".
        2. Use the llm_query tool to summarize the information and answer the question "What is the population of Tokyo?".
        3. done.
        """

    def llm_query_tool_prompt(self, tool_call_history, task):
        return f"""
        You are a helpful assistant with access to these tools: {self.available_tools}

        Here's what you have done so far:

        * Tool History:
        ```{tool_call_history}```

        * User Request:
        ```{task}```
        Now, you MUST provide a concise answer that accomplishes the User Request based on the Tool History.
        """

    def next_step_prompt(self, steps_done, max_steps, call_history, llm_plan, user_query):
        return f"""
        You are a problem-solving assistant that chooses the next step.
        Your Progress:
        * Steps taken: {steps_done}/{max_steps}
        * Previous steps and tool outputs: {call_history}
        * Plan: {llm_plan}
        NEVER repeat previous steps.
        You have access to these tools: {self.available_tools}
        NEVER make up things.

        * User Query: {user_query}
        Now, call the NEXT tool to accomplish the User Query, using JSON format.
        You MUST return ONLY ONE JSON object: {{"name": "...", "arguments": "..."}}
        If you are able to accomplish/answer the User Query, return ONLY "done"
        """

    def prepare_final_output_prompt(self, tool_call_history, user_query):
        return f"""
        <Tool History> {tool_call_history}
        </Tool History>
        <User Request>{user_query}
        </User Request>
        NOW, you MUST do your best effort: Answer the User Request based on Tool History above.
        Final Output:
        """
