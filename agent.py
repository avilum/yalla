import argparse
import json
import subprocess
import os
import sys
from time import sleep, time
from functools import lru_cache
import pathlib
import logging
import urllib.parse
import string
import base64
import tempfile
import requests
import openai
import docker
import selenium.webdriver
from bs4 import BeautifulSoup


################################################################## CONFIGURATION ###################################################################
MAX_STEPS = 5
SLEEP_SECONDS_BETWEEN_STEPS = 0
EXECUTOR_MODEL_NAME = "gpt-3.5-turbo-16k"
PLANNER_MODEL_NAME = "gpt-4o"

################################################################## LOGGING ###################################################################

pathlib.Path(".runs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(f".runs/{int(time())}.log"),
        logging.StreamHandler(),
    ],
)


class COLORS:
    CLEARSCREEN = "\033[2J\033[H"
    DEFAULT = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"


################################################################## PROMPTS #########################################################################
AVAILABLE_TOOLS = """
Tools Available:

1. **create_text_file:** Creates a text file inside the ubuntu terminal container. 
   *Example:* `create_text_file: my_file.txt, Hello World!`

2. **ubuntu_terminal:** Runs bash commands within a Ubuntu terminal environment.
   *Warning:* Always use absolute paths and double quotes. Don't use `sudo` or `docker`. Each command runs in a new terminal.
   *Example:* `git clone https://github.com/some/repo && ...`
   *Example:* `ubuntu_terminal: ls -l /workdir && ps aux`

3. **web_browser:** Accesses websites and retrieves their HTML content.
   *Warning:* Always start URLs with "https://".
   *Warning:* Use duckduckgo to search.
   *Example:* `web_browser: https://www.wikipedia.org/`
   *Example:* `web_browser: Hugging Face New Models`
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
You have access to these tools: {available_tools}

Here's what you've done so far:
* Steps taken: {steps_done}/{max_steps}
* Previous steps and tool outputs: {call_history}
* Last input: {last_step}
* Last output: {last_output}
* Plan: {llm_plan}

* Your TOP LEVEL TASK: {user_query}
Now, call exactly ONE of the available tools, using JSON format.
Never repeat previous steps. Don't make up things.
You MUST return nothing but the following JSON object: {{"tool_name": "...", "tool_arguments": "..."}}
If the task is complete, simply reply "done". Be concise. 
"""
####################################################################################################################################################
PREPARE_FINAL_OUTPUT_PROMPT = """
The last tool output was: {last_output}
You have finished solving the task: {user_query}
Based on this, provide a clear and concise final answer to the user's request.
Final Output:
"""
####################################################################################################################################################


class LLMAgent:
    def __init__(
        self,
        docker_client: docker.DockerClient,
        openai_client: openai.OpenAI,
        planner_model_name: str,
        executor_model_name: str,
        available_tools=AVAILABLE_TOOLS,
        llm_query_tool_prompt=LLM_QUERY_TOOL_PROMPT,
        next_step_prompt=NEXT_STEP_PROMPT,
        break_down_to_steps_prompt=BREAK_DOWN_TO_STEPS_PROMPT,
    ):
        logging.debug("Starting agent...")
        logging.debug(f"Backend: {openai_client.base_url}")
        self._docker_client = docker_client
        self._openai_client = openai_client
        self._planner_model_name = planner_model_name
        self._executor_model_name = executor_model_name

        self._available_tools = available_tools
        self._llm__query_tool_prompt = llm_query_tool_prompt
        self._next_step_prompt = next_step_prompt
        self._break_down_to_steps_prompt = break_down_to_steps_prompt
        self.tool_call_history = []
        self.driver: selenium.webdriver = None
        self.container: docker.Container = None
        self._terminal_initialized = False

    def run_task(
        self, user_query: str, max_steps: int, sleep_seconds_between_steps: float
    ):
        logging.info(f"Starting task: {user_query}")
        llm_step_by_step_plan: list[str] = self._break_down_to_steps(user_query)
        logging.debug(f"Plan: {llm_step_by_step_plan}")
        llm_step_by_step_plan = "\n".join(llm_step_by_step_plan)

        print(COLORS.CLEARSCREEN)
        print(
            f"# TASK:{os.linesep}{COLORS.GREEN}{user_query}{COLORS.DEFAULT}{os.linesep}"
        )
        print("💭 LLM Plan 💭")
        print(llm_step_by_step_plan)
        input(f"{os.linesep}Press any key to continue.{os.linesep}")

        last_output = None
        last_step = None
        step_output = None
        i = 0
        while i < max_steps:
            logging.info(f"\n######### Step {i+1} #########")
            logging.info("\nChoosing next step...")
            next_step = self._choose_next_step(
                user_query=user_query,
                llm_plan=llm_step_by_step_plan,
                last_step=last_step,
                last_output=last_output,
                steps_done=i,
                max_steps=max_steps,
                call_history=self.tool_call_history,
            )
            if next_step[:4].lower().startswith("done"):
                logging.debug(
                    "The LLM marked the task as 'done', preparing final output."
                )
                last_step = next_step
                last_output = step_output
                break

            try:
                json_start, json_end = next_step.index("{"), next_step.rindex("}") + 1
                step_json_str = next_step[json_start:json_end]
                step_json = json.loads(step_json_str)
            except (json.JSONDecodeError, ValueError):
                logging.warning(
                    f"Invalid Call! Expected JSON with tool_name and tool_arguments. Got: '{next_step}'"
                )
                continue

            try:
                tool_name = step_json["tool_name"]
            except KeyError:
                logging.warning(
                    f"Invalid Call! Expected JSON with tool_name and tool_arguments, but tool_name is missing. Got: '{next_step}'"
                )
                continue

            try:
                tool_arguments = step_json["tool_arguments"]
            except KeyError:
                logging.warning(
                    f"Invalid Call! Expected JSON with tool_name and tool_arguments, but tool_name is missing. Got: '{next_step}'"
                )
                continue

            logging.info(f"🛠️ Tool: {COLORS.CYAN}{tool_name}{COLORS.DEFAULT}")
            logging.info(f"🔧 Arguments: {COLORS.CYAN}{tool_arguments}{COLORS.DEFAULT}")
            match tool_name:
                case "create_text_file":
                    cmd_input = tool_arguments
                    delimiter_idx = cmd_input.index(",")
                    file_name = cmd_input[:delimiter_idx]
                    content = cmd_input[delimiter_idx + 1 :]
                    step_output = self._create_text_file(str(file_name), str(content))
                case "ubuntu_terminal":
                    step_output = self._run_terminal_command(str(tool_arguments))
                case "web_browser":
                    step_output = self._browse_page(str(tool_arguments))
                case "llm_query":
                    llm_prompt_with_context = self._llm__query_tool_prompt.format(
                        available_tools=self._available_tools,
                        tool_call_history=self.tool_call_history,
                        last_step=last_step,
                        last_output=last_output,
                        task=tool_arguments,
                    )
                    step_output = self._communicate_with_llm(llm_prompt_with_context)
                    logging.info(
                        f"LLM Output: {COLORS.YELLOW}{step_output}{COLORS.DEFAULT}"
                    )
                case _:
                    step_output = f"Bad Tool Call! Expected one of: create_text_file, ubuntu_terminal, web_browser, llm_query. Got: {tool_name}"
                    logging.warning(f"{COLORS.RED}{step_output}{COLORS.DEFAULT}")
                    i += 1
                    continue

            # TODO: Add re-ranking to history, keeping the most relevant steps.
            # Appending output to history
            step_output = str(step_output)
            self.tool_call_history.append((next_step, step_output))

            last_step = next_step
            last_output = step_output

            i += 1
            logging.debug(f"Sleeping for {sleep_seconds_between_steps} seconds...")
            sleep(sleep_seconds_between_steps)
            logging.info(f"######### Finished step {i} #########")

        # Prepare the final output using llm query
        final_str_prompt = PREPARE_FINAL_OUTPUT_PROMPT.format(
            user_query=user_query,
            # last_output=last_output
            last_output=self.tool_call_history,
        )
        step_output = self._communicate_with_llm(final_str_prompt)
        last_output = step_output

        print(f"{os.linesep}Agent Output:")
        print(f"{COLORS.GREEN}{last_output}{COLORS.DEFAULT}")
        print("🤖 Thanks for trying LLM Agent 🤖")

    def _create_text_file(self, file_name: str, file_content: str):
        """
        Creates a temporary text file locally and copies it to the container.
        This tool was created to enable multi-line code files with indentations,
        Instead of using 'echo <<EOF ...' which is breaking in the docker command escaping.
        """
        if not self._terminal_initialized:
            self._initialize_container()

        file_content = file_content.lstrip()
        file_content = f"""{file_content.strip('"').strip("'")}"""
        logging.debug(f"# FILENAME: {file_name}")
        logging.debug(f"# CONTENT:{os.linesep}{file_content}")
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file_content.encode())

        subprocess.run(
            f'docker cp "{temp_file.name}" {self.container.id}:{file_name.lstrip()}',
            shell=True,
            check=True,
        )

        # Cleanup
        temp_file.close()
        os.unlink(temp_file.name)
        return f"Successfully created {file_name}"

    def _run_terminal_command(self, command):
        if not self._terminal_initialized:
            self._initialize_container()

        command = command.strip()
        if command.startswith('"') and command.endswith('"'):
            command = command.strip('"')

        logging.info(f" 🖥️ TERMINAL IN DOCKER: {COLORS.GREEN}{command}{COLORS.DEFAULT}")

        # Encoding the command to base64 to avoid escaping errors with code and multi-lines.
        base64_command = base64.b64encode(command.encode()).decode()
        bash_command = f"""/bin/bash -c 'echo {base64_command} | base64 -d | bash'"""

        try:
            cmd_output = self.container.exec_run(
                bash_command,
                stdin=True,
                workdir="/workdir",
            )
            if cmd_output.exit_code != 0:
                raise Exception(f"ERROR: {cmd_output}")

            logging.info(
                f"✅ TERMINAL OUTPUT: {COLORS.GREEN}{cmd_output}{COLORS.DEFAULT}"
            )
        except Exception as e:
            cmd_output = str(e)
            logging.info(
                f"🔴 TERMINAL ERROR: {COLORS.YELLOW}{cmd_output}{COLORS.DEFAULT}"
            )
        return cmd_output

    def _initialize_container(self):
        logging.info("Setting up Docker container...")
        self.container = self._docker_client.containers.run(
            "python",
            command="sleep 180",
            working_dir="/workdir",
            environment=["DEBIAN_FRONTEND=noninteractive"],
            remove=True,
            detach=True,
        )
        self.container.exec_run("apt update -y && apt-get install sudo cd -y")
        self._terminal_initialized = True

    def _break_down_to_steps(self, user_query):
        prompt = self._break_down_to_steps_prompt.format(
            available_tools=self._available_tools, user_query=user_query
        )
        step_by_step_tasks_stream = self._openai_client.chat.completions.create(
            model=self._planner_model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        llm_step_by_step_plan_text = self._process_stream_until_complete(
            stream=step_by_step_tasks_stream, live_print_to_stdout=True
        )
        return [_ for _ in llm_step_by_step_plan_text.split("\n") if _]

    @lru_cache(maxsize=128)
    def _fetch_page_content_for_agent(self, url: str):
        """
        Selenium-free Alternative to "_browse_page" method.
        Browses a clean page congtent via Jina free tools, eliminating the need for selenium.
        """
        if not url:
            return "Please provide a valid URL to the browser tool"

        url = url.strip().strip(string.punctuation)

        # TODO: If the llm mistakenly provides a URL without http, use duckduck go search.
        # if not url.startswith("http"):
        #     url = urllib.parse.quote(url)
        #     url = f"https://duckduckgo.com/{url}"

        # TODO: Reader API is great but doesn't work with all websites, such as duckduckgo/google.
        # url = f"https://r.jina.ai/{url}" # Jina Reader API (free)

        url = f"https://s.jina.ai/{url}"  # Jina Search API (free)
        try:
            res = requests.get(url, stream=True, timeout=10)
            res.raise_for_status()
            res = res.text
            print(res)
            return res
        except Exception as e:
            return str(e)

    @lru_cache(maxsize=128)
    def _browse_page(self, url: str):
        """
        Browses a page with selenium. Once loading is done, extracts all text from the text tags - except scripts, headers and footers.
        Links are enriched with '**<link_title>**: <link_href>' to enable continous browsing and improve context.
        """
        if not self.driver:
            self.driver = selenium.webdriver.Firefox()

        if not url:
            return "Please provide a valid URL to the browser tool"

        url = url.strip().strip(string.punctuation)

        if not url.startswith("http"):
            url = urllib.parse.quote(url)
            url = f"https://duckduckgo.com/{url}"

        # url = f"https://s.jina.ai/{url}"  # Jina Search API (free)
        logging.info(f"{COLORS.CYAN}📤 Browsing page: {url}{COLORS.DEFAULT}")
        try:
            self.driver.get(url)
        except selenium.common.exceptions.WebDriverException as e:
            return f"Failed to get {url}. Error: {e}"

        source = self.driver.page_source
        soup = BeautifulSoup(
            source, "html.parser"
        )  # html parser is fast enough and extracts text slower yet better than lxml.

        # Remove all script and style elements
        for tag_to_drop in soup(["script", "style", "footer", "header"]):
            tag_to_drop.extract()

        # Find all <a> tags and get their href attributes
        links = [
            (anchor.get_text(strip=True), anchor.get("href"))
            for anchor in soup.find_all("a")
        ]

        text = soup.get_text()

        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())

        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

        # Drop blank lines
        text = "\n".join(chunk for chunk in chunks if chunk)

        # Combine text with links for better context
        text_content_with_links = text + "\n".join(
            f"*{title}*: {link}" for title, link in links if link
        )
        print(text_content_with_links)
        logging.debug("Done browsing.")
        return text_content_with_links

    def _communicate_with_llm(self, prompt):
        logging.debug(f"🤖 Communicating with LLM, prompt: {prompt}")
        current_stream = self._openai_client.chat.completions.create(
            model=self._executor_model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        return self._process_stream_until_complete(
            current_stream, live_print_to_stdout=True
        )

    def _choose_next_step(
        self,
        user_query,
        llm_plan,
        last_step,
        last_output,
        steps_done,
        max_steps,
        call_history,
    ):
        """
        Chooses the next step for the LLM agent based on the given parameters.

        Args:
            user_query (str): The user's query.
            llm_plan (str): The LLM plan.
            last_step (str): The last step performed by the agent.
            last_output (str): The output of the last step performed by the agent.
            steps_done (int): The number of steps already performed.
            max_steps (int): The maximum number of steps allowed.
            call_history (list): The history of API calls made by the agent.

        Returns:
            str: The next step to be performed by the agent.
        """
        prompt = self._next_step_prompt.format(
            available_tools=self._available_tools,
            user_query=user_query,
            llm_plan=llm_plan,
            call_history=call_history,
            last_step=last_step,
            last_output=last_output,
            steps_done=steps_done,
            max_steps=max_steps,
        )
        logging.debug(f"Executing prompt: {prompt}")
        next_step_response_stream = self._openai_client.chat.completions.create(
            model=self._executor_model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        next_step_str = self._process_stream_until_complete(
            next_step_response_stream, live_print_to_stdout=False
        )
        next_step_str = (
            next_step_str.strip(" ").strip("-").strip("<").strip(">").strip('"')
        )
        logging.info(f"🤖 Next: {next_step_str}")
        return next_step_str

    @staticmethod
    def _process_stream_until_complete(
        stream, live_print_to_stdout: bool = False
    ) -> str:
        """
        A helper method to process the stream of responses from the OpenAI API.
        """
        txt = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                if live_print_to_stdout:
                    print(
                        f"{COLORS.CYAN}{chunk.choices[0].delta.content}{COLORS.DEFAULT}",
                        end="",
                    )
                txt += chunk.choices[0].delta.content
        return txt

    def __del__(self):
        """
        Cleanup for the agent
        """
        try:
            if self.driver:
                logging.debug("Closing Selenium driver...")
                self.driver.close()
            if self.container.status == "running":
                logging.debug("Stopping Docker container...")
                self.container.stop()
        except Exception:
            ...


def main(
    user_query: str,
    should_use_local_llm: bool,
    planner_model_name: str,
    executor_model_name: str,
    max_steps: int,
    sleep_seconds_between_steps: float,
):
    # Set up Docker API client
    # docker_client: docker.DockerClient = docker.DockerClient()
    docker_client: docker.DockerClient = docker.DockerClient(
        "unix:///Users/avi/.colima/default/docker.sock"
    )
    if should_use_local_llm:
        openai_client: openai.OpenAI = openai.OpenAI(
            # Or, using ollama/llama.cpp server:
            base_url="http://127.0.0.1:11434/v1",
            api_key="test",
        )
    else:
        openai_client: openai.OpenAI = openai.OpenAI(
            api_key=os.environ["OPENAI_API_KEY"]
        )

    agent = LLMAgent(
        docker_client=docker_client,
        openai_client=openai_client,
        planner_model_name=planner_model_name,
        executor_model_name=executor_model_name,
    )
    agent.run_task(
        user_query=user_query,
        max_steps=max_steps,
        sleep_seconds_between_steps=sleep_seconds_between_steps,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Agent")
    parser.add_argument("--query", type=str, help="User query for the LLM Agent")
    parser.add_argument("--verbose", help="Verbose output", action="store_true")
    parser.add_argument(
        "--planner",
        type=str,
        help="The name of the model that will plan the steps.",
        default=PLANNER_MODEL_NAME,
    )
    parser.add_argument(
        "--executor",
        type=str,
        help="The name of the model that will execute each step.",
        default=EXECUTOR_MODEL_NAME,
    )
    parser.add_argument(
        "--local",
        help="Instead of OpenAI. Use local LLM backend (ollama/llama.cpp) on port 11434",
        action="store_true",
    )
    parser.add_argument(
        "--steps",
        help="The maximum number of steps to take to solve the problem.",
        type=int,
        default=MAX_STEPS,
    )
    parser.add_argument(
        "--sleep",
        help="The number of seconds to sleep between steps.",
        type=float,
        default=SLEEP_SECONDS_BETWEEN_STEPS,
    )
    args = parser.parse_args()

    # Set OpenAI logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("openai").setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger("openai").setLevel(logging.WARNING)

    planner_model_name = args.planner
    executor_model_name = args.executor
    if args.local and (
        planner_model_name.startswith("gpt-") or executor_model_name.startswith("gpt-")
    ):
        print(
            f"""local backends doesn\'t support GPT models: planner="{planner_model_name}" \
executor="{executor_model_name}"; Use gemma2 instead. """
        )
        sys.exit(1)

    if not args.query:
        print("Please provide a user query using --query")
        sys.exit(1)

    max_steps = args.steps
    sleep_seconds_between_steps = args.sleep
    main(
        args.query,
        args.local,
        planner_model_name,
        executor_model_name,
        max_steps,
        sleep_seconds_between_steps,
    )
