import argparse
import base64
import json
import logging
import os
import pathlib
import string
import subprocess
import sys
import tempfile
import urllib.parse
from abc import ABC
from abc import abstractmethod
from functools import lru_cache
from time import sleep
from time import time

import docker
import openai
import requests
import selenium.webdriver
from bs4 import BeautifulSoup

from prompts import LlamaPrompts
from prompts import PromptBase


################################################################## CONFIGURATION ###################################################################
MAX_STEPS = 3
SLEEP_SECONDS_BETWEEN_STEPS = 0
EXECUTOR_MODEL_NAME = "gpt-4o-mini"
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


class AbstractLLMAgent(ABC):
    def __init__(
        self,
        docker_client: docker.DockerClient,
        openai_client: openai.OpenAI,
        planner_model_name: str,
        executor_model_name: str,
        prompt: PromptBase,
        use_jina: bool = False,
    ):
        logging.debug("Starting agent...")
        logging.debug(f"Backend: {openai_client.base_url}")
        self._docker_client = docker_client
        self._openai_client = openai_client
        self._planner_model_name = planner_model_name
        self._executor_model_name = executor_model_name
        self.prompts = prompt
        self.tool_call_history = []
        self._use_jina = use_jina
        self._use_tools = True  # Automatically determined and set to false if the model does not support tool calling
        self.driver: selenium.webdriver = None
        self.container: docker.Container = None
        self._terminal_initialized = False

    @abstractmethod
    def run_task(self, user_query: str, max_steps: int, sleep_seconds_between_steps: float):
        raise NotImplementedError

    @abstractmethod
    def _create_text_file(self, file_name: str, file_content: str):
        raise NotImplementedError

    @abstractmethod
    def _run_terminal_command(self, command):
        raise NotImplementedError

    @abstractmethod
    def _initialize_container(self):
        raise NotImplementedError

    @abstractmethod
    def _break_down_to_steps(self, user_query):
        raise NotImplementedError


class LLMAgent(AbstractLLMAgent):
    def __init__(
        self,
        docker_client: docker.DockerClient,
        openai_client: openai.OpenAI,
        planner_model_name: str,
        executor_model_name: str,
        prompt: PromptBase,
        use_jina: bool = False,
    ):
        super().__init__(
            docker_client=docker_client,
            openai_client=openai_client,
            planner_model_name=planner_model_name,
            executor_model_name=executor_model_name,
            prompt=prompt,
            use_jina=use_jina,
        )

    def run_task(
        self,
        user_query: str,
        max_steps: int,
        sleep_seconds_between_steps: float,
        requires_manual_approval: bool = True,
        summarize_intermediate_steps: bool = False,
    ):
        logging.info(f"Starting task: {user_query}")
        llm_step_by_step_plan: list[str] = self._break_down_to_steps(user_query)
        logging.debug(f"Plan: {llm_step_by_step_plan}")
        llm_step_by_step_plan = "\n".join(llm_step_by_step_plan)

        print(COLORS.CLEARSCREEN)
        print(f"# TASK:{os.linesep}{COLORS.GREEN}{user_query}{COLORS.DEFAULT}{os.linesep}")
        print("💭 LLM Plan 💭")
        print(llm_step_by_step_plan)

        if requires_manual_approval:
            input(f"{os.linesep}Press any key to continue...")
            print(os.linesep)

        i = 0
        while i < max_steps:
            logging.info(f"\n\n######### Step {i+1} #########")
            logging.debug("Choosing next step...")
            step_output = None
            next_step, tool_call, arguments = self._choose_next_step(
                user_query=user_query,
                llm_plan=llm_step_by_step_plan,
                steps_done=i,
                max_steps=max_steps,
                call_history=self.tool_call_history,
            )

            # Early stopping if the LLM marked the task as 'done'
            if next_step and next_step[:4].lower().startswith("done"):
                logging.debug("The LLM marked the task as 'done', preparing final output.")
                self.add_tool_call_to_history("done", i, "done", "done")
                break

            if tool_call and arguments:
                logging.info("[Tool Call Response]")
                # After function calling support in the API
                tool_name, tool_arguments = tool_call, arguments
            else:
                logging.info("[Message Response]")
                # Before function calling support in the API
                step_json = {}
                try:
                    json_start, json_end = (
                        next_step.index("{"),
                        next_step.rindex("}") + 1,
                    )
                    step_json_str = next_step[json_start:json_end]
                    step_json = json.loads(step_json_str)
                except (json.JSONDecodeError, ValueError):
                    msg = f"Invalid Call! Expected JSON with name and arguments. Got: '{next_step}'"
                    logging.warning(msg)

                tool_name = None
                tool_arguments = None
                try:
                    tool_name = step_json["name"]
                except KeyError:
                    msg = f"Invalid Tool Call! Expected JSON with name and arguments, but 'name' is missing. Got: '{next_step}'"
                    logging.warning(msg)

                try:
                    tool_arguments = step_json["arguments"]
                except KeyError:
                    msg = f"Invalid Tool Call! Expected JSON with name and arguments, but 'arguments' is missing. Got: '{next_step}'"
                    logging.warning(msg)

            # The LLM is repeating itself
            if self.tool_call_history:
                duplicate_calls = (
                    self.tool_call_history[-1]["tool_name"] == tool_name
                    and self.tool_call_history[-1]["tool_arguments"] == tool_arguments
                )
                if duplicate_calls:
                    i += 1
                    self.add_tool_call_to_history(
                        step_output="Repeated tool call! Use different arguments.",
                        i=i,
                        tool_name=tool_name,
                        tool_arguments=tool_arguments,
                    )
                    logging.warning(f"Repeated tool call - retrying step {i}...")
                    continue

            if tool_name and tool_arguments:
                logging.info(f"🛠️ Tool:      {COLORS.CYAN}{tool_name}{COLORS.DEFAULT}")
                logging.info(f"🔧 Arguments: {COLORS.CYAN}{tool_arguments}{COLORS.DEFAULT}")
                step_output = self.tool_call(tool_name, tool_arguments)
            else:
                logging.warning(f"No tool was invoked in step {i}")

            if not step_output:
                i += 1
                logging.warning(f"Retrying step {i}...")
                continue

            # Appending output to history
            if summarize_intermediate_steps:
                step_output = self._communicate_with_llm(
                    prompt=f"""
                                                     <tool_name>{tool_name}</tool_name>
                                                     <tool_arguments>{tool_arguments}</tool_arguments>
                                                     <tool_output>{step_output}</tool_output>
                                                     Summarize tool_output and keep the most important and relevant information that can help to solve the task: "{user_query}" in the next step.
                                                     Consice Markdown:
                                                     """,
                    #  max_tokens=256,
                    use_tools=False,
                )
                # Think step by step concisely.

            step_output = str(step_output)
            self.add_tool_call_to_history(step_output, i, tool_name, tool_arguments)

            # if summarize_intermediate_steps:
            is_done = self._communicate_with_llm(
                prompt=f"""
                                                    <tools_history>{self.tool_call_history}</tools_history>
                                                    <tool_output>{step_output}</tool_output>
                                                    Based on the tool_history and tool_output, was the TASK completed successfully?
                                                    TASK: "{user_query}."
                                                    Answer: yes OR no.
                                                    """,
                max_tokens=3,
                use_tools=False,
            )

            logging.info(f"done? {is_done}")
            if "yes" in is_done.lower():
                break

            i += 1
            logging.debug(f"Sleeping for {sleep_seconds_between_steps} seconds...")
            sleep(sleep_seconds_between_steps)
            logging.debug(f"######### Finished step {i} #########")

        # Prepare the final output using llm query
        final_str_prompt = self.prompts.prepare_final_output_prompt(
            user_query=user_query,
            tool_call_history=self.tool_call_history,
        )
        last_output = self._communicate_with_llm(final_str_prompt, use_tools=False)
        logging.info(f"Final Output:{os.linesep*2}{last_output}")
        print(f"{COLORS.GREEN}{os.linesep*2}{last_output}{COLORS.DEFAULT}")
        print("🤖 Thanks for trying LLM Agent 🤖")

    def tool_call(self, tool_name: str, tool_arguments: str) -> str:
        try:
            tool_name = tool_name.replace("*", "").replace('"', "").strip()
            match tool_name:
                # TODO: Use reflection and OOP classes for tools and to hard-coded tools names.
                case "create_text_file":
                    # cmd_input = tool_arguments
                    try:
                        # delimiter_idx = cmd_input.index(",")
                        # file_name = cmd_input[:delimiter_idx]
                        # content = cmd_input[delimiter_idx + 1 :]
                        file_name = tool_arguments["file_name"]
                        file_content = tool_arguments["file_content"]
                        step_output = self._create_text_file(str(file_name), str(file_content))
                    except Exception as e:
                        step_output = str(e)
                case "ubuntu_terminal":
                    step_output = self._run_terminal_command(**tool_arguments)
                case "web_browser":
                    if self._use_jina:
                        step_output = self._fetch_page_content_for_agent(**tool_arguments)
                    else:
                        if isinstance(tool_arguments, dict):
                            step_output = self._browse_page(**tool_arguments)
                        else:
                            step_output = self._browse_page(tool_arguments)
                case "llm_query":
                    llm_prompt_with_context = self.prompts.llm_query_tool_prompt(
                        tool_call_history=self.tool_call_history,
                        task=tool_arguments,
                    )
                    step_output = self._communicate_with_llm(llm_prompt_with_context)
                    logging.info(f"LLM Output: {COLORS.YELLOW}{step_output}{COLORS.DEFAULT}")
                case _:
                    step_output = f"Bad Tool Call! Expected one of: create_text_file, ubuntu_terminal, web_browser, llm_query. Got: {tool_name}"
                    logging.warning(f"{COLORS.RED}{step_output}{COLORS.DEFAULT}")
            return step_output
        except Exception as e:
            logging.fatal(f"Failed to execute tool: {e}")
            raise

    def add_tool_call_to_history(self, step_output, i, tool_name, tool_arguments):
        self.tool_call_history.append(
            {
                "step_number": i,
                "tool_name": tool_name,
                "tool_arguments": tool_arguments,
                "tool_output": step_output,
            }
        )

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

        target_path: str = f"{file_name.lstrip()}"
        subprocess.run(
            f'docker cp "{temp_file.name}" {self.container.id}:{target_path}',
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
        bash_command = f"""/bin/bash -c 'echo {base64_command} | base64 -d | bash' &"""

        try:
            cmd_output = self.container.exec_run(
                bash_command,
                stdin=True,
                workdir="/workdir",
            )
            if cmd_output.exit_code != 0:
                raise Exception(f"ERROR: {cmd_output}")

            logging.info(f"✅ TERMINAL OUTPUT: {COLORS.GREEN}{cmd_output}{COLORS.DEFAULT}")
        except Exception as e:
            cmd_output = str(e)
            logging.info(f"🔴 TERMINAL ERROR: {COLORS.YELLOW}{cmd_output}{COLORS.DEFAULT}")
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
        self.container
        self.container.exec_run("apt update")
        self.container.exec_run("apt-get install -y sudo net-tools")
        self._terminal_initialized = True

    def _break_down_to_steps(self, user_query):
        prompt = self.prompts.break_down_to_steps_prompt(user_query=user_query)

        # TODO: support models without tool calling
        llm_step_by_step_plan_text = self._communicate_with_llm(model_name=self._planner_model_name, prompt=prompt)
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
            res = requests.get(url, timeout=15)
            res.raise_for_status()
            res = res.text
            # print(res)
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
        links = [(anchor.get_text(strip=True), anchor.get("href")) for anchor in soup.find_all("a")]

        text = soup.get_text()

        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())

        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

        # Drop blank lines
        text = "\n".join(chunk for chunk in chunks if chunk)

        # Combine text with links for better context
        text_content_with_links = text + "\n".join(f"*{title}*: {link}" for title, link in links if link)
        logging.debug(text_content_with_links)
        logging.debug("Done browsing.")
        return text_content_with_links

    def _communicate_with_llm(
        self,
        prompt,
        max_tokens=1024,
        model_name=None,
        use_tools=True,
        stream=True,
        return_response_content=True,
    ):
        logging.debug(f"🤖 Communicating with LLM, prompt: {prompt}")
        model_name = model_name or self._executor_model_name
        if self._use_tools and use_tools:
            try:
                openai_response = self._openai_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    tools=self.prompts.available_tools,
                    max_completion_tokens=max_tokens,
                    stream=stream,
                )
                if return_response_content:
                    return self._process_stream_until_complete(openai_response, live_print_to_stdout=True)
                return openai_response
            except openai.BadRequestError as e:
                if "does not support tools" in str(e):
                    self._use_tools = False
                    logging.debug(
                        f"The model {model_name} does not support tools, so tool calling API will not be used."
                    )
                else:
                    raise

        openai_response = self._openai_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens,
            stream=stream,
        )
        if return_response_content:
            return self._process_stream_until_complete(openai_response, live_print_to_stdout=True)
        return openai_response

    def _choose_next_step(
        self,
        user_query,
        llm_plan,
        steps_done,
        max_steps,
        call_history,
    ):
        """
        Chooses the next step for the LLM agent based on the given arguments.

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
        prompt = self.prompts.next_step_prompt(
            user_query=user_query,
            llm_plan=llm_plan,
            call_history=call_history,
            steps_done=steps_done,
            max_steps=max_steps,
        )
        logging.debug(f"Executing prompt: {prompt}")
        next_step_response = self._communicate_with_llm(prompt=prompt, stream=False, return_response_content=False)
        tool_calls = next_step_response.choices[0].message.tool_calls
        if not tool_calls:
            message_content = next_step_response.choices[0].message.content
            logging.info(f"No tool call. Message: {message_content}; Tools:{tool_calls}")
            return message_content, None, None

        # Tool call
        next_step_str = next_step_response.choices[0].message.content
        tool_name = next_step_response.choices[0].message.tool_calls[0].function
        tool_name = tool_name.name
        tool_arguments = json.loads(next_step_response.choices[0].message.tool_calls[0].function.arguments)

        logging.info(f"📜 Tool History: {json.dumps(self.tool_call_history, indent=4)}")
        logging.info(f"🤖 Next Step:      {next_step_str}")
        logging.info(f"🛠️ Tool Call:      {tool_name}")
        logging.info(f"🛠️ Tool Arguments: {tool_arguments}")
        return next_step_str, tool_name, tool_arguments

    @staticmethod
    def _process_stream_until_complete(stream, live_print_to_stdout: bool = False) -> str:
        """
        A helper method to process the stream of responses from the OpenAI API.
        """
        if not isinstance(stream, openai.Stream):
            # Not a streaming response
            return stream.choices[0].message.content

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
    use_jina: bool,
    requires_manual_approval: bool = False,
    summarize_intermediate_steps: bool = False,
):
    # Set up Docker API client
    docker_client: docker.DockerClient = docker.DockerClient()
    if should_use_local_llm:
        openai_client: openai.OpenAI = openai.OpenAI(
            # Or, using ollama/llama.cpp server:
            base_url="http://localhost:11434/v1",
            api_key="test",
        )
    else:
        openai_client: openai.OpenAI = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    prompts: PromptBase = LlamaPrompts()
    agent = LLMAgent(
        docker_client=docker_client,
        openai_client=openai_client,
        use_jina=use_jina,
        prompt=prompts,
        planner_model_name=planner_model_name,
        executor_model_name=executor_model_name,
    )
    agent.run_task(
        user_query=user_query,
        max_steps=max_steps,
        sleep_seconds_between_steps=sleep_seconds_between_steps,
        requires_manual_approval=requires_manual_approval,
        summarize_intermediate_steps=summarize_intermediate_steps,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Agent")
    parser.add_argument(
        "--file",
        type=str,
        help="A file containing the user prompt for the LLM Agent.",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--query",
        type=str,
        help="User query for the LLM Agent",
        default=None,
        required=False,
    )
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
    parser.add_argument(
        "--use-jina",
        help="Instead of Selenium, use Jina AI free APIs to fetch page content. WARNING: Jina APIs might fail fetch certain websites.",
        action="store_true",
    )
    parser.add_argument(
        "--approve-plan",
        help="Manually approve the plan of the LLM before execution.",
        action="store_true",
    )
    parser.add_argument(
        "--summarize-steps",
        help="Summarize each step output instead of keeping it as-is. May benefit certain tasks.",
        action="store_true",
    )

    args = parser.parse_args()

    if not args.file and not args.query:
        raise ValueError("One of --file or --query must be specified.")
    if args.file and args.query:
        raise ValueError("ONLY ONE of --file or --query must be specified.")
    user_query = args.query
    if args.file:
        with open(args.file, "r") as file:
            user_query = file.read().strip()

    # Set OpenAI logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("openai").setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger("openai").setLevel(logging.WARNING)

    planner_model_name = args.planner
    executor_model_name = args.executor
    if args.local and (planner_model_name.startswith("gpt-") or executor_model_name.startswith("gpt-")):
        logging.error(
            f"""local backends doesn\'t support GPT models: planner="{planner_model_name}" \
executor="{executor_model_name}"; Use llama3.1 instead. """
        )
        sys.exit(1)

    max_steps = args.steps
    sleep_seconds_between_steps = args.sleep
    use_jina = args.use_jina
    approve_plan = args.approve_plan
    summarize_steps = args.summarize_steps
    main(
        user_query,
        args.local,
        planner_model_name,
        executor_model_name,
        max_steps,
        sleep_seconds_between_steps,
        use_jina=use_jina,
        requires_manual_approval=approve_plan,
        summarize_intermediate_steps=summarize_steps,
    )
