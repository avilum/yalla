# Yet Another LLM Agent (YaLLa)
A tiny LLM Agent with minimal dependencies, focused on local inference.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Tools](#tools)
- [Uses OpenAI API specification](#uses-openai-api-specification)
- [Get Started](#get-started)
- [Run the agent completely on device with Ollama](#run-the-agent-completely-on-device-with-ollama)
- [Run the agent using ChatGPT and OpenAI](#run-the-agent-using-chatgpt-and-openai)
- [Examples](#examples)
- [Execution logs and history](#execution-logs-and-history)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Tools

1. **ubuntu_terminal:** Runs bash commands inside Ubuntu temporary container via `docker`.
2. **web_browser:** Accesses websites or search free text, to retrieve the inner text content and links. Uses `selenium` by default. can also use Jina AI's free browsing APIs.
3. **llm_query:** Communicate with the large language model.
4. **create_text_file:** Creates a text file inside the container.
5. **open_api:** Interacts with an OpenAPI service to fetch data or perform operations.

## Uses OpenAI API specification
- Local
  - <a href="https://ollama.com/">Ollama</a>
  - <a href="https://github.com/ggerganov/llama.cpp/server">llama.cpp</a>
- OpenAI
  - ```shell
    export OPENAI_API_KEY="sk-..."
    ```

## Get Started
**Clone:**
```shell
git clone https://github.com/avilum/yalla.git && cd yalla
```
**Install:**
```shell
python3 -m pip install -r requirements.txt
```
**Run:**
```shell
# helper function
function agent(){
  python3.11 agent.py $*
}

agent --help
```

## Run the agent completely on device with Ollama
1. Run ollama
```shell
ollama pull gemma2 && ollama serve
```
2. Run the agent with `--local`
```shell
agent --local --planner gemma2 --executor gemma2 --query "What happened to donald trump?"
agent --local --planner gemma2 --executor phi3 --query "Who acquired Deci AI"
agent --local --verbose --query "Create a fastapi app and run it in on port 8082." --steps 7
agent --query 'open_api: {"url": "https://api.example.com/data", "method": "GET"}'
```
## Run the agent using ChatGPT and OpenAI
1. Set the OpenAI API Key:
```shell
export OPENAI_API_KEY="sk-..."
```
2. Run the agent
```shell
agent --query "Who acquired Deci AI"
agent --planner gpt-4o --executor gpt-3.5-turbo --verbose --query "What are the top trending models on huggingface from last week?"
```

## Examples
Input:
```shell
agent --local --planner gemma2 --executor gemma2 --query "Who acquired Deci AI?"
```
Output:
```text
Nvidia acquired Deci AI for $300 million.
```
---
Input:
```shell
agent --local --planner gemma2 --executor gemma2 --query "What happened to donald trump"
```
Output:
```text
Based on the provided information from CNN and The New York Times, former President Donald Trump was involved in a shooting at a rally in Butler, Pennsylvania on July 13, 2024.

He was reportedly rushed off stage with blood on his ear and face. While the Secret Service confirmed he is safe, details about the incident and the shooter are still emerging.

It's important to note that information about this event is rapidly developing and may change. To get the most up-to-date information, it's best to refer to reliable news sources.
```
---
Input:
```shell
agent --local --planner="gemma2" --executor="gemma2" --query "Tell me about Avi Lumelsky from Oligo and about his researches"
```
Output:
```text
Avi Lumelsky is a security researcher at Oligo.

He is actively involved in researching and raising awareness about vulnerabilities in popular AI frameworks, particularly the Ray AI framework.  His work has been featured in prominent tech publications like TechTarget, Security Boulevard, The Register, and VentureBeat.

Lumelsky's research has highlighted critical vulnerabilities that can be exploited to gain control of AI workloads, compute resources, and sensitive data. He is a recognized speaker at cybersecurity conferences like BSidesBUD, where he shares his findings and insights with the broader security community.


You can find more information about Avi Lumelsky and his work on his LinkedIn profile and in the articles listed in the provided context.
```
---
Input:
```shell
agent --local --planner gemma2 --executor gemma2 --query "What are the trending models on huggingface right now?"
```
Output:
```text
Here are some of the trending models on HuggingFace right now, across various categories like text generation, image generation, and question answering:

* **Text Generation:**
    * Qwen2-72B-Instruct
    * Mixtral-8x7B-Instruct-v0.1
    * Gemma-2-9b-it
    * Gemma-2-27b-it
* **Image Generation:**
    * Stable Diffusion XL-Base 1.0
    * Kolors
    * AuraFlow
    * Stable Diffusion 3-Medium
* **Other:**
    * Microsoft's Florence-2-large (Image-Text-to-Text)
    * OpenAI's Whisper-large-v3 (Automatic Speech Recognition)

Keep in mind that trends change rapidly!  You can explore the full list and sort by popularity on the HuggingFace website.
```
---
## Execution logs and history
By default all the runs will be logged to the `.runs/` directory.<br>

More examples can be found at the <a href="examples/">/examples</a> folder.
