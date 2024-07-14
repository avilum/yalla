# Yet Another LLM Agent
A tiny LLM Agent with minimal dependencies, focused on local inference.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Tools](#tools)
- [Backends](#backends)
- [Get Started](#get-started)
- [Example Runs](#example-runs)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Tools

1. **ubuntu_terminal:** Runs bash commands inside Ubuntu temporary container via `docker`.
2. **web_browser:** Accesses websites or search free text, to retrieve the inner text content and links. Uses `selenium` by default, can also use Jina AI's free browsing APIs.
3. **llm_query:** Communicate with the large language model.
4. **create_text_file:** Creates a text file inside the container.

## Backends
- Local (`Ollama` / `llama.cpp`).
- OpenAI-like API

## Get Started
**Clone:**
```shell
git clone https://github.com/avilum/agent.git && cd agent
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

# Local backend 
# ollama pull gemma2 && ollama serve
agent --local --planner gemma2 --executor gemma2 --query "What happened to donald trump?"
agent --local --planner gemma2 --executor phi3 --query "Who acquired Deci AI"
agent --local --verbose --steps 7 --query "Create a fastapi app and run it in on port 8082."

# Using OpenAI
# export OPENAI_API_KEY="sk-..."
agent --query "Who acquired Deci AI"
agent --planner gpt-4o --executor gpt-3.5-turbo --verbose --query "What are the top trending models on huggingface from last week?"
```

## Example Runs
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
A shooting occurred at a Donald Trump campaign rally in Pennsylvania on July 14, 2024.  The details of the incident and the perpetrator's motives are still unfolding.  Major news outlets like CNN, USA Today, AP News, and BBC are providing updates on the situation. To get the most current information, I recommend checking those sources directly. 
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
agent --planner gpt-4o --executor gpt-3.5-turbo --verbose --query "What are the top trending models names on huggingface trending page?" --sleep 1
```
Output:
```text
The top trending model names on the Hugging Face trending page include controlnet-union-sdxl-1.0, Kolors, AuraFlow, stable-diffusion-3-medium, NuminaMath-7B-TIR, and many others.
```
---
