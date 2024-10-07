#!/bin/bash
# Usage: source agent.sh

function agent(){
  python3.11 /Users/avi/git/llm_agent/agent/agent.py $*
}
