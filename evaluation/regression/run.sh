#!/bin/bash
set -eo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CASES_DIR=$SCRIPT_DIR/cases
AGENTHUB_DIR=$SCRIPT_DIR/../../agenthub
# Check if DEBUG variable is already set
if [ -z "${DEBUG}" ]; then
    read -p "Enter value for DEBUG (leave blank for default): " debug_value
    if [ -n "${debug_value}" ]; then
        export DEBUG="${debug_value}"
    else
        export DEBUG="0"
    fi
fi
# Check if OPENAI_API_KEY variable is already set
if [ -z "${OPENAI_API_KEY}" ]; then
    read -sp "Enter value for OPENAI_API_KEY: " openai_key
    echo
    export OPENAI_API_KEY="${openai_key}"
fi
# Get the  MODEL variable
read -sp "Enter value for model running agents: " model
echo

if [ -z "$model" ]; then
    MODEL="gpt-4-0125-preview"
else
    MODEL="$model"
fi

echo "Running with model: $MODEL" 

# add python path
export PYTHONPATH="$PYTHONPATH:$SCRIPT_DIR/../../" 

# hardcode pairs for directory to python class mapping 
declare -A directory_class_pairs=(
    [langchains_agent]="LangchainsAgent"
    [codeact_agent]="CodeActAgent"
    [planner_agent]="PlannerAgent"
)


# for each agent
for agent_dir in ./agenthub/*_agent; do
  agent=$(basename "$agent_dir")
  # iterate over cases dir
  if [[ -n $TEST_AGENT && $agent != $TEST_AGENT ]]; then
    continue
  fi
  for case in $(ls $CASES_DIR); do
    # run the case
    if [[ -n $TEST_CASE && $case != $TEST_CASE ]]; then
      continue
    fi
    echo "Running case: $case"
    case_dir=$CASES_DIR/$case
    task=$(cat $case_dir/task.txt)
    outputs_dir=$case_dir/outputs
    agent_dir=$outputs_dir/$agent
    echo "agent: $agent_dir"
    # create agent dir if not exist 
    if [ ! -d "$agent_dir" ]; then
       mkdir -p $agent_dir 
    fi
    rm -rf $agent_dir/workspace
    if [[ -d $case_dir/start ]]; then
      cp -r $case_dir/start $agent_dir/workspace
    else
      mkdir $agent_dir/workspace
    fi
    python3 $SCRIPT_DIR/../../opendevin/main.py -d $agent_dir/workspace -c ${directory_class_pairs[$agent]} -t "${task}" -m $MODEL  | tee $agent_dir/logs.txt
    rm -rf $agent_dir/workspace/.git
  done
done
