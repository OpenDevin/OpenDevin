#!/bin/bash

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
)

# Initialize counters for successful and failed test cases
success_count=0
fail_count=0

# for each agent directory
for agent_dir in $(find "$AGENTHUB_DIR" -type d -name '*agent'); do
    agent=$(basename "$agent_dir")

    # For each test case directory
    for case_dir in $CASES_DIR/*; do
        case=$(basename "$case_dir")

        echo "Running case: $case"
        task=$(<"$case_dir/task.txt")
        outputs_dir="$case_dir/outputs/$agent"
        echo "agent: $outputs_dir"

        # Create agent directory if it does not exist
        mkdir -p "$outputs_dir"

        # Remove existing workspace and create new one
        rm -rf "$outputs_dir/workspace"
        mkdir "$outputs_dir/workspace"

        # Copy start directory to workspace if it exists
        if [ -d "$case_dir/start" ]; then
            cp -r "$case_dir/start"/* "$outputs_dir/workspace"
        fi

        if [ -f "$case_dir/test.sh" ]; then
            # Run main.py and capture output
            if python3 "$SCRIPT_DIR/../../opendevin/main.py" -d "$outputs_dir/workspace" -c "${directory_class_pairs[$agent]}" -t "$task" -m "$MODEL" | tee "$outputs_dir/logs.txt"; then
                # If main.py succeeds, run test.sh
                if bash "$case_dir/test.sh" "$outputs_dir/workspace"; then
                    ((success_count++))
                else
                    ((fail_count++))
                fi
            else
                # If main.py fails, increment the fail count
                ((fail_count++))
            fi
        else
            # If main.py fails, increment the fail count
            ((fail_count++))
        fi
        # Remove .git directory from workspace
        rm -rf "$outputs_dir/workspace/.git"
    done
done

# Display test results
echo "Successful test cases: $success_count"
echo "Failed test cases: $fail_count"
