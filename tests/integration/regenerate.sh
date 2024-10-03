#!/bin/bash
set -eo pipefail

##############################################################
##           CONSTANTS AND ENVIRONMENTAL VARIABLES          ##
##############################################################

echo -e "\n\n============================================================"

# unset environmental variables that might disturb testing
unset OPENAI_API_KEY
unset SANDBOX_ENV_OPENAI_API_KEY
unset OPENAI_BASE_URL
unset OPENAI_MODEL

# Get the absolute path of the script directory
get_script_dir() {
    local source="${BASH_SOURCE[0]}"
    while [ -h "$source" ]; do
        local dir="$( cd -P "$( dirname "$source" )" && pwd )"
        source="$(readlink -f "$source" 2>/dev/null || echo "$source")"
        [[ $source != /* ]] && source="$dir/$source"
    done
    echo "$( cd -P "$( dirname "$source" )" && pwd )"
}

TMP_FILE="${TMP_FILE:-tmp.log}"

if [ -z "$WORKSPACE_BASE" ]; then
  WORKSPACE_BASE=$(pwd)
fi

DEBUG=true  # needed for llm logging to create mock files!

if [ -z "$LOG_TO_FILE" ]; then
  LOG_TO_FILE=true
fi

export SCRIPT_DIR=$(get_script_dir)
export PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
export LOG_DIR="$PROJECT_ROOT/logs"
echo "Current working directory: $(pwd)"
echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "LOG_DIR: $LOG_DIR"
echo "LOG_TO_FILE: $LOG_TO_FILE"

WORKSPACE_BASE=${WORKSPACE_BASE}/_test_workspace
mkdir -p "$WORKSPACE_BASE"
chmod -R 777 "$WORKSPACE_BASE"
WORKSPACE_BASE=$(realpath "$WORKSPACE_BASE")

if [ -z "$WORKSPACE_MOUNT_PATH" ]; then
  WORKSPACE_MOUNT_PATH="$WORKSPACE_BASE"
else
  WORKSPACE_MOUNT_PATH="${WORKSPACE_MOUNT_PATH}/_test_workspace"
  mkdir -p "$WORKSPACE_MOUNT_PATH"
  chmod -R 755 "$WORKSPACE_MOUNT_PATH"
  WORKSPACE_MOUNT_PATH=$(realpath "$WORKSPACE_MOUNT_PATH")
fi

WORKSPACE_MOUNT_PATH_IN_SANDBOX="${WORKSPACE_MOUNT_PATH_IN_SANDBOX:-/workspace}"

echo "WORKSPACE_BASE: $WORKSPACE_BASE"
echo "WORKSPACE_MOUNT_PATH: $WORKSPACE_MOUNT_PATH"
echo "WORKSPACE_MOUNT_PATH_IN_SANDBOX: $WORKSPACE_MOUNT_PATH_IN_SANDBOX"

# Ensure we're in the correct directory
cd "$PROJECT_ROOT" || exit 1

mkdir -p "$WORKSPACE_BASE"

# use environmental variable if exists
TEST_RUNTIME="${TEST_RUNTIME:-eventstream}"
if [ -z "$SANDBOX_BASE_CONTAINER_IMAGE" ]; then
  SANDBOX_BASE_CONTAINER_IMAGE="nikolaik/python-nodejs:python3.11-nodejs22"
fi

MAX_ITERATIONS=20
echo "TEST_RUNTIME: $TEST_RUNTIME"

agents=(
  "DelegatorAgent"
  "ManagerAgent"
  "BrowsingAgent"
  "CodeActAgent"
  "PlannerAgent"
  "CodeActSWEAgent"
)
tasks=(
  "Fix typos in bad.txt."
  "Write a shell script 'hello.sh' that prints 'hello'."
  "Use Jupyter IPython to write a text file containing 'hello world' to '/workspace/test.txt'."
  "Write a git commit message for the current staging area."
  #"Install and import pymsgbox==1.0.9 and print it's version in /workspace/test.txt."
  "Browse localhost:8000, and tell me the ultimate answer to life."
)
test_names=(
  "test_edits"
  "test_write_simple_script"
  "test_ipython"
  "test_simple_task_rejection"
  #"test_ipython_module"  NOT DETERMINISTIC IN NUMBER OF LLM RESPONSES!
  "test_browse_internet"
)

num_of_tests=${#test_names[@]}
num_of_agents=${#agents[@]}

##############################################################
##                      FUNCTIONS                           ##
##############################################################

# run integration test against a specific agent & test
run_test() {
  # Ensure we're in the correct directory
  cd "$PROJECT_ROOT" || exit 1

  local pytest_cmd="poetry run pytest --cache-clear -vvsxx $SCRIPT_DIR/test_agent.py::$test_name"
  # Check if TEST_IN_CI is defined
  if [ -n "$TEST_IN_CI" ]; then
    pytest_cmd+=" --cov=agenthub --cov=openhands --cov-report=xml --cov-append"
  fi

  env SCRIPT_DIR="$SCRIPT_DIR" \
    PROJECT_ROOT="$PROJECT_ROOT" \
    WORKSPACE_BASE="$WORKSPACE_BASE" \
    WORKSPACE_MOUNT_PATH="$WORKSPACE_MOUNT_PATH" \
    MAX_ITERATIONS="$MAX_ITERATIONS" \
    DEFAULT_AGENT=$agent \
    TEST_RUNTIME="$TEST_RUNTIME" \
    DEBUG=$DEBUG \
    LLM=$LLM \
    LOG_TO_FILE=$LOG_TO_FILE \
    FORCE_REGENERATE=$FORCE_REGENERATE \
    SANDBOX_BASE_CONTAINER_IMAGE="$SANDBOX_BASE_CONTAINER_IMAGE" \
    $pytest_cmd 2>&1 | tee "$TMP_FILE"

  # Capture the exit code of pytest
  pytest_exit_code=${PIPESTATUS[0]}

  if grep -q "docker.errors.DockerException" $TMP_FILE; then
    echo "Error: docker.errors.DockerException found in the output. Exiting."
    echo "Please check if your Docker daemon is running!"
    exit 1
  fi

  if grep -q "tenacity.RetryError" $TMP_FILE; then
    echo "Error: tenacity.RetryError found in the output. Exiting."
    echo "This is mostly a transient error. Please retry."
    exit 1
  fi

  if grep -q "ExceptionPxssh" $TMP_FILE; then
    echo "Error: ExceptionPxssh found in the output. Exiting."
    echo "Could not connect to sandbox via ssh. Please stop any stale docker container and retry."
    exit 1
  fi

  if grep -q "Address already in use" $TMP_FILE; then
    echo "Error: Address already in use found in the output. Exiting."
    echo "Browsing tests need a local http server. Please check if there's any zombie process running start_http_server.py."
    exit 1
  fi

  # Return the exit code of pytest
  return $pytest_exit_code
}

# browsing capability needs a local http server
launch_http_server() {
  poetry run python $SCRIPT_DIR/start_http_server.py &
  HTTP_SERVER_PID=$!
  echo "Test http server launched, PID = $HTTP_SERVER_PID"
  sleep 5
}

cleanup() {
  cd "$PROJECT_ROOT/tests"
  cd "$PROJECT_ROOT"
  echo "Cleaning up before exit..."
  if [ -n "$HTTP_SERVER_PID" ]; then
    echo "Killing HTTP server..."
    kill $HTTP_SERVER_PID || true
    unset HTTP_SERVER_PID
  fi
  [ -f "$TMP_FILE" ] && rm "$TMP_FILE"
  echo "Cleanup done!"
}

# Trap the EXIT signal to run the cleanup function
if [ -z "$NOTRAP" ]; then
  trap cleanup EXIT
fi

# generate prompts again, using existing LLM responses under tests/integration/mock/[test_runtime]_runtime/[agent]/[test_name]/response_*.log
# this is a compromise; the prompts might be non-sense yet still pass the test, because we don't use a real LLM to
# respond to the prompts. The benefit is developers don't have to regenerate real responses from LLM, if they only
# apply a small change to prompts.
regenerate_without_llm() {
  cd "$PROJECT_ROOT"

  # set -x to print the command being executed
  set -x
  env SCRIPT_DIR="$SCRIPT_DIR" \
      PROJECT_ROOT="$PROJECT_ROOT" \
      WORKSPACE_BASE="$WORKSPACE_BASE" \
      WORKSPACE_MOUNT_PATH="$WORKSPACE_MOUNT_PATH" \
      MAX_ITERATIONS="$MAX_ITERATIONS" \
      FORCE_APPLY_PROMPTS=true \
      DEFAULT_AGENT="$agent" \
      TEST_RUNTIME="$TEST_RUNTIME" \
      LLM="$LLM" \
      DEBUG="$DEBUG" \
      LOG_TO_FILE="$LOG_TO_FILE" \
      FORCE_REGENERATE="$FORCE_REGENERATE" \
      SANDBOX_BASE_CONTAINER_IMAGE="$SANDBOX_BASE_CONTAINER_IMAGE" \
      poetry run pytest -s "$SCRIPT_DIR/test_agent.py::$test_name"
  set +x
}

regenerate_with_llm() {
  cd "$PROJECT_ROOT"

  rm -rf "$WORKSPACE_BASE/*"
  if [ -d "$SCRIPT_DIR/workspace/$test_name" ]; then
    cp -r "$SCRIPT_DIR/workspace/$test_name"/* "$WORKSPACE_BASE"
  fi

  rm -rf "$LOG_DIR"
  rm -rf "$SCRIPT_DIR/mock/${TEST_RUNTIME}_runtime/$agent/$test_name/*"
  # set -x to print the command being executed
  set -x
  echo -e "/exit\n" | \
    env SCRIPT_DIR="$SCRIPT_DIR" \
      PROJECT_ROOT="$PROJECT_ROOT" \
      WORKSPACE_BASE="$WORKSPACE_BASE" \
      WORKSPACE_MOUNT_PATH="$WORKSPACE_MOUNT_PATH" \
      DEFAULT_AGENT=$agent \
      RUNTIME="$TEST_RUNTIME" \
      SANDBOX_BASE_CONTAINER_IMAGE="$SANDBOX_BASE_CONTAINER_IMAGE" \
      LLM="$LLM" \
      DEBUG="$DEBUG" \
      LOG_TO_FILE="$LOG_TO_FILE" \
      FORCE_REGENERATE="$FORCE_REGENERATE" \
      poetry run python "$PROJECT_ROOT/openhands/core/main.py" \
      -i "$MAX_ITERATIONS" \
      -t "$task Do not ask me for confirmation at any point." \
      -c $agent
  set +x

  mkdir -p "$SCRIPT_DIR/mock/${TEST_RUNTIME}_runtime/$agent/$test_name/"
  mv "$LOG_DIR"/llm/**/* "$SCRIPT_DIR/mock/${TEST_RUNTIME}_runtime/$agent/$test_name/"
}

##############################################################
##                      MAIN PROGRAM                        ##
##############################################################

if [ "$num_of_tests" -ne "${#test_names[@]}" ]; then
  echo "Every task must correspond to one test case"
  exit 1
fi

rm -rf "$LOG_DIR"
rm -rf "$WORKSPACE_BASE/*"
for ((i = 0; i < num_of_tests; i++)); do
  task=${tasks[i]}
  test_name=${test_names[i]}

  # skip other tests if only one test is specified
  if [[ -n "$ONLY_TEST_NAME" && "$ONLY_TEST_NAME" != "$test_name" ]]; then
    continue
  fi

  if [ "$test_name" = "test_browse_internet" ]; then
    launch_http_server
  fi

  for ((j = 0; j < num_of_agents; j++)); do
    agent=${agents[j]}

    # skip other agents if only one agent is specified
    if [[ -n "$ONLY_TEST_AGENT" && "$ONLY_TEST_AGENT" != "$agent" ]]; then
      continue
    fi

    echo -e "\n============================================================"
    echo -e "======== STEP 1: Running $test_name for $agent"
    echo -e "============================================================\n\n"
    # reset dir so getcwd() shouldn't fail
    cd "$PROJECT_ROOT/tests"
    cd "$PROJECT_ROOT"

    rm -rf "$WORKSPACE_BASE/*"
    if [ -d "$SCRIPT_DIR/workspace/$test_name" ]; then
      cp -r "$SCRIPT_DIR/workspace/$test_name"/* "$WORKSPACE_BASE"
    fi

    if [ "$TEST_ONLY" ]; then
      set -e
    else
      # Temporarily disable 'exit on error'
      set +e
    fi

    TEST_STATUS=1
    if [ -z "$FORCE_REGENERATE" ]; then
      run_test
      TEST_STATUS=$?
    fi
    # Re-enable 'exit on error'
    set -e

    if [[ $TEST_STATUS -ne 0 ]]; then

      if [ "$FORCE_USE_LLM" ]; then
        echo -e "\n============================================================"
        echo -e "======== FORCE_USE_LLM, skipping step 2 & 3"
        echo -e "============================================================\n\n"
      elif [ ! -d "$SCRIPT_DIR/mock/${TEST_RUNTIME}_runtime/$agent/$test_name" ]; then
        echo -e "\n============================================================"
        echo -e "======== No existing mock responses for ${TEST_RUNTIME}_runtime/$agent/$test_name, skipping step 2 & 3"
        echo -e "============================================================\n\n"
      else
        echo -e "\n============================================================"
        echo -e "======== STEP 2: $test_name failed, regenerating prompts for $agent WITHOUT money cost"
        echo -e "============================================================\n\n"

        # Temporarily disable 'exit on error'
        set +e
        regenerate_without_llm

        echo -e "\n============================================================"
        echo -e "======== STEP 3: $test_name prompts regenerated for $agent, rerun test again to verify"
        echo -e "============================================================\n\n\n"
        run_test
        TEST_STATUS=$?
        # Re-enable 'exit on error'
        set -e
      fi

      if [[ $TEST_STATUS -ne 0 ]]; then
        echo -e "\n============================================================"
        if [ "$FORCE_USE_LLM" ]; then
          echo -e "======== STEP 4: $test_name REGENERATION for $agent WITH money cost"
        else
          echo -e "======== STEP 4: $test_name failed, regenerating prompts and responses for $agent WITH money cost"
        fi
        echo -e "============================================================\n\n\n"

        regenerate_with_llm

        echo -e "\n============================================================"
        echo -e "======== STEP 5: $test_name prompts and responses regenerated for $agent, rerun test again to verify"
        echo -e "============================================================\n\n\n"
        cd "$PROJECT_ROOT/tests"
        cd "$PROJECT_ROOT"
        # Temporarily disable 'exit on error'
        set +e
        run_test
        TEST_STATUS=$?
        # Re-enable 'exit on error'
        set -e

        if [[ $TEST_STATUS -ne 0 ]]; then
          echo -e "\n\n============================================================"
          echo -e "========== $test_name for $agent RERUN FAILED"
          echo -e "============================================================"
          echo -e "There are multiple possibilities:"
          echo -e "  1. The agent is unable to finish the task within $MAX_ITERATIONS steps."
          echo -e "  2. The agent thinks itself has finished the task, but fails the validation in the test code."
          echo -e "  3. There is something non-deterministic in the prompt."
          echo -e "  4. There is a bug in this script, or in OpenHands code."
          echo -e "NOTE: Some of the above problems could sometimes be fixed by a retry (with a more powerful LLM)."
          echo -e "      You could also consider improving the agent, increasing MAX_ITERATIONS, or skipping this test for this agent."
          echo -e "============================================================\n\n"
          exit 1
        else
          echo -e "\n\n============================================================"
          echo -e "========$test_name for $agent RERUN PASSED"
          echo -e "============================================================\n\n\n"
          sleep 1
        fi
      else
          echo -e "\n\n============================================================"
          echo -e "========$test_name for $agent RERUN PASSED"
          echo -e "============================================================\n\n\n"
          sleep 1
      fi
    else
      echo -e "\n\n============================================================"
      echo -e "\n========== $test_name for $agent PASSED"
      echo -e "\n============================================================\n\n\n"
      sleep 1
    fi
  done

  if [ "$test_name" = "test_browse_internet" ]; then
    kill $HTTP_SERVER_PID || true
  fi
done

rm -rf "$LOG_DIR"
rm -rf "$WORKSPACE_BASE"
echo "Done!"
cd "$PROJECT_ROOT"
