# CodeAct-based Agent Framework

This folder implements the [CodeAct idea](https://arxiv.org/abs/2402.13463) that relies on LLM to autonomously perform actions in a Bash shell. It requires more from the LLM itself: LLM needs to be capable enough to do all the stuff autonomously, instead of stuck in an infinite loop. 

A minimalistic example can be found at [research/codeact/examples/run_flask_server_with_bash.py](./examples/run_flask_server_with_bash.py):

```bash
mkdir workspace
PYTHONPATH=`pwd`:$PYTHONPATH python3 opendevin/main.py -d ./workspace -c CodeActAgent -t "Please write a flask app that returns 'Hello, World\!' at the root URL, then start the app on port 5000. python3 has already been installed for you."
```


Example: prompts `gpt-4-0125-preview` to write a flask server, install `flask` library, and start the server.

<img width="951" alt="image" src="https://github.com/OpenDevin/OpenDevin/assets/38853559/325c3115-a343-4cc5-a92b-f1e5d552a077">

<img width="957" alt="image" src="https://github.com/OpenDevin/OpenDevin/assets/38853559/68ad10c1-744a-4e9d-bb29-0f163d665a0a">

Most of the things are working as expected, except at the end, the model did not follow the instruction to stop the interaction by outputting `<execute> exit </execute>` as instructed. 

**TODO**: This should be fixable by either (1) including a complete in-context example like [this](https://github.com/xingyaoww/mint-bench/blob/main/mint/tasks/in_context_examples/reasoning/with_tool.txt), OR (2) collect some interaction data like this and fine-tune a model (like [this](https://github.com/xingyaoww/code-act), a more complex route).
