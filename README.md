<p align="center">
  <img alt="OpenDevin Logo" src="./logo.png" width="150" />
</p>

# OpenDevin: Code Less, Make More

![License](https://img.shields.io/badge/license-MIT-green)

[demo-video.webm](https://github.com/OpenDevin/OpenDevin/assets/38853559/5b1092cc-3554-4357-a279-c2a2e9b352ad)


## Mission 🎯
Welcome to OpenDevin, an open-source project aiming to replicate [Devin](https://www.cognition-labs.com/introducing-devin), an autonomous AI software engineer who is capable of executing complex engineering tasks and collaborating actively with users on software development projects. This project aspires to replicate, enhance, and innovate upon Devin through the power of the open-source community.

## Work in Progress

OpenDevin is still a work in progress. But you can run the alpha version to see things working end-to-end.

### Requirements
* [Docker](https://docs.docker.com/engine/install/)
* [Python](https://www.python.org/downloads/) >= 3.10
* [NodeJS](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm) >= 14.8

### Installation
First, make sure Docker is running:
```bash
docker ps # this should exit successfully
```
Then pull our latest image [here](https://github.com/opendevin/OpenDevin/pkgs/container/sandbox)
```bash
docker pull ghcr.io/opendevin/sandbox:v0.1
```

Then start the backend:
```bash
export OPENAI_API_KEY="..."
export WORKSPACE_DIR="/path/to/your/project"
python -m pip install -r requirements.txt
uvicorn opendevin.server.listen:app --port 3000
```
Then in a second terminal:
```bash
cd frontend
npm install
npm run start -- --port 3001
```

You'll see OpenDevin running at localhost:3001

### Picking a Model
We use LiteLLM, so you can run OpenDevin with any foundation model, including OpenAI, Claude, and Gemini.
LiteLLM has a [full list of providers](https://docs.litellm.ai/docs/providers).

To change the model, set the `LLM_MODEL` and `LLM_API_KEY` environment variables.

For example, to run Claude:
```bash
export LLM_API_KEY="your-api-key"
export LLM_MODEL="claude-3-opus-20240229"
```

You can also set the base URL for local/custom models:
```bash
export LLM_BASE_URL="https://localhost:3000"
```

And you can customize which embeddings are used for the vector database storage:
```bash
export LLM_EMBEDDING_MODEL="llama2" # can be "llama2", "openai", "azureopenai", or "local"
```

### Running on the Command Line
You can run OpenDevin from your command line:
```bash
PYTHONPATH=`pwd` python opendevin/main.py -d ./workspace/ -i 100 -t "Write a bash script that prints 'hello world'"
```

## 🤔 What is [Devin](https://www.cognition-labs.com/introducing-devin)?

Devin represents a cutting-edge autonomous agent designed to navigate the complexities of software engineering. It leverages a combination of tools such as a shell, code editor, and web browser, showcasing the untapped potential of LLMs in software development. Our goal is to explore and expand upon Devin's capabilities, identifying both its strengths and areas for improvement, to guide the progress of open code models.

## 🐚 Why OpenDevin?

The OpenDevin project is born out of a desire to replicate, enhance, and innovate beyond the original Devin model. By engaging the open-source community, we aim to tackle the challenges faced by Code LLMs in practical scenarios, producing works that significantly contribute to the community and pave the way for future advancements.

## ⭐️ Research Strategy

Achieving full replication of production-grade applications with LLMs is a complex endeavor. Our strategy involves:

1. **Core Technical Research:** Focusing on foundational research to understand and improve the technical aspects of code generation and handling.
2. **Specialist Abilities:** Enhancing the effectiveness of core components through data curation, training methods, and more.
3. **Task Planning:** Developing capabilities for bug detection, codebase management, and optimization.
4. **Evaluation:** Establishing comprehensive evaluation metrics to better understand and improve our models.


## 🛠 Technology Stack

- **Sandboxing Environment:** Ensuring safe execution of code using technologies like Docker and Kubernetes.
- **Frontend Interface:** Developing user-friendly interfaces for monitoring progress and interacting with Devin, potentially leveraging frameworks like React or creating a VSCode plugin for a more integrated experience.

## 🚀 Next Steps

An MVP demo is urgent for us. Here are the most important things to do:

- UI: a chat interface, a shell demonstrating commands, a browser, etc.
- Architecture: an agent framework with a stable backend, which can read, write and run simple commands
- Agent: capable of generating bash scripts, running tests, etc.
- Evaluation: a minimal evaluation pipeline that is consistent with Devin's evaluation.

After finishing building the MVP, we will move towards research in different topics, including foundation models, specialist capabilities, evaluation, agent studies, etc.

## Contributors
Thanks goes to these wonderful contributors ([emoji key](https://allcontributors.org/docs/en/emoji-key) following [all-contributors](https://github.com/all-contributors/all-contributors) specification):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<table>
  <tbody>
    <tr>
      <td align="center"><a href="https://github.com/rbren"><img src="https://avatars.githubusercontent.com/rbren?v=4&s=100" width="100px;" alt="rbren"/><br /><sub><b>rbren</b></sub></a><br />
      <a href="https://github.com/OpenDevin/OpenDevin/commits?author=rbren" title="Code">💻</a>
      <a href="https://github.com/OpenDevin/OpenDevin/commits?author=rbren" title="Documentation">📖</a>
      <a href="#infra-rbren" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a>
      <a href="#maintenance-rbren" title="Maintenance">🚧</a>
      <a href="#ideas-rbren" title="Ideas & Planning">🤔</a>
      <a href="#review-rbren" title="Reviewed Pull Requests">👀</a>
      <a href="#tool-rbren" title="Tools">🔧</a>
      <a href="#test-rbren" title="Tests">⚠️</a>
      <a href="#bug-rbren" title="Bug reports">🐛</a></td>
      <td align="center"><a href="https://github.com/xingyaoww"><img src="https://avatars.githubusercontent.com/xingyaoww?v=4&s=100" width="100px;" alt="xingyaoww"/><br /><sub><b>xingyaoww</b></sub></a><br />
      <a href="https://github.com/OpenDevin/OpenDevin/commits?author=xingyaoww" title="Code">💻</a>
      <a href="https://github.com/OpenDevin/OpenDevin/commits?author=xingyaoww" title="Documentation">📖</a>
      <a href="#design-xingyaoww" title="Design">🎨</a>
      <a href="#ideas-xingyaoww" title="Ideas & Planning">🤔</a>
      <a href="#projectManagement-xingyaoww" title="Project Management">📆</a>
      <a href="#research-xingyaoww" title="Research">🔬</a></td>
      <td align="center"><a href="https://github.com/yimothysu"><img src="https://avatars.githubusercontent.com/yimothysu?v=4&s=100" width="100px;" alt="yimothysu"/><br /><sub><b>yimothysu</b></sub></a><br />
      <a href="https://github.com/OpenDevin/OpenDevin/commits?author=yimothysu" title="Code">💻</a>
      <a href="#design-yimothysu" title="Design">🎨</a>
      <a href="#infra-yimothysu" title="Infrastructure">🚇</a>
      <a href="#doc-yimothysu" title="Documentation">📖</a>
      <a href="#tool-yimothysu" title="Tools">🔧</a></td>
      <td align="center"><a href="https://github.com/huybery"><img src="https://avatars.githubusercontent.com/huybery?v=4&s=100" width="100px;" alt="huybery"/><br /><sub><b>huybery</b></sub></a><br />
      <a href="https://github.com/OpenDevin/OpenDevin/commits?author=huybery" title="Code">💻</a>
      <a href="#design-huybery" title="Design">🎨</a>
      <a href="#doc-huybery" title="Documentation">📖</a></td>
      <td align="center"><a href="https://github.com/yufansong"><img src="https://avatars.githubusercontent.com/yufansong?v=4&s=100" width="100px;" alt="yufansong"/><br /><sub><b>yufansong</b></sub></a><br />
      <a href="https://github.com/OpenDevin/OpenDevin/commits?author=yufansong" title="Code">💻</a>
      <a href="#doc-yufansong" title="Documentation">📖</a>
      <a href="#research-yufansong" title="Research">🔬</a></td>
    </tr>
    <tr>
      <td align="center"><a href="https://github.com/iFurySt"><img src="https://avatars.githubusercontent.com/iFurySt?v=4&s=100" width="100px;" alt="iFurySt"/><br /><sub><b>iFurySt</b></sub></a><br />
      <a href="https://github.com/OpenDevin/OpenDevin/commits?author=iFurySt" title="Code">💻</a>
      <a href="#infra-iFurySt" title="Infrastructure">🚇</a>
      <a href="#security-iFurySt" title="Security">🛡️</a></td>
      <td align="center"><a href="https://github.com/JustinLin610"><img src="https://avatars.githubusercontent.com/JustinLin610?v=4&s=100" width="100px;" alt="JustinLin610"/><br /><sub><b>JustinLin610</b></sub></a><br />
      <a href="#doc-JustinLin610" title="Documentation">📖</a>
      <a href="#projectManagement-JustinLin610" title="Project Management">📆</a></td>
      <td align="center"><a href="https://github.com/geohotstan"><img src="https://avatars.githubusercontent.com/geohotstan?v=4&s=100" width="100px;" alt="geohotstan"/><br /><sub><b>geohotstan</b></sub></a><br />
      <a href="https://github.com/OpenDevin/OpenDevin/commits?author=geohotstan" title="Code">💻</a>
      <a href="#tool-geohotstan" title="Tools">🔧</a>
      <a href="#test-geohotstan" title="Tests">⚠️</a></td>
      <td align="center"><a href="https://github.com/jojeyh"><img src="https://avatars.githubusercontent.com/jojeyh?v=4&s=100" width="100px;" alt="jojeyh"/><br /><sub><b>jojeyh</b></sub></a><br />
      <a href="https://github.com/OpenDevin/OpenDevin/commits?author=jojeyh" title="Code">💻</a>
      <a href="#bug-jojeyh" title="Bug reports">🐛</a>
      <a href="#infra-jojeyh" title="Infrastructure">🚇</a></td>
      <td align="center"><a href="https://github.com/zch-cc"><img src="https://avatars.githubusercontent.com/zch-cc?v=4&s=100" width="100px;" alt="zch-cc"/><br /><sub><b>zch-cc</b></sub></a><br />
      <a href="https://github.com/OpenDevin/OpenDevin/commits?author=zch-cc" title="Code">💻</a>
      <a href="#test-zch-cc" title="Tests">⚠️</a>
      <a href="#doc-zch-cc" title="Documentation">📖</a></td>
    </tr>
    <tr>
      <td align="center"><a href="https://github.com/powerzbt"><img src="https://avatars.githubusercontent.com/powerzbt?v=4&s=100" width="100px;" alt="powerzbt"/><br /><sub><b>powerzbt</b></sub></a><br />
      <a href="#code-powerzbt" title="Code">💻</a>
      <a href="#promotion-powerzbt" title="Promotion">📣</a></td>
      <td align="center"><a href="https://github.com/libowen2121"><img src="https://avatars.githubusercontent.com/libowen2121?v=4&s=100" width="100px;" alt="libowen2121"/><br /><sub><b>libowen2121</b></sub></a><br />
      <a href="#research-libowen2121" title="Research">🔬</a>
      <a href="#doc-libowen2121" title="Documentation">📖</a></td>
      <td align="center"><a href="https://github.com/Jiaxin-Pei"><img src="https://avatars.githubusercontent.com/Jiaxin-Pei?v=4&s=100" width="100px;" alt="Jiaxin-Pei"/><br /><sub><b>Jiaxin-Pei</b></sub></a><br />
      <a href="https://github.com/OpenDevin/OpenDevin/commits?author=Jiaxin-Pei" title="Code">💻</a>
      <a href="#data-Jiaxin-Pei" title="Data">🔣</a></td>
      <td align="center"><a href="https://github.com/eltociear"><img src="https://avatars.githubusercontent.com/eltociear?v=4&s=100" width="100px;" alt="eltociear"/><br /><sub><b>eltociear</b></sub></a><br />
      <a href="https://github.com/OpenDevin/OpenDevin/commits?author=eltociear" title="Documentation">📖</a>
      <a href="#design-eltociear" title="Design">🎨</a></td>
      <td align="center"><a href="https://github.com/Ghat0tkach"><img src="https://avatars.githubusercontent.com/Ghat0tkach?v=4&s=100" width="100px;" alt="Ghat0tkach"/><br /><sub><b>Ghat0tkach</b></sub></a><br />
      <a href="https://github.com/OpenDevin/OpenDevin/commits?author=Ghat0tkach" title="Code">💻</a>
      <a href="#design-Ghat0tkach" title="Design">🎨</a></td>
    </tr>
    <tr>
      <td align="center"><a href="https://github.com/suryavirkapur"><img src="https://avatars.githubusercontent.com/suryavirkapur?v=4&s=100" width="100px;" alt="suryavirkapur"/><br /><sub><b>suryavirkapur</b></sub></a><br />
      <a href="https://github.com/OpenDevin/OpenDevin/commits?author=suryavirkapur" title="Code">💻</a>
      <a href="#infra-suryavirkapur" title="Infrastructure">🚇</a></td>
      <td align="center"><a href="https://github.com/neubig"><img src="https://avatars.githubusercontent.com/neubig?v=4&s=100" width="100px;" alt="neubig"/><br /><sub><b>neubig</b></sub></a><br />
      <a href="https://github.com/OpenDevin/OpenDevin/commits?author=neubig" title="Code">💻</a>
      <a href="#ideas-neubig" title="Ideas & Planning">🤔</a></td>
      <td align="center"><a href="https://github.com/asadm"><img src="https://avatars.githubusercontent.com/asadm?v=4&s=100" width="100px;" alt="asadm"/><br /><sub><b>asadm</b></sub></a><br />
      <a href="https://github.com/OpenDevin/OpenDevin/commits?author=asadm" title="Business Development">💼</a>
      <a href="#legal-asadm" title="Legal">⚖️</a></td>
      <td align="center"><a href="https://github.com/xiangyue9607"><img src="https://avatars.githubusercontent.com/xiangyue9607?v=4&s=100" width="100px;" alt="xiangyue9607"/><br /><sub><b>xiangyue9607</b></sub></a><br />
      <a href="https://github.com/OpenDevin/OpenDevin/commits?author=xiangyue9607" title="Code">💻</a>
      <a href="#projectManagement-xiangyue9607" title="Project Management">📆</a></td>
      <td align="center"><a href="https://github.com/sikgyu"><img src="https://avatars.githubusercontent.com/sikgyu?v=4&s=100" width="100px;" alt="sikgyu"/><br /><sub><b>sikgyu</b></sub></a><br />
      <a href="https://github.com/OpenDevin/OpenDevin/commits?author=sikgyu" title="Code">💻</a>
      <a href="#bug-sikgyu" title="Bug reports">🐛</a></td>
    </tr>
    <tr>
      <td align="center"><a href="https://github.com/penberg"><img src="https://avatars.githubusercontent.com/penberg?v=4&s=100" width="100px;" alt="penberg"/><br /><sub><b>penberg</b></sub></a><br />
      <a href="https://github.com/OpenDevin/OpenDevin/commits?author=penberg" title="Code">💻</a>
      <a href="#infra-penberg" title="Infrastructure">🚇</a></td>
      <td align="center"><a href="https://github.com/enyst"><img src="https://avatars.githubusercontent.com/enyst?v=4&s=100" width="100px;" alt="enyst"/><br /><sub><b>enyst</b></sub></a><br />
      <a href="#ideas-enyst" title="Ideas & Planning">🤔</a>
      <a href="#research-enyst" title="Research">🔬</a></td>
      <td align="center"><a href="https://github.com/RohitX0X"><img src="https://avatars.githubusercontent.com/RohitX0X?v=4&s=100" width="100px;" alt="RohitX0X"/><br /><sub><b>RohitX0X</b></sub></a><br />
      <a href="#code-RohitX0X" title="Code">💻</a>
      <a href="#bug-RohitX0X" title="Bug reports">🐛</a></td>
      <td align="center"><a href="https://github.com/dincengincan"><img src="https://avatars.githubusercontent.com/dincengincan?v=4&s=100" width="100px;" alt="dincengincan"/><br /><sub><b>dincengincan</b></sub></a><br />
      <a href="#design-dincengincan" title="Design">🎨</a>
      <a href="#doc-dincengincan" title="Documentation">📖</a></td>
      <td align="center"><a href="https://github.com/Aadya1603"><img src="https://avatars.githubusercontent.com/Aadya1603?v=4&s=100" width="100px;" alt="Aadya1603"/><br /><sub><b>Aadya1603</b></sub></a><br />
      <a href="#code-Aadya1603" title="Code">💻</a>
      <a href="#infra-Aadya1603" title="Infrastructure">🚇</a></td>
    </tr>
    <tr>
      <td align="center"><a href="https://github.com/h9-tect"><img src="https://avatars.githubusercontent.com/h9-tect?v=4&s=100" width="100px;" alt="h9-tect"/><br /><sub><b>h9-tect</b></sub></a><br />
      <a href="https://github.com/OpenDevin/OpenDevin/commits?author=h9-tect" title="Code">💻</a>
      <a href="#ideas-h9-tect" title="Ideas & Planning">🤔</a></td>
      <td align="center"><a href="https://github.com/Rocchegiacomo"><img src="https://avatars.githubusercontent.com/Rocchegiacomo?v=4&s=100" width="100px;" alt="Rocchegiacomo"/><br /><sub><b>Rocchegiacomo</b></sub></a><br />
      <a href="#bug-Rocchegiacomo" title="Bug reports">🐛</a>
      <a href="#maintenance-Rocchegiacomo" title="Maintenance">🚧</a></td>

    </tr>
  </tbody>
</table>


<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->


## How to Contribute

OpenDevin is a community-driven project, and we welcome contributions from everyone. Whether you're a developer, a researcher, or simply enthusiastic about advancing the field of software engineering with AI, there are many ways to get involved:

- **Code Contributions:** Help us develop the core functionalities, frontend interface, or sandboxing solutions.
- **Research and Evaluation:** Contribute to our understanding of LLMs in software engineering, participate in evaluating the models, or suggest improvements.
- **Feedback and Testing:** Use the OpenDevin toolset, report bugs, suggest features, or provide feedback on usability.

For details, please check [this document](./CONTRIBUTING.md).

## Join Us
We use Slack to discuss. Feel free to fill in the [form](https://forms.gle/758d5p6Ve8r2nxxq6) if you would like to join the Slack organization of OpenDevin. We will reach out shortly if we feel you are a good fit to the current team! 

Stay updated on OpenDevin's progress, share your ideas, and collaborate with fellow enthusiasts and experts. Together, we can make significant strides towards simplifying software engineering tasks and creating more efficient, powerful tools for developers everywhere.

🐚 **Code less, make more with OpenDevin.**

[![Star History Chart](https://api.star-history.com/svg?repos=OpenDevin/OpenDevin&type=Date)](https://star-history.com/#OpenDevin/OpenDevin&Date)
