
## Goals

- Short-term Goals: Focus on GPT-4 / Claude-3 or API-based projects.
- Demo Goal: Aim to replicate existing demos.
- Accuracy Goal: Work on the evaluation of the SWE-Bench (aiming for a 15% accuracy improvement) and Dev-Bench.
- Long-term Goal: Explore the potential for automatic coding/software engineering advancements.

## Milestones

- TODO

## Roadmap

1. Community operation
   1. Establish an open source committee and clarify the board members and management mechanism. This committee is mainly used for SOP design and operation.
   2. Set up maintainer(s) and join some developers into the github group. They can determine which code can be incorporated and push everything forward.
   3. Design and provide community operation SOP documents, determine entry and exit criteria, and determine the responsibilities of different roles.
2. Use Case and interaction
   1. Document learning: Supports specifying a URL or keyword to learn documents, use corresponding documents, and use tools.
   2. Repair based on issue: Given a github repo and issue, the Agent can issue a Pull Request for Review
   3. Confidential information mosaic: supports inquiry and mosaic of confidential information (such as passwords)
3. Effect: Exceeding devin’s score on SWE-bench
   1. The trial-and-error process, that is, the agent generates both patch and test-patch (for self-test), and then the agent debugs it until it succeeds, and then generates the real patch, which may improve the effect by ~2 times
4. Basic module
   1. Account system: Provides registration, login, retrieval, and logout functions for different accounts
   2. Data infra: Establish and determine the usage process of PostgreSQL, Redis, and S3/MinIO to provide multi-modal and multi-level data storage
   3. Data model: Use SQLAlchemy and pydantic to initialize the data model, including account, session, message, log, etc.
   4. Session: Support Session module to isolate different contexts
   5. Asynchronous interaction: Supports asynchronous interaction. All information can be supplied asynchronously without interrupting the work process.
5. Agent
   1. Environment & Sandbox
       1. Support docker: You need to ensure that Python, Shell, and browser can all install necessary dependencies and run under docker.  
           1. Note that it is recommended to pre-install a series of dependencies, such as chromium driver, to reduce the cost of repeated pulls.
       2. Support baremetal: Same as above
       3. Support vm: Same as above
   2. Memory
       1. Short-term memory: supports the storage and complete reading of all historical information within a time window
       2. Long-term memory: supports storage, retrieval, and weighted reordering of all historical information within a time window
       3. Program memory: supports symbolic storage and retrieval and weighted reordering of all historical information within a time window
       4. Auxiliary module: supports recording, filtering, weighting, and retrieval of any memory
   3. Tools
       1. Browser
           1. Selenium: Natural interactions can occur, such as link clicks and jumps, page rendering, etc.
           2. Playwright: Same as above
       2. Search
           1. SERPAPI: for fast structured interaction with different search engines
           2. SERPER: Same as above
           3. Duckduckgo: Fast structured interaction with ddg
           4. Google official API: fast structured interaction with google
       3. Command line support
       4. Code editor (such as VSCode)
       5. Text editor
   4. Evaluation
       1. Use Agent to access the basic evaluation process of SWE-bench
       2. Determine a clear SWE-bench development data set and unify everyone’s experimental standards
       3. Find the right sponsor, especially considering that this may cost tens of thousands of dollars
   5. LLM
       1. Support GPT-4
       2. Support GPT-4V
       3. Support ollama
       4. Support other APIs compatible with OpenAI
6. Documentation
   1. Establish an independent documentation site
   2. Determine the maintainers of the document site and related publicity to ensure that each meeting is properly publicized
