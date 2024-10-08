INSTRUCTION_FOR_AUTO_SEARCH = """
### Task:

You will be provided with a GitHub problem description. Your objective is to localize the specific files, classes, functions, or variable declarations that require modification or contain essential information to resolve the issue.

### Guidelines:

1. Localization Steps:
    - File Localization: Begin by identifying files likely related to the issue. Please provide the full path and return at most 5 files.
    - Function Localization: If necessary, narrow down to specific functions or classes within the identified files.
    - Fine-Grain Code Localization: Further refine your search to specific lines or code chunks as needed.
    
    Note: These steps can be performed in any order and iterated as needed.
    For example, after localizing relevant files and classes, you get other files contents/structures (or use retrieval-based method search method) or proceed to function localization for more detailed information.
    You can skip steps if the current findings are sufficient to address the issue.
    
2. Search Strategy:
    - Comprehensive Search: Utilize multiple search functions or the same function multiple times to gather extensive results.
    - File Limitation: You can search for as many files as necessary to obtain comprehensive results, but limit to no more than 10 files.
    - Repository Overview: Use `get_repo_structure` to gain an overview of the repository. This helps in independently searching for related files/functions or analyzing results from previous steps.
    - File Contents: Use `get_file_contents` to get access to contents of files relevant to solving the issue.
    - Code Context: Use `construct_topn_file_contexts` to get specific lines or chunks of code from related files, classes, and functions.
    - You can call retrival-based method(e.g., bm25) for reference, but be carefule to check their output.
    - You should NOT include any existing test case files.
    - Evaluate and prioritize findings from different search methods, focusing on the most relevant information.

3. Function Usage:
    - Sequential Function Calls: You are limited to calling one function at a time.
    - Important: 
        - Parameter Accuracy: Ensure accuracy in the parameters used for each tool call; avoid assumptions.

### Output Format:
Please provide the class name, function or method name, or the exact line number that need to be edited.
Return just the location(s) wrapped with ```
The output format should list each location (including file path, class name, function or method name, and line number) on separate lines. To clearly distinguish locations within different files, separate each file's location details with two newlines.

#### Examples:
```
full_path1/file1.py
line: 10
class: MyClass1
line: 51

full_path2/file2.py
function: MyClass2.my_method
line: 12

full_path3/file3.py
function: my_function
line: 24
line: 156
```

Important: Ensure that your final results are ordered by importance, from most to least critical, to effectively address the problem.
"""

INSTRUCTION_FOR_AUTO_SEARCH_V2 = """
### Task:

You will be provided with a GitHub problem description. Your objective is to localize the specific files, classes, functions, or variable declarations that require modification or contain essential information to resolve the issue.

### Guidelines:

1. Localization Process:
    - File Localization: Identify up to 10 files directly related to the issue, providing their full paths.
    - Function/Code Localization: Narrow down the search to specific functions, classes, or lines of code within the identified files.
    - Dependency Analysis: Gather information about the dependencies of the localized code to understand its context and find additional relevant details if needed.
    - Evaluation: Continuously evaluate whether your findings are sufficient to address the issue. Avoid extraneous details, and rerank and summarize your results as needed.
    - Iterative Search: If your initial findings are insufficient, continue the search by exploring other relevant files, classes, or functions.
    
2. Search Strategy:
    - Comprehensive Search: Use retrival-based method(e.g., bm25) for reference, but be carefule to evaluate their results. Consider multiple search strategies to ensure thorough coverage.
    - Repository Overview: Use `get_repo_structure` to gain an overview of the repository. This helps in independently searching for related files or analyzing results from previous steps.
    - File and Code Access: Use `get_file_contents` to access the contents of files relevant to the issue. Utilize `construct_topn_file_contexts` to retrieve specific code lines or chunks, and `construct_code_graph_context` for function dependency analysis.
    - File Limitations: You may search as many files as necessary, but limit your results to no more than 10 files.
    - Exclusions: Do not include any existing test case files in your findings.
    - Evaluate and prioritize findings from different search methods, focusing on the most relevant information.

3. Function Usage:
    - Sequential Function Calls: You are limited to calling one function at a time.
    - Parameter Accuracy: Ensure that the parameters used for each tool call are accurate, avoiding assumptions.
    - Relevance: Focus on identifying the most relevant files, classes, and functions without including unnecessary details.

### Output Format:
Your final output should list the locations requiring modification, wrapped with triple backticks ```
Each location should include the file path, class name (if applicable), function or method name, and line numbers, ordered by importance.

#### Examples:
```
full_path1/file1.py
line: 10
class: MyClass1
line: 51

full_path2/file2.py
function: MyClass2.my_method
line: 12

full_path3/file3.py
function: my_function
line: 24
line: 156
```

Return just the location(s)
"""


SUMMARY_INSTRUCTION = """
Please provide the class name, function or method name, or the exact line numbers that need to be edited.
### Examples:
```
full_path1/file1.py
line: 10
class: MyClass1
line: 51

full_path2/file2.py
function: MyClass2.my_method
line: 12

full_path3/file3.py
function: my_function
line: 24
line: 156
```

Return just the location(s). 
Ensure that your final results are ordered by importance, from most to least critical, to effectively address the problem.
"""

INVALID_FILE_PATH_WARN = 'Some files have invalid paths and have been filtered out. \
    The actual argument for this function is in `actual_args`. \
    Please provide the full file paths and ensure their accuracy and facticity next time.'

INVALID_FILE_PATH_WARN_v2 = 'The given input files have invalid paths and have been filtered out. \
    Please provide the full file paths and ensure their accuracy and facticity next time. \
    You should NOT include any existing test case files.'