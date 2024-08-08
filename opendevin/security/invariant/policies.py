DEFAULT_INVARIANT_POLICY = """from invariant.detectors import semgrep, secrets, CodeIssue

raise "Disallow secrets in bash commands [risk=medium]" if:
    (call: ToolCall)
    call is tool:cmd_run
    any(secrets(call.function.arguments.command))

raise "Vulnerability in python code [risk=medium]" if:
    (call: ToolCall)
    call is tool:ipython_run_cell
    semgrep_res := semgrep(call.function.arguments.code, lang="python")
    any(semgrep_res)

raise "Vulnerability in bash command [risk=medium]" if:
    (call: ToolCall)
    call is tool:cmd_run
    semgrep_res := semgrep(call.function.arguments.command, lang="bash")
    any(semgrep_res)
"""