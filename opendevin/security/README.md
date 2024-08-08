# Security

Given the impressive capabilities of OpenDevin and similar coding agents, ensuring robust security measures is essential to prevent unintended actions or security breaches. The SecurityAnalyzer framework provides a structured approach to monitor and analyze agent actions for potential security risks.

To enable this feature:
* From the web interface
    * Open Configuration (by clicking the gear icon in the bottom right)
    * Select a Security Analyzer from the dropdown
    * Save settings
    * (to disable) repeat the same steps, but click the X in the Security Analyzer dropdown
* From config.toml
```toml
[security]
# Enable confirmation mode
confirmation_mode = true
# The security analyzer to use
security_analyzer = "your-security-analyzer"
```
(to disable) remove the lines from config.toml

## SecurityAnalyzer Base Class

The `SecurityAnalyzer` class (analyzer.py) is an abstract base class designed to listen to an event stream and analyze actions for security risks and eventually act before the action is executed. Below is a detailed explanation of its components and methods:

### Initialization

- **event_stream**: An instance of `EventStream` that the analyzer will listen to for events.

### Event Handling

- **on_event(event: Event)**: Handles incoming events. If the event is an `Action`, it evaluates its security risk and acts upon it.

### Abstract Methods

- **handle_api_request(request: Request)**: Abstract method to handle API requests.
- **log_event(event: Event)**: Logs events.
- **act(event: Event)**: Defines actions to take based on the analyzed event.
- **security_risk(event: Action)**: Evaluates the security risk of an action and returns the risk level.
- **close()**: Cleanups resources used by the security analyzer.

In conclusion, a concrete security analyzer should evaluate the risk of each event and act accordingly (e.g. auto-confirm, send Slack message, etc).

For customization and decoupling from the OpenDevin core logic, the security analyzer can define its own API endpoints that can then be accessed from the frontend. These API endpoints need to be secured (do not allow more capabilities than the core logic
provides).

## How to implement your own Security Analyzer

1. Create a submodule in [security](/opendevin/security/) with your analyzer's desired name
    * Have your main class inherit from [SecurityAnalyzer](/opendevin/security/analyzer.py)
    * Optional: define API endpoints for `/api/security/{path:path}` to manage settings,
2. Add your analyzer class to the [options](/opendevin/security/options.py) to have it be visible from the frontend combobox
3. Optional: implement your modal frontend (for when you click on the lock) in [security](/frontend/src/components/modals/security/) and add your component to [Security.tsx](/frontend/src/components/modals/security/Security.tsx)

## Implemented Security Analyzers

### Invariant

It uses the [Invariant Analyzer](https://github.com/invariantlabs-ai/invariant) to analyze traces and detect potential issues with OpenDevin's workflow. It uses confirmation mode to ask for user confirmation on potentially risky actions.

This allows the agent to run autonomously without fear that it will inadvertently compromise security or perform unintended actions that could be harmful.

Features:

* Detects:
    * potential secret leaks by the agent
    * security issues in Python code
    * malicious bash commands
* Logs:
    * actions and their associated risk
    * OpenDevin traces in JSON format
* Run-time settings:
    * the [invariant policy](https://github.com/invariantlabs-ai/invariant?tab=readme-ov-file#policy-language)
    * acceptable risk threshold
