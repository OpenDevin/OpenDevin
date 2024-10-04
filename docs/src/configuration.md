# OpenDevin Configuration Options

OpenDevin provides various configuration options to customize its behavior. This page documents all available options.

## General Configuration

- `project_name`: The name of your project.
- `output_dir`: The directory where output files will be saved.
- `max_iterations`: The maximum number of iterations for the AI to attempt solving a task.
- `max_time`: The maximum time (in seconds) for the AI to work on a task.

## AI Model Configuration

- `model`: The AI model to use (e.g., "gpt-4", "gpt-3.5-turbo").
- `temperature`: Controls the randomness of the AI's output (0.0 to 1.0).
- `max_tokens`: The maximum number of tokens to generate in the AI's response.

## Execution Environment

- `python_path`: The path to the Python interpreter to use.
- `allowed_modules`: A list of Python modules that are allowed to be imported.
- `timeout`: The maximum execution time for a single command (in seconds).

## Logging and Debugging

- `log_level`: The level of logging detail (e.g., "DEBUG", "INFO", "WARNING", "ERROR").
- `log_file`: The file path for saving logs.
- `debug_mode`: Enable or disable debug mode (true/false).

## Security

- `allow_internet_access`: Allow the AI to access the internet (true/false).
- `allowed_domains`: A list of allowed domains if internet access is enabled.
- `max_file_size`: The maximum size (in bytes) of files that can be created or modified.

## Custom Behavior

- `custom_prompts`: A dictionary of custom prompts to use for specific tasks.
- `task_specific_settings`: A dictionary of settings that apply to specific tasks or modules.

Please refer to the OpenDevin documentation for more detailed information on how to use these configuration options in your project.
