---
name: remote-development
description: Execute development tasks on remote servers via SSH. Use this skill whenever the user asks to run tests, execute commands, or perform debugging on remote servers, mentions needing to work on a server, or wants to deploy/test code remotely. This skill handles SSH connections, environment setup, and command execution on remote development servers.
---

# Remote Development

Execute development tasks, tests, and debugging operations on remote servers through SSH.

## When to Use This Skill

Use this skill when the user:
- Asks to run tests or execute commands on a remote server
- Mentions needing to debug or deploy code on a server
- Wants to work on a remote development environment
- References connecting to a server for development tasks
- Needs to perform operations that require remote server access

## Prerequisites

This skill requires:
- SSH access to remote servers with key-based authentication configured
- Server configurations defined in servers.json
- SSH client available in the local environment

## Server Configuration

Servers must be configured in `servers.json` with the following structure:

```json
{
  "servers": [
    {
      "name": "Server A",
      "host": "192.168.1.100",
      "username": "developer",
      "workspace": "/path/to/workspace"
    },
    {
      "name": "Server B",
      "host": "192.168.1.101",
      "username": "developer", 
      "workspace": "/path/to/different/workspace"
    }
  ]
}
```

## Workflow

### Step 1: Load Server Configuration

Read `servers.json` to get available servers:

```bash
cat servers.json
```

If the file doesn't exist or is invalid, ask the user to provide server information.

### Step 2: Present Server Options

Show the user available servers and ask them to choose one:

```
Available servers:
1. Server A (192.168.1.100)
2. Server B (192.168.1.101)

Which server would you like to use? (Enter 1, 2, or server name)
```

### Step 3: Identify Task and Environment

Ask the user for:
1. **Conda environment name** (if they want to activate one)
2. **Command to execute** (test command, build script, etc.)

Example prompts:
- "What conda environment should I activate? (Press Enter to skip)"
- "What command would you like me to run on the remote server?"

### Step 4: Execute Remote Command

Construct and execute the SSH command with the following sequence:

```bash
ssh <username>@<host> "cd <workspace> && source conda_source.sh && conda activate <env> && <command>"
```

Example:
```bash
ssh developer@192.168.1.100 "cd /home/developer/workspace && source conda_source.sh && conda activate pto-env && python3 tests/run_cpu.py"
```

**Important:**
- Always quote the entire remote command
- Include `cd <workspace>` first
- Source `conda_source.sh` before activating environment
- Use `conda activate <env>` if environment provided
- Pass through the user's command exactly as specified

### Step 5: Present Results

Display the command output clearly to the user. Include:
- The full command that was executed
- The output from the remote server
- Any errors or warnings
- Exit status if available

## Command Execution Guidelines

- **Preserve user intent**: Execute commands exactly as specified
- **Show output**: Display both stdout and stderr
- **Handle errors**: If a command fails, show the error output clearly
- **Interactive commands**: Warn user that interactive commands may not work well over SSH
- **Long-running commands**: For long operations, consider running in background with nohup

## Environment Setup

### Conda Environment

If the user specifies a conda environment:
1. Always source `conda_source.sh` first (located in workspace root)
2. Then activate with `conda activate <environment_name>`
3. Execute the user's command

If no environment is specified, skip conda activation and run the command directly after sourcing `conda_source.sh`.

### Working Directory

Always start by changing to the configured workspace directory:
```bash
cd <workspace>
```

## Error Handling

### Connection Failures

If SSH connection fails:
- Check if the server is reachable
- Verify SSH keys are configured
- Suggest testing SSH connection manually: `ssh <username>@<host>`

### Command Failures

If the remote command fails:
- Show the error output
- Suggest checking if the command exists on the remote server
- Verify the workspace path is correct
- Check if conda environment exists (if specified)

### Missing Configuration

If `servers.json` is missing:
- Ask user to create it with server information
- Provide template structure
- Suggest placing it in the skill directory or project root

## Examples

### Example 1: Run Tests

**User request:** "Run the CPU tests on server A"

**Flow:**
1. Load servers.json, show available servers
2. User selects "Server A"
3. Ask for conda environment: User enters "pto-env"
4. Ask for command: User enters "python3 tests/run_cpu.py"
5. Execute: `ssh developer@192.168.1.100 "cd /workspace && source conda_source.sh && conda activate pto-env && python3 tests/run_cpu.py"`
6. Display results

### Example 2: Build Project

**User request:** "Build the project on server B"

**Flow:**
1. User selects "Server B"
2. Ask for environment: User enters "build-env"
3. Ask for command: User enters "make && make test"
4. Execute and display results

### Example 3: Check Status

**User request:** "Check what's running on server A"

**Flow:**
1. User selects "Server A"
2. Ask for environment: User presses Enter (skip)
3. Ask for command: User enters "ps aux | grep python"
4. Execute and display results

## Best Practices

- **Always quote remote commands** to handle spaces and special characters
- **Use full paths** when in doubt about the working directory
- **Check exit codes** if you need to verify success/failure
- **Show timestamps** for long-running operations
- **Preserve formatting** from command output when possible
- **Ask before destructive operations** like rm, mv, or git push

## Limitations

- Interactive commands that require user input may not work well
- GUI applications cannot be run through this skill
- Very long-running operations might timeout; consider using nohup or screen
- File transfers are not supported (use scp/sftp manually for that)
