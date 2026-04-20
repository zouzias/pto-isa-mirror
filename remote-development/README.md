# Remote Development Skill

Execute development tasks, tests, and debugging operations on remote servers through SSH.

## Features

- Connect to multiple pre-configured remote servers
- Automatic workspace directory navigation
- Conda environment activation support
- Execute any command on remote servers
- Clean output formatting for easy debugging

## Setup

### 1. Configure Servers

Copy `servers.json.template` to `servers.json` and update with your server details:

```bash
cp servers.json.template servers.json
```

Edit `servers.json` with your actual server information:

```json
{
  "servers": [
    {
      "name": "My Server A",
      "host": "server-a.example.com",
      "username": "your-username",
      "workspace": "/path/to/workspace"
    },
    {
      "name": "My Server B", 
      "host": "server-b.example.com",
      "username": "your-username",
      "workspace": "/path/to/other-workspace"
    }
  ]
}
```

### 2. Configure SSH Access

Ensure SSH key-based authentication is set up for your servers:

```bash
# Test SSH connection
ssh your-username@server-a.example.com
```

If you can connect without entering a password, SSH keys are configured correctly.

### 3. Verify Conda Setup

Ensure `conda_source.sh` exists in your workspace directory on each server. This file should set up the conda environment.

## Usage Examples

### Run Tests

```
User: "Run the CPU tests on server A with the pto-env environment"
```

The skill will:
1. Ask you to select Server A
2. Ask for conda environment (you enter: pto-env)
3. Ask for command (you enter: python3 tests/run_cpu.py)
4. Execute via SSH and show results

### Quick Status Check

```
User: "Check what's running on server B"
```

The skill will:
1. Ask you to select Server B
2. Skip conda environment (press Enter)
3. Ask for command (you enter: ps aux | grep python)
4. Execute and display results

### Build Project

```
User: "Build the project on server A"
```

The skill will:
1. Connect to Server A
2. Optionally activate build environment
3. Execute build commands
4. Show build output

## How It Works

The skill follows this workflow:

1. **Load Configuration**: Read server details from `servers.json`
2. **Select Server**: Present options and ask you to choose
3. **Environment Setup**: Ask for conda environment (optional)
4. **Command Execution**: Ask what command to run
5. **Execute Remotely**: Run via SSH with proper environment setup
6. **Display Results**: Show output clearly formatted

## Configuration File Location

The skill looks for `servers.json` in:
1. First, the skill directory: `remote-development/servers.json`
2. If not found, it will ask you to create it

## Requirements

- SSH client installed locally
- SSH key-based authentication configured for remote servers
- Conda available on remote servers (if using conda environments)
- `conda_source.sh` file in workspace directory on remote servers

## Troubleshooting

### SSH Connection Fails

- Test connection manually: `ssh username@host`
- Verify SSH keys are configured
- Check server hostname/IP is correct

### Command Not Found

- Verify the workspace path is correct
- Check if conda environment exists (if specified)
- Ensure command is available on remote server

### Conda Activation Fails

- Verify `conda_source.sh` exists in workspace
- Check conda environment name is correct
- Test manually: `conda activate <env>`

## Security Notes

- Never commit `servers.json` with real credentials to version control
- Use SSH key-based authentication (not passwords)
- Keep server access credentials secure
- Consider using environment variables for sensitive data

## License

This skill is part of the PTO Tile Library project.
