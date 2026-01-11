# Agent Utilities

This directory contains utilities for the AI agent to improve workflow efficiency.

## Command Monitor (`command_monitor.py`)

**Purpose**: Optimize `command_status` polling by predicting command completion time.

### Features

1. **Pattern-Based Estimation**
   - Detects operations from log output (model loading, embedding, parsing)
   - Estimates time based on operation type and scale

2. **Adaptive Backoff**
   - Progressively increases wait time when no output is detected
   - Prevents excessive polling of long-running commands

3. **Operation-Specific Times**
   - Model loading: ~120s
   - Embedding: ~0.15s per chunk
   - PDF parsing: ~30s per file
   - Evaluation: ~8s per question

### Usage Example

```python
from .agent.command_monitor import get_optimal_wait_time, reset_estimator

# Start monitoring a new command
reset_estimator()

# Get optimal wait time based on current output
output = "Embedding 781 chunks..."
wait_time = get_optimal_wait_time(command_line, output)

# Use in command_status call
command_status(CommandId=cmd_id, WaitDurationSeconds=wait_time)
# Will wait ~137s instead of generic 60s
```

### Benefits

- **Reduced Token Usage**: Fewer unnecessary `command_status` calls
- **Faster Feedback**: Appropriate timing for each operation phase
- **Responsive**: Quick checks for fast operations, patient for slow ones

### Testing

Run the module directly to see estimation examples:

```bash
python .agent/command_monitor.py
```

## Best Practices

1. **Reset on New Commands**: Call `reset_estimator()` when starting a new command
2. **Provide Full Output**: Pass complete output for accurate pattern matching
3. **Set Reasonable Caps**: Module caps estimates at 5 minutes to prevent indefinite waits
4. **Monitor Check Count**: After 3-4 "no output" checks, consider alternative actions
