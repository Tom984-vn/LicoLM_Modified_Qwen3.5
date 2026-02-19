## What does this PR do?

Brief description of the change.

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update
- [ ] Refactoring (no behavior change)

## Testing

- [ ] Tested on x86-64 (Linux/macOS/Windows)
- [ ] Tested on ARM64 (Raspberry Pi)
- [ ] Tested with TinyLlama 1.1B Q4_K_M

**Test command:**
```bash
./picolm model.gguf -p "The capital of France is" -n 20 -t 0
```

**Output:**
```
(paste output here)
```

## Checklist

- [ ] Code compiles without warnings (`make native`)
- [ ] No new dependencies added
- [ ] Memory usage not increased (check stderr output)
- [ ] Works with `--json` mode (if touching generation/sampling)
