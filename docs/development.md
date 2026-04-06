# Development

## Install with development tools

```bash
pip install -e .[dev]
```

## Run tests

```bash
pytest
```

## Lint and format

```bash
ruff check .
ruff format .
```

## Add a new geometry later

1. Add a geometry implementation under `src/geocydata/geometry/`
2. Register it in `src/geocydata/registry/geometries.py`
3. Implement or reuse a sampler in `src/geocydata/sampling/`
4. Extend tests with geometry-specific residual and CLI coverage
5. Document the new benchmark and any schema changes

