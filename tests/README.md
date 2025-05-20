# LinguaSign Tests

This directory contains test files for the LinguaSign project. We use pytest as our testing framework.

## Directory Structure

```
tests/
├── unit/                # Unit tests
│   ├── datasets/        # Tests for dataset functionality
│   ├── models/          # Tests for model architectures
│   └── api/             # Tests for API endpoints
├── integration/         # Integration tests
│   ├── model_pipeline/  # Tests for the complete model pipeline
│   └── api_frontend/    # Tests for API and frontend integration
├── fixtures/            # Test fixtures and mock data
├── conftest.py          # pytest configuration
└── README.md            # This file
```

## Running Tests

To run all tests:

```bash
pytest
```

To run specific test categories:

```bash
# Run only unit tests
pytest tests/unit/

# Run only model tests
pytest tests/unit/models/

# Run with coverage reporting
pytest --cov=.
```

## Writing Tests

When adding new tests, please follow these guidelines:

1. **Naming Convention**: Test files should be named `test_*.py`. Test functions should be named `test_*`.

2. **Organization**: Place tests in the appropriate directory based on what they're testing.

3. **Documentation**: Each test function should have a docstring explaining what it's testing.

4. **Fixtures**: Use fixtures for shared test resources. Place fixtures in `conftest.py` or in the `fixtures` directory.

5. **Mocking**: Use mocks for external dependencies. Import them from `unittest.mock`.

## Test Coverage

We aim for high test coverage, especially for critical components. Run the coverage report to identify areas that need more testing:

```bash
pytest --cov=. --cov-report=html
```

Then open `htmlcov/index.html` in your browser to view the report.

## Test Data

Test data should be small, self-contained, and deterministic. Add any necessary test data to the `fixtures` directory.

For tests that require model weights or other large files, use small dummy weights or mock the model behavior.

## Continuous Integration

Tests are automatically run in our CI pipeline. PRs cannot be merged unless all tests pass.
