# Style Guide for LinguaSign

This document outlines the coding and documentation style guidelines for the LinguaSign project. Consistent style helps make our codebase more maintainable and easier to understand for all contributors.

## Python Code Style

LinguaSign follows the [PEP 8](https://pep8.org/) style guide for Python code with a few project-specific additions.

### General Guidelines

- Use 4 spaces for indentation (no tabs)
- Maximum line length of 88 characters (compatible with Black)
- Use snake_case for variable and function names
- Use CamelCase for class names
- Use UPPER_CASE for constants

### Imports

Organize imports in the following order, with a blank line between each group:

1. Standard library imports
2. Related third-party imports
3. Local application/library specific imports

Example:
```python
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from models.base import BaseModel
from utils.preprocessing import preprocess_video
```

### Docstrings

Use Google-style docstrings:

```python
def complex_function(param1, param2):
    """Short description of the function.

    More detailed description of the function and its behavior.

    Args:
        param1 (int): Description of param1.
        param2 (str): Description of param2.

    Returns:
        bool: Description of return value.

    Raises:
        ValueError: If param1 is negative.
        TypeError: If param2 is not a string.

    Examples:
        >>> complex_function(1, "example")
        True
    """
```

For classes:

```python
class SampleClass:
    """Summary of class.

    More detailed description of the class and its behavior.

    Attributes:
        attr1 (int): Description of attr1.
        attr2 (str): Description of attr2.
    """

    def __init__(self, param1, param2):
        """Initialize SampleClass.

        Args:
            param1 (int): Description of param1.
            param2 (str): Description of param2.
        """
```

### Comments

- Use comments sparingly and only when necessary to explain complex logic
- Keep comments up-to-date when modifying code
- Use complete sentences with proper capitalization and punctuation

Example:
```python
# Bad: increment i
i += 1

# Good: Increment the counter to track the number of processed frames
frame_counter += 1
```

### Code Formatting Tools

We use the following tools for code formatting and linting:

- **Black**: For automatic code formatting
- **isort**: For sorting imports
- **flake8**: For style guide enforcement

Before submitting a pull request, run:

```bash
black .
isort .
flake8
```

## Documentation Style

### Markdown Files

- Use ATX-style headers (`#` for headers, not underlines)
- Use sentence case for headers (capitalize only the first word)
- Use ordered lists for sequential instructions
- Use unordered lists for non-sequential items
- Use backticks for code, file names, and technical terms

### README Files

Each directory should have a README.md file with:

1. A brief description of the directory's purpose
2. An explanation of the key files and subdirectories
3. Usage examples where applicable
4. Links to relevant documentation

## Git Workflow

### Commit Messages

- Use the imperative mood ("Add feature" not "Added feature")
- Limit the first line to 72 characters
- Include the issue number if applicable
- Include a brief description of the changes

Format:
```
[component]: Brief description

More detailed explanation if necessary.

Fixes #123
```

Examples:
```
models: Add transformer-based sign language translator

Implement a transformer encoder-decoder architecture for sign language translation,
based on the paper "Neural Sign Language Translation" (CVPR 2018).

Fixes #45
```

### Branch Naming

- Use kebab-case (lowercase with hyphens)
- Prefix with the type of change:
  - `feature/` for new features
  - `fix/` for bug fixes
  - `docs/` for documentation changes
  - `refactor/` for code refactoring
  - `test/` for adding or fixing tests

Examples:
- `feature/transformer-model`
- `fix/memory-leak-in-preprocessing`
- `docs/update-model-architecture`
- `refactor/improve-data-loader`

## Frontend Code Style (if applicable)

### JavaScript/TypeScript

- Use 2 spaces for indentation
- Use semicolons
- Use camelCase for variables and functions
- Use PascalCase for classes and React components
- Use ESLint and Prettier for formatting

### CSS/SCSS

- Use 2 spaces for indentation
- Use kebab-case for class names
- Use SCSS nesting sparingly (max 3 levels)
- Follow BEM (Block Element Modifier) naming convention

## Continuous Integration

Our CI pipeline will automatically check for style violations. Pull requests with style issues will not be merged until the issues are fixed.

## Exceptions

If you need to deviate from the style guide for a valid reason (e.g., compatibility with an external library), include a comment explaining why the exception is necessary.

Example:
```python
# pylint: disable=invalid-name
# Using camelCase to match the API of the external library
someExternalFunction = lib.getFunction()
```

## Acknowledgment

By contributing to LinguaSign, you agree to follow these style guidelines. Thank you for helping maintain code quality and consistency!
