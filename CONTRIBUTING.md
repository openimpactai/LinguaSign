# Contributing to LinguaSign

Thank you for your interest in contributing to LinguaSign! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Pull Requests](#pull-requests)
- [Development Workflow](#development-workflow)
- [Style Guidelines](#style-guidelines)
- [Community Involvement](#community-involvement)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [contact@openimpactai.org](mailto:contact@openimpactai.org).

## Getting Started

To get started with contributing to LinguaSign, first set up your development environment:

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR-USERNAME/LinguaSign.git
   cd LinguaSign
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a branch for your contribution:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## How to Contribute

### Reporting Bugs

When reporting bugs, please include:

1. A clear and descriptive title
2. Steps to reproduce the bug
3. Expected behavior
4. Actual behavior
5. Screenshots or code samples if applicable
6. System information (OS, Python version, etc.)

Use the GitHub issue tracker to report bugs by creating a new issue with the "bug" label.

### Suggesting Enhancements

Enhancement suggestions are welcome! When suggesting enhancements, please include:

1. A clear and descriptive title
2. A detailed description of the proposed enhancement
3. Any potential implementation details or ideas
4. Why this enhancement would be useful to most users

Use the GitHub issue tracker to suggest enhancements by creating a new issue with the "enhancement" label.

### Pull Requests

1. Fork the repository
2. Create a feature branch
3. Make your changes with clear commit messages
4. Make sure your code passes all tests
5. Submit a pull request with a clear description of the changes

For significant changes, please open an issue first to discuss your proposed changes.

## Development Workflow

1. Choose an issue to work on or create a new one
2. Comment on the issue to let others know you're working on it
3. Fork the repository and create a feature branch
4. Make your changes with clear commit messages
5. Add or update tests if necessary
6. Update documentation if necessary
7. Submit a pull request
8. Respond to feedback and update your pull request if necessary

## Style Guidelines

### Code Style

LinguaSign follows the PEP 8 style guide for Python code. We use `flake8` and `black` for code linting and formatting.

Before submitting a pull request, please run:

```bash
flake8 .
black .
```

### Documentation Style

- Use Markdown for documentation
- Follow the established documentation structure
- For docstrings, use the Google style:

```python
def function_with_types_in_docstring(param1, param2):
    """Example function with types documented in the docstring.
    
    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.
    
    Returns:
        bool: The return value. True for success, False otherwise.
    """
    return True
```

### Commit Message Guidelines

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

## Community Involvement

### Special Considerations for Sign Language Processing

When contributing to LinguaSign, please keep in mind:

1. **Deaf Community Involvement**: We highly value input from deaf and hard-of-hearing individuals. Sign language is not just a technical problem but a cultural and linguistic one.

2. **Dataset Ethics**: Be mindful of the ethical considerations around sign language datasets, including consent, representation, and cultural sensitivity.

3. **Accessibility Testing**: When developing UI components, ensure they are accessible according to WCAG guidelines.

## Thank You!

Thank you for contributing to LinguaSign! Your efforts help make sign language technology more accessible to everyone.
