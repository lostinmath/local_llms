# Contributing to Local LLMs

Thank you for your interest in contributing to Local LLMs! This document provides guidelines for contributing to the project.

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help maintain a welcoming environment

## How to Contribute

### Reporting Issues

- Check if the issue already exists
- Provide clear description and reproduction steps
- Include relevant system information (OS, Python version, etc.)

### Submitting Changes

1. Fork the repository
2. Create a new branch for your feature/fix
3. Make your changes with clear commit messages
4. Add/update tests as needed
5. Ensure all tests pass
6. Submit a pull request

### Development Setup

```bash
# Clone the repository
git clone https://github.com/lostinmath/local_llms.git
cd local_llms

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black local_llms/
isort local_llms/
```

### Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and concise

### Testing

- Write tests for new features
- Ensure existing tests pass
- Aim for good test coverage

### Documentation

- Update README.md for user-facing changes
- Add docstrings to new functions/classes
- Update examples if needed

## Privacy & Security

Since this project focuses on privacy and security:

- Never commit sensitive data or credentials
- Test security features thoroughly
- Document security implications of changes
- Follow secure coding practices

## Questions?

Feel free to open an issue for questions or discussions.

Thank you for contributing! 🎉
