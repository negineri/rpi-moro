# Python Coding Guidelines (For Large-Scale Projects, Quality-Oriented)

## 1. **General Principles**

- **Prioritize readability**: “Code is read more often than it is written.”
- **Consistency is key**: Maintain a unified style throughout the project.
- **Minimal dependencies**: Use only what’s necessary.
- **Testability by design**: Write code that’s easy to test.
- **Documentation matters**: Use type hints and docstrings consistently.

---

## 2. **Code Style**

Follow PEP8.

---

## 3. **Type Hinting (PEP484)**

- **All functions and methods must include type annotations**

```python
def fetch_data(url: str, timeout: int = 10) -> dict:
    ...
```

- **Use variable annotations when appropriate**

```python
users: list[str] = []
```

---

## 4. **Docstrings (PEP257 + Google Style)**

- Add docstrings to **all functions, classes, and modules**

```python
def login(user_id: str, password: str) -> bool:
    """
    Authenticates a user.

    Args:
        user_id (str): User ID.
        password (str): Password.

    Returns:
        bool: True if authentication succeeds, otherwise False.
    """
```

---

## 5. **Structure and Design Philosophy**

- Functions should do **one thing only**
- Keep classes **small and focused**
- **No global variables**
- Use **dependency injection** where applicable (especially for external APIs, DBs)

---

## 6. **Testing**

- Use **pytest** as the standard testing framework
- Store tests in a dedicated `tests/` directory
- Use function names like `test_<function_name>()`
- Emphasize **unit testing** and **mocking**
- Aim for **90%+ test coverage** (enforced via CI)

---

## 7. **Static Analysis and Formatting**

- Code Formatter: **black**
- Linters: **ruff**, **mypy**, and **pylint**
- In CI, enforce:
  - Code formatting
  - Type checking
  - Docstring coverage
  - Cyclomatic complexity limits

---

## 8. **Dependency and Environment Management**

- Use **`pyproject.toml`** with **uv** for dependency management
- Separate development and production environments
- Use `.env` files or `settings.py` for configuration
- **Do not hard-code secrets**

---

## 9. **Logging and Exception Handling**

- Use the standard **`logging`** library (avoid `print()`)
- Catch exceptions meaningfully—**do not overuse `try/except`**
- Define **custom exception classes** where needed

```python
class AuthenticationError(Exception):
    """Raised when authentication fails."""
    pass
```

---

## 10. **Suggested Project Structure**

```text
project_name/
├── app/
│   ├── __init__.py
│   ├── models/
│   ├── services/
│   ├── controllers/
│   └── utils/
├── tests/
│   ├── unit/
│   └── integration/
├── config/
│   └── settings.py
├── pyproject.toml
└── README.md
```

---

## 11. **CI/CD Recommendations**

- Use **GitHub Actions** or **GitLab CI**
- Pipelines should perform:
  - Linting
  - Type checking
  - Testing + coverage reporting
  - Build verification (including Docker if used)

---

## 12. **Tools Recommendations**

- Use **uv** instead of pip, pyenv and other.
