"""Минимальный пример: печатает приветствие."""

from __future__ import annotations


def greet(name: str = "roman") -> str:
    return f"hello world {name}"


def main() -> None:
    print(greet())


if __name__ == "__main__":
    main()
