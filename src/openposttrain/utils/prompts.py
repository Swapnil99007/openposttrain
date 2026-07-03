from pathlib import Path


def load_prompt_template(prompt_path: str) -> str:
    path = Path(prompt_path)

    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    template = path.read_text()

    if "{question}" not in template:
        raise ValueError(f"Prompt template must contain '{{question}}': {prompt_path}")

    return template


def format_prompt(template: str, question: str) -> str:
    return template.format(question=question)
