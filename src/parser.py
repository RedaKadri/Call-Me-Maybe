from typing import TypeVar

from pydantic import TypeAdapter

from src.schemas import FunctionDefinition, Prompt


def parse_args(argv: list[str]) -> dict[str, str]:
    if len(argv) % 2 != 0:
        raise ValueError("Unexpected arguments format")
    pairs = [(argv[i], argv[i + 1]) for i in range(0, len(argv), 2)]

    config = {
        "functions_definition": "data/input/functions_definition.json",
        "input": "data/input/function_calling_tests.json",
        "output": "data/output/function_calls.json",
    }
    valid_config_keys = set(config.keys())

    for flag, value in pairs:
        if not flag.startswith("--"):
            raise ValueError(f"Expected a '--' flag, got '{flag}'.")
        key = flag.removeprefix("--")
        if key not in valid_config_keys:
            raise ValueError(
                f"Unknown flag '{flag}'. Valid flags: "
                + ", ".join(f"--{k}" for k in valid_config_keys)
            )
        config[key] = value

    return config


def config_parser(config: dict[str, str]) -> any:
    T = TypeVar("T")

    def load_file(path: str, model: TypeAdapter[T]) -> T:
        with open(path) as f:
            return model.validate_json(f.read())

    functions_definition = load_file(
        config["functions_definition"], TypeAdapter(list[FunctionDefinition])
    )
    prompts = load_file(config["input"], TypeAdapter(list[Prompt]))

    return {
        "functions_definition": functions_definition,
        "prompts": prompts
    }
