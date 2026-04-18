from pydantic import TypeAdapter

from src.schemas import FunctionDefinition, Prompt


def parse_cli_args(argv: list[str]) -> dict[str, str]:
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


def load_input_data(config: dict[str, str]) -> dict[str, list]:
    with open(config["functions_definition"]) as f:
        functions_definition = TypeAdapter(list[FunctionDefinition]).validate_json(
            f.read()
        )

    with open(config["input"]) as f:
        prompts = TypeAdapter(list[Prompt]).validate_json(f.read())

    return {
        "functions_definition": functions_definition,
        "prompts": prompts,
    }
