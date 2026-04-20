import datetime
import sys
import time

from pydantic import ValidationError

from src.call_me_maybe import CallMeMaybe
from src.parser import load_input_data, parse_cli_args

if __name__ == "__main__":
    try:
        settings = parse_cli_args(sys.argv[1:])
    except ValueError as e:
        print(
            f"Error: {e}\n"
            "Usage: uv run python -m src "
            "[--functions_definition <path>] "
            "[--input <path>] "
            "[--output <path>]"
        )
        sys.exit(1)

    try:
        input_data = load_input_data(settings)
    except FileNotFoundError as e:
        print(f"Error: File not found — {e.filename}")
        sys.exit(1)
    except PermissionError as e:
        print(f"Error: Permission denied reading '{e.filename}'")
        sys.exit(1)
    except ValidationError as e:
        print(e)
        sys.exit(1)

    try:
        call_me_maybe = CallMeMaybe(
            input_data["functions_definition"], input_data["prompts"]
        )

        start = time.time()
        call_me_maybe.run()
        end = time.time() - start

        print(f"\n{datetime.timedelta(seconds=end)}")
    except (KeyboardInterrupt, EOFError):
        pass
