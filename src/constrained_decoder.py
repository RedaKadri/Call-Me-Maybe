import json
from typing import Any

import numpy as np
from pydantic_core import from_json

from llm_sdk import Small_LLM_Model
from src.schemas import FunctionDefinition


class ConstrainedDecoder:
    def __init__(
            self, functions_definitions: list[FunctionDefinition]
    ) -> None:
        self.model = Small_LLM_Model()

        self.functions_definitions: list[dict[str, Any]] = [
            {
                "name": fn.name,
                "description": fn.description,
                "parameters": dict(
                    (key, value.model_dump())
                    for key, value in fn.parameters.items()
                ),
                "returns": fn.returns.model_dump(),
            }
            for fn in functions_definitions
        ]
        self.valid_functions_names = [
            *(str(fn["name"]) for fn in self.functions_definitions),
            "none",
        ]

        with open(self.model.get_path_to_vocab_file()) as f:
            self.vocab = json.loads(f.read())

        self.prompt_prefix = '{"name": "'
        self.stop_tokens_id = {151643, 151645}

        self.state: dict[str, Any] = {
            "step": "INIT",
            "fn_name": None,
            "fn_params": None,
            "curr_fn_param_idx": 0,
        }

    def _is_valid_function_name(self, name: str) -> bool:
        for valid_function_name in self.valid_functions_names:
            if valid_function_name.startswith(name):
                return True
        return False

    def _is_valid_param_value(
            self, next_token: str, curr_param: str, token: str, type: str
    ) -> bool:
        if type == "string":
            if len(curr_param) == 0:
                return token.startswith('Ġ"')
            elif ('",' in next_token or '"}' in next_token) and "\\" not in next_token:
                if len(self.state["fn_params"]) == (
                        self.state["curr_fn_param_idx"] + 1
                ):
                    return token == '"}}'
                else:
                    return token.endswith('",')

        if type == "number" or type == "integer":
            if len(curr_param) == 0:
                return token in ["Ġ", "Ġ-"]
            elif next_token == "," or next_token == "}}":
                if len(self.state["fn_params"]) == (
                        self.state["curr_fn_param_idx"] + 1
                ):
                    return token == "}}"
                else:
                    return token == ","
            elif token in "0123456789.":
                return True

        if type == "bool":
            if len(curr_param) == 0:
                return token in ["Ġtrue", "Ġfalse"]
            elif next_token == "," or next_token == "}}":
                if len(self.state["fn_params"]) == (
                        self.state["curr_fn_param_idx"] + 1
                ):
                    return token == "}}"
                else:
                    return token == ","

        return False

    def _apply_constraints(
            self, logits: np.ndarray, output: str
    ) -> np.ndarray:
        constrained_logits = logits.copy()

        match self.state["step"]:
            case "EXPECT_FUNCTION_NAME":
                curr = output[len(self.prompt_prefix):]
                allowed_ids = [
                    id
                    for token, id in self.vocab.items()
                    if self._is_valid_function_name(curr + token)
                ]

            case "EXPECT_FUNCTION_NAME_END":
                allowed_ids = [self.vocab['"}'], self.vocab['",']]

            case "EXPECT_PARAMETER_KEY":
                if self.state["fn_name"] is None:
                    partial_output = from_json(output, allow_partial=True)

                    self.state.update(
                        {
                            "fn_name": partial_output["name"],
                            "fn_params": [
                                {param: param_info["type"]}
                                for fn in self.functions_definitions
                                if fn["name"] == partial_output["name"]
                                for param, param_info in fn[
                                    "parameters"
                                ].items()
                            ],
                        }
                    )

                (curr_fn_param_key,) = self.state["fn_params"][
                    self.state["curr_fn_param_idx"]
                ]

                allowed_ids = self.model.encode(curr_fn_param_key)[0].tolist()

            case "EXPECT_PARAMETER_KEY_END":
                allowed_ids = self.vocab['":']

            case "EXPECT_PARAMETER_VALUE":
                curr = output[(output.rfind(":") + 1):]
                ((_, curr_fn_param_type),) = self.state["fn_params"][
                    self.state["curr_fn_param_idx"]
                ].items()

                next_token_id = np.argmax(logits)
                next_token = self.model.decode([int(next_token_id)])

                allowed_ids = [
                    id
                    for token, id in self.vocab.items()
                    if self._is_valid_param_value(
                        next_token, curr, token, curr_fn_param_type
                    )
                ]

            case "EXPECT_PARAMETER_VALUE_END":
                allowed_ids = self.vocab['Ġ"']

            case _:
                allowed_ids = []

        if allowed_ids:
            constrained_logits = np.full_like(logits, -np.inf)
            constrained_logits[allowed_ids] = logits[allowed_ids]

        return constrained_logits

    def _update_state(self, output: str) -> str | None:
        if self.state["step"] == "EXPECT_FUNCTION_NAME":
            curr = output[len(self.prompt_prefix):]
            if curr in self.valid_functions_names:
                self.state["step"] = "EXPECT_FUNCTION_NAME_END"

        elif self.state["step"] == "EXPECT_FUNCTION_NAME_END":
            if output.endswith('",'):
                self.state["step"] = "EXPECT_PARAMETER_KEY"
                return ' "parameters": {"'
            if output.endswith('"}'):
                self.state["step"] = "END"

        elif self.state["step"] == "EXPECT_PARAMETER_KEY":
            curr = output[(output.rfind('"') + 1):]
            (curr_fn_param_key,) = self.state["fn_params"][
                self.state["curr_fn_param_idx"]
            ]
            if curr == curr_fn_param_key:
                self.state["step"] = "EXPECT_PARAMETER_KEY_END"

        elif self.state["step"] == "EXPECT_PARAMETER_KEY_END":
            self.state["step"] = "EXPECT_PARAMETER_VALUE"

        elif self.state["step"] == "EXPECT_PARAMETER_VALUE":
            ((_, curr_fn_param_type),) = self.state["fn_params"][
                self.state["curr_fn_param_idx"]
            ].items()
            if (
                (
                    ((curr_fn_param_type == "number" or curr_fn_param_type == "integer") and output.endswith(','))
                    or (curr_fn_param_type == "string" and output.endswith('",'))
                )
                or output.endswith("}")
            ):
                if output.endswith("}"):
                    self.state["step"] = "END"
                else:
                    self.state["step"] = "EXPECT_PARAMETER_VALUE_END"

        elif self.state["step"] == "EXPECT_PARAMETER_VALUE_END":
            self.state["curr_fn_param_idx"] += 1
            self.state["step"] = "EXPECT_PARAMETER_KEY"

        return None

    def generate(self, prompt: str, max_tokens: int = 84) -> str:
        self.state = {
            "step": "EXPECT_FUNCTION_NAME",
            "fn_name": None,
            "fn_params": None,
            "curr_fn_param_idx": 0,
        }

        output = self.prompt_prefix

        input_ids = self.model.encode(prompt + output)[0].tolist()

        for _ in range(max_tokens):
            logits = np.array(self.model.get_logits_from_input_ids(input_ids))

            logits = self._apply_constraints(logits, output)
            next_token_id = int(np.argmax(logits))

            if next_token_id in self.stop_tokens_id:
                break

            input_ids.append(next_token_id)

            token_text = self.model.decode([next_token_id])
            output += token_text

            injection = self._update_state(output)
            if injection:
                output += injection
                input_ids += self.model.encode(injection)[0].tolist()

            print(self.state["step"])
            print(output)
        print()

        return output
