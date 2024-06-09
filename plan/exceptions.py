from typing import Dict, Literal, Union


class ExecutionException(Exception):

    def __init__(
        self,
        code: int,
        message: str,
        taxonomy: str,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.taxonomy = taxonomy
        self.message = message


class MisgeneratedPlanException(ExecutionException):

    def __init__(self, code: int, message: str, output: str, tokens: int) -> None:
        super().__init__(
            code=code,
            message=message,
            taxonomy="misgenerated_plan",
        )
        self.output = output
        self.tokens = tokens


class ToolException(ExecutionException):

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        tool_name: str,
        taxonomy: Literal[
            "illegal_value",
            "illegal_type",
            "out_of_enum",
            "illegal_state",
        ],
        code: int,
        message: str,
        details: Union[Dict] = None,
    ) -> None:
        super().__init__(code=code, message=message, taxonomy=taxonomy)
        if details is None:
            details = {}
        self.tool_name = tool_name
        self.details = details


class MemoryException(ExecutionException):

    def __init__(
        self,
        taxonomy: Literal[
            "illegal_format",
            "illegal_read",
            "illegal_write",
        ],
        code: int,
        message: str,
    ) -> None:
        super().__init__(
            code=code,
            message=message,
            taxonomy=taxonomy,
        )


class UnknownToolException(ExecutionException):

    def __init__(self, code: int) -> None:
        super().__init__(
            code=code,
            taxonomy="unknown_tool",
            message="",
        )
