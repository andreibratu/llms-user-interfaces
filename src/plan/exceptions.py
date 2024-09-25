from enum import IntEnum


class ExceptionCode(IntEnum):
    UNEXPECTED = 0
    ARGUMENT_VALIDATION = 1
    MEMORY = 2
    TOOL_SIGNATURE = 3
    PLAN_GENERATION = 4
    INVALID_ARGUMENT = 5


class BenchmarkException(Exception):
    def __init__(
        self,
        code: ExceptionCode,
        message: str,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class MisgeneratedPlanException(BenchmarkException):
    def __init__(self, message: str, output: str) -> None:
        super().__init__(
            code=ExceptionCode.PLAN_GENERATION,
            message=message,
        )
        self.output = output
