import sys


def _build_error_message(error_message: str) -> str:
    """Build a message that includes the file and line number of an error."""
    _, _, exc_traceback = sys.exc_info()
    file_name = exc_traceback.tb_frame.f_code.co_filename
    return (
        f"Error occurred in script [{file_name}] "
        f"at line [{exc_traceback.tb_lineno}]: {error_message}"
    )


class CustomException(Exception):
    """Application exception that captures the originating file and line number."""

    def __init__(self, error_message: str) -> None:
        super().__init__(error_message)
        self.error_message = _build_error_message(error_message)

    def __str__(self) -> str:
        return self.error_message
