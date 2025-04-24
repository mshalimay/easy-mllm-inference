class TestAPIError(Exception):
    """
    Exception for debugging. Mimics attributes of Google API errors.
    """

    def __init__(self, message: str, status: str = ""):
        super().__init__(message)
        self.message = message
        self.status = status
