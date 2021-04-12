class Error(Exception):
    """Base class for other exceptions"""
    pass


class ModelNoTrainYetError(Error):
    """Raised when the model has been trained yet but training plots are asked"""
    pass