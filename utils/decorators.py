def error_handler(func):
    """
    Decorator to handle errors in functions
    Args:
        func: Function to decorate
    Return:
        True/False, message
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return False, str(e)

    return wrapper
