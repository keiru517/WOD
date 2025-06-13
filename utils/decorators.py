def error_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return True, func(*args, **kwargs)
        except Exception as e:
            return False, str(e)

    return wrapper
