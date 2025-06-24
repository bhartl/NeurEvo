from datetime import datetime


def get_timestamp(fmt: str = '%Y.%m.%d, %H:%M:%S'):
    """ Retrieves a stringified timestamp according to the provided format

    :param fmt: Format string for timestamp, defaults to '%Y.%m.%d, %H:%M:%S'.
    :return: Stringified timestamp of specified format.
    """
    return datetime.now().strftime(fmt)
