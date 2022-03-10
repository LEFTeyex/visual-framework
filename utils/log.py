r"""
LOGGER utils.
Only for logging to set LOGGER in global.
"""

import logging

from functools import wraps

from utils.typeslib import _str_or_None

__all__ = ['LOGGER', 'add_log_file',
           'log_loss',
           'logging_initialize', 'logging_start_finish'
           ]


# TODO add log functions for log everything everywhere (deal results)
def _set_logger(name: str, ch_level=logging.INFO, developing=True):
    r"""
    Set LOGGER with StreamHandler.
    Args:
        name: str = __name__ usually use __name__ in log.py or other name
        ch_level: = logging.DEBUG, logging.INFO etc.

    Return LOGGER
    """
    # get logger and set level
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # create formatter for handler
    # todo args can change
    if developing:
        fmt = '{asctime:<18} {levelname:<10} {filename:<20} {lineno:<4} {message:<80} {name}'
    else:
        fmt = '{levelname:<10} {message:<80}'
    formatter = logging.Formatter(fmt=fmt,
                                  datefmt='%Y-%m-%d %H:%M:%S',
                                  style='{')
    # set StreamHandler
    ch = logging.StreamHandler()
    ch.setLevel(ch_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


r"""Set LOGGER for global, use it by from utils.log import LOGGER, add_log_file"""
# todo args can change
LOGGER = _set_logger(__name__, ch_level=logging.INFO, developing=False)


def add_log_file(filepath, fh_level=logging.DEBUG, mode: str = 'a'):
    r"""
    Add FileHandler for logger to write in *.txt
    Args:
        filepath: = '**/logger.log'
        fh_level: = logging.DEBUG, logging.INFO etc.
        mode: str = 'a', 'w' etc.
    """
    # TODO The way to setting filepath need to design in the future
    # create formatter for handler
    # todo args can change
    formatter = logging.Formatter(fmt='{asctime:<18} {levelname:<10} {filename:<20} {lineno:<4} {message:<80} {name}',
                                  datefmt='%Y-%m-%d %H:%M:%S',
                                  style='{')
    # set FileHandler
    fh = logging.FileHandler(filepath, mode=mode)
    fh.setLevel(fh_level)
    fh.setFormatter(formatter)
    LOGGER.addHandler(fh)


def log_loss(when: str, epoch, name, loss):
    when = when.title()
    LOGGER.debug(f'{when} epoch{epoch}: {name} is {loss}')


def logging_initialize(name: _str_or_None = None):
    r"""
    Decorator with parameter name.
    Add LOGGER.info(f'Initialize {...}') in the beginning and end of func.

    Examples:
        @logging_initialize('test')
        def test(): ...

    Args:
        name: _str_or_None = the func name for logging message
    """

    def get_function(func):
        log_name = name if name is not None else func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            LOGGER.info(f'Initializing {log_name}...')
            func(*args, **kwargs)
            LOGGER.info(f'Initialize {log_name} successfully')

        return wrapper

    return get_function


def logging_start_finish(name: _str_or_None = None):
    r"""
    Decorator with parameter name.
    Add LOGGER.info(f'Start {...}') in the beginning and end of func.

    Examples:
        @logging_start_finish('testing')
        def test(): ...

    Args:
        name: _str_or_None = the func name for logging message
    """

    def get_function(func):
        log_name = name if name is not None else func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            LOGGER.info(f'Start {log_name}')
            func(*args, **kwargs)
            LOGGER.info(f'Finished {log_name}')

        return wrapper

    return get_function


@logging_initialize('test')
def _test():
    r"""test LOGGER"""
    add_log_file(filepath='logger.log')
    LOGGER.debug('debug')
    LOGGER.info('info')
    LOGGER.warning('warning')
    LOGGER.error('error')
    LOGGER.critical('critical')
    try:
        raise TypeError
    except TypeError as error:
        LOGGER.exception(error)


if __name__ == '__main__':
    _test()
    import functools

    functools.lru_cache()
