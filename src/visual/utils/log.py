r"""
LOGGER utils.
Only for logging to set LOGGER in global.
"""

import logging

from functools import wraps

from .typeslib import str_or_None

__all__ = ['LOGGER', 'add_log_file', 'log_fps_time', 'log_results',
           'logging_initialize', 'logging_start_finish']


# TODO add log functions for log everything everywhere (deal results)
def _set_logger(name: str, ch_level=logging.INFO, developing=True):
    r"""
    Set LOGGER with StreamHandler.
    Args:
        name: str = __name__ usually use __name__ in log.py or other name.
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
    Add FileHandler for logger to write in *.log.
    Args:
        filepath: = '**/logger.log'.
        fh_level: = logging.DEBUG, logging.INFO etc.
        mode: str = 'a', 'w' etc.
    """
    # TODO The way to setting filepath need to upgrade in the future
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


def logging_initialize(name: str_or_None = None):
    r"""
    Decorator with parameter name.
    Add LOGGER.info(f'Initialize {...}') in the beginning and end of func.

    Examples:
        @logging_initialize('test')
        def test(): ...

    Args:
        name: str_or_None = the func name for logging message.
    """

    def function(func):
        log_name = func.__name__ if name is None else name

        @wraps(func)
        def wrapper(*args, **kwargs):
            LOGGER.info(f'Initializing {log_name}...')
            x = func(*args, **kwargs)
            LOGGER.info(f'Initialize {log_name} successfully')
            return x

        return wrapper

    return function


def logging_start_finish(name: str_or_None = None):
    r"""
    Decorator with parameter name.
    Add LOGGER.info(f'Start {...}') in the beginning and end of func.

    Examples:
        @logging_start_finish('testing')
        def test(): ...

    Args:
        name: str_or_None = the func name for logging message.
    """

    def function(func):
        log_name = name if name is not None else func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            LOGGER.info(f'Start {log_name}')
            x = func(*args, **kwargs)
            LOGGER.info(f'Finished {log_name}')
            return x

        return wrapper

    return function


def log_results(results, result_names):
    fmt = '<10.4f'
    space = ' ' * 50
    show = ''.join([f'{x}: {y:{fmt}} ' for x, y in zip(result_names, results)])
    LOGGER.info(f'{space}{show}')


def log_fps_time(fps_time):
    space = ' ' * 50
    LOGGER.info(f'{space}Speed {fps_time[1]:.2f} ms per image, FPs: {fps_time[0]:.1f}, no accuracy')


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
