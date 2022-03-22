r"""
LOGGER utils.
Only for logging to set LOGGER in global.
"""

import logging

from functools import wraps

from utils import WRITER
from utils.typeslib import _str_or_None

__all__ = ['LOGGER', 'add_log_file',
           'log_loss', 'log_loss_and_metrics',
           'logging_initialize', 'logging_start_finish']


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


def log_loss(when: str, epoch: int, name, loss):
    when = when.title()
    LOGGER.debug(f'{when} epoch{epoch}: {name} is {loss}')


def log_loss_and_metrics(when: str, epoch: int, last, writer, cls_names, loss_name, loss_all, metrics, fps_time):
    """
    metrics = (ap_all, f1_all, p_all, r_all, cls_name_number)

    ap_all = (ap50_95, ap50, ap75, ap)
    f1_all = (mf1, f1)
    p_all = (mp, p)
    r_all = (mr, r)
    cls_name_number = (cls, cls_number)
    """
    when = when.title()
    separate = '-' * 60
    t_fmt = '<15'  # title format
    fmt = t_fmt + '.3f'
    space = ' ' * 50

    # speed
    if last:
        LOGGER.info(f'{space}Speed {fps_time[1]:.2f} ms per image, FPs: {fps_time[0]:.1f}, accuracy')
    else:
        LOGGER.info(f'{space}Speed {fps_time[1]:.2f} ms per image, FPs: {fps_time[0]:.1f}, no accuracy')

    (ap50_95, ap50, ap75, ap), (mf1, f1), (mp, p), (mr, r), (cls, cls_number) = metrics
    if ap is not None:
        name_writer = ('P_50', 'R_50', 'F1_50', 'AP50', 'AP75', 'AP5095')
        value_writer = (mp[0], mr[0], mf1[0], ap50, ap75, ap50_95)
        WRITER.add_epoch_curve(writer, 'val_metrics', value_writer, name_writer, epoch)

        LOGGER.debug(f'{separate}')
        log_loss(when, epoch, loss_name, loss_all)
        LOGGER.debug(f'P_50: {mp[0]}, R_50: {mr[0]}, F1_50: {mf1[0]}, '
                     f'AP50: {ap50}, AP75: {ap75}, AP/AP5095: {ap50_95}')
        LOGGER.debug(f'P_5095: {mp}, R_5095: {mr}, F1_5095: {mf1}')

        rest = (cls, ap, f1, p, r)
        LOGGER.debug(f'cls_name, cls_number, (IoU=0.50:0.95) AP, F1, P, R')
        for c, ap_c, f1_c, p_c, r_c in zip(*rest):
            name_c = cls_names[c]
            number_c = cls_number[c]
            LOGGER.debug(f'{name_c}, {number_c}, {ap_c}, {f1_c}, {p_c}, {r_c}')
        LOGGER.debug(f'{separate}')

        if last:
            LOGGER.info(f"{'class_name':{t_fmt}}"
                        f"{'class_number':{t_fmt}}"
                        f"{'R':{t_fmt}}"
                        f"{'P':{t_fmt}}"
                        f"{'F1':{t_fmt}}"
                        f"{'AP50':{t_fmt}}"
                        f"{'AP75':{t_fmt}}"
                        f"{'AP/AP5095':{t_fmt}}")
            for c, ap_c, f1_c, p_c, r_c in zip(*rest):
                name_c = cls_names[c]
                number_c = cls_number[c]
                LOGGER.info(f'{name_c:{t_fmt}}'
                            f'{number_c:{t_fmt}}'
                            f'{r_c[0]:{fmt}}'
                            f'{p_c[0]:{fmt}}'
                            f'{f1_c[0]:{fmt}}'
                            f'{ap_c[0]:{fmt}}'
                            f'{ap_c[5]:{fmt}}'
                            f'{sum(ap_c) / len(ap_c):{fmt}}')
    else:
        name_writer = ('P_50', 'R_50', 'F1_50', 'AP50', 'AP75', 'AP5095')
        value_writer = (0,) * 6
        WRITER.add_epoch_curve(writer, 'val_metrics', value_writer, name_writer, epoch)
        LOGGER.debug(f'{separate}')
        log_loss(when, epoch, loss_name, loss_all)
        LOGGER.debug(f'others is None, cls number is {cls_number}')
        LOGGER.debug(f'{separate}')


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
        log_name = func.__name__ if name is None else name

        @wraps(func)
        def wrapper(*args, **kwargs):
            LOGGER.info(f'Initializing {log_name}...')
            x = func(*args, **kwargs)
            LOGGER.info(f'Initialize {log_name} successfully')
            return x

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
            x = func(*args, **kwargs)
            LOGGER.info(f'Finished {log_name}')
            return x

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
