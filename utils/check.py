r"""
Check utils.
Consist of all check function.
"""

__all__ = ['check_odd', 'check_even', 'check_0_1', 'check_only_one_set']


def check_odd(x: int):
    r"""
    Check int x, if it is odd number return True, otherwise return False.
    Args:
        x: int = integral number

    Return True or False
    """
    return abs(x) & 1 == 1


def check_even(x: int):
    r"""
    Check int x, if it is even number return True, otherwise return False.
    Args:
        x: int = integral number

    Return True or False
    """
    return abs(x) & 1 == 0


def check_0_1(x):
    r"""
    Check x whether it is in interval (0, 1)
    Args:
        x: any number = number

    Return True or False
    """
    if 0 <= x <= 1:
        return True
    else:
        return False


def check_only_one_set(one, two):
    r"""
    Check only one of them can be True(0 represents True which means it has set).
    AttributeError(f'Only one of {one} and {two} can be set, please reset it')
    Args:
        one: = any
        two: = any

    Return True or False
    """
    one = bool(one) if str(one) != '0' else True
    two = bool(two) if str(two) != '0' else True
    if (one & two) or (not one | two):
        return False
    else:
        return True


def _test():
    r"""test all function or class in check.py"""
    check_odd(1)
    check_odd(2)
    check_even(1)
    check_even(2)
    check_only_one_set(0, 0)
    check_only_one_set(True, True)
    check_only_one_set(False, False)
    check_only_one_set(True, False)
    check_only_one_set(False, True)


if __name__ == '__main__':
    _test()
