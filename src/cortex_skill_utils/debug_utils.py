from typing import Callable, Any

from logging import getLogger

LOG = getLogger(__name__)


class StopEarlyException(Exception):
    pass


def runit(func: Callable, **kwargs):
    try:
        ret = func(**kwargs)
        return ret
    except StopEarlyException as e:
        ret = tuple(e.args)
        if len(ret) == 1:
            return ret[0]
        return ret
    except Exception as e:
        LOG.info('Exception:', e)
        raise


def retval(*args: Any):
    print("!! stopping run early...")
    raise StopEarlyException(*args)
