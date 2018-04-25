import functools
import inspect
import traceback


def except_print(f=None, no_wrap=False):
    if f is None:
        return functools.partial(except_print, no_wrap=no_wrap)

    @functools.wraps(f)
    def res(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except BaseException:
            traceback.print_stack()
            traceback.print_exc()
            raise

    if not no_wrap:
        formatted_args = inspect.formatargspec(*inspect.getfullargspec(f))
        fndef = 'lambda {}: res{}'.format(formatted_args.lstrip('(').rstrip(')'), formatted_args)
        return eval(fndef, {'res': res})
    else:
        return res