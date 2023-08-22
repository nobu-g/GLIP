from contextlib import contextmanager


@contextmanager
def nullcontext(enter_result=None, **kwargs):
    yield enter_result


try:
    from torch.cuda.amp import GradScaler, autocast, custom_bwd, custom_fwd
except:
    print("[Warning] Library for automatic mixed precision is not found, AMP is disabled!!")
    GradScaler = nullcontext
    autocast = nullcontext
    custom_fwd = nullcontext
    custom_bwd = nullcontext
