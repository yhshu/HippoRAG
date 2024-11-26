from argparse import ArgumentTypeError


def string_to_bool(v):
    """
    From https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    @param v:
    @return:
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )
