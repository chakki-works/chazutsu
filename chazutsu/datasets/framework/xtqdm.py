def get_xtqdm():
    from tqdm import tqdm
    from tqdm import tqdm_notebook

    if is_jupyter():
        return tqdm_notebook
    else:
        return tqdm


def is_jupyter():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True
        elif shell == "TerminalInteractiveShell":
            return False
        else:
            return False
    except NameError:
        return False


xtqdm = get_xtqdm()
