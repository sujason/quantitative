def constant(val):
    """
    Returns a new and unique variable to be used as a constant. Also adds the
    id of that variable to the _constants set, so that Context objects can tell
    if a variable should be treated as constant, or not.  Constants are not
    shadowed by symbolic variables, and provide a way to safely use functions
    like range or iterators.

    This should not work:

        @function
        def fn(x, a):
            x.sum(axis=a)
            ...

    But this should:

        @function
        def fn(x, a):
            x.sum(axis=constant(a))
            ...

    """
    return val


def tag(obj, tag):
    """
    Tags an object with a certain keyword. By default, all symbolic objects are
    associated with the id of the Python object they shadow in a Context's
    svars dict.  By calling this function on an object and providing a
    (hashable) tag, users can more easily access the symbolic representation of
    any objects that might only be created during function execution.

    Example:

        @function
        def fn(x):
            y = tag(x + 2, 'y')
            z = y * 3
            return z

        fn.s_vars['y'] # returns the symbolic version of y

    """
    return obj
