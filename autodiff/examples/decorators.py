from autodiff import function, gradient


@function
def f(x):
    return x ** 2


@gradient
def g(x):
    return x ** 2


if __name__ == '__main__':
    print 'x = 20'
    print 'f(x) = {0}'.format(f(20.0))
    print 'f\'(x) = {0}'.format(g(20.0))
