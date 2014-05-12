"""
Imports the core classes for performing optimal experimental design using the
Fisher information matrix/Cramer-Rao Lower Bound.
"""
__version__ = 0.01


from quantitative.higherad import HigherAD
from quantitative.crlb import calc_crlb
from quantitative.opt_helper import MultiOptimizationHelper
from quantitative.multistart import MultiStart


if __name__ == '__main__':
	# TODO unit tests on all imported modules