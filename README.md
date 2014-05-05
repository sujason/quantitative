## Welcome to Quantitative
This is the home of a project that seeks to leverage the vast array of tools available in Python to create an open source framework for the analysis of quantitative MRI methods.  Users are free to use it as is but contributions are also  welcome.  With new pulse sequences, signal equations, and framework improvements, we can build a tool together that becomes a new standard for the quantitative imaging community.
### Features
* Calculation of the Cramér-Rao Lower Bound via [automatic differentiation](http://en.wikipedia.org/wiki/Automatic_differentiation)
* Optimal experimental design using the Fisher information matrix and CRLB as a criterion

### Future
* Analysis of signals described by Bloch simulation and extended phase graph methods

### Current applications
* DESPOT1 (Deoni et al.)
* DESPOT2-FM (Deoni et al.)
* MPRAGE T1 mapping (Liu et al.)

### Requirements
The chief requirements for this package beyond NumPy and SciPy are [theano](http://deeplearning.net/software/theano/), [decorator](https://pypi.python.org/pypi/decorator/3.4.0), and [PyAutoDiff](https://github.com/LowinData/pyautodiff/tree/python2-ast).  The first two are available through PyPI but PyAutoDiff must be installed manually.


## Getting Started with Python
### Installation
The best way to install Python for scientific computing is with a pre-made environment or bundle of packages.  There are many competitors but two of the major players are: [Anaconda](https://store.continuum.io/cshop/anaconda/) and [Enthought Canopy](https://www.enthought.com/products/canopy/) (formerly Enthought Python Distribution).  These are both free.  Paid or academic licenses offer access to accelerated libraries or better support.  They help to assemble the most widely used open source community projects including NumPy, Matplotlib, SciPy, and IPython in a one-click install.

Most major Python packages are available from the central package index, [PyPI](https://pypi.python.org/pypi).  New packages can be pulled and installed from PyPI with the shell commands:
* `easy_install <package name>`
* `pip install <package name>`

Admin privileges are usually required.


### Learning
There is of course a heavy inertia to learning a new programming language, but I believe the effort is well worth it especially if you'd like to move beyond Matlab.  I found the following resources immensely helpful:
* [Dive Into Python](http://www.diveintopython.net) is a strong place to start.  The things learned here may not be immediately useful for scientific computing but they provide concepts and paradigms that will shape how you structure your future Python code.
* [Python Scientific Lecture Notes](http://scipy-lectures.github.io/index.html) gives a thorough introduction to NumPy, SciPy, and matplotlib.  This is where Python becomes a force for scientific inquiry.  It will feel right at home for Matlab and R users.
* [PyVideo](http://www.pyvideo.org) is an amazing resource that catalogs training sessions and talks from past PyCon events.  These typically cost thousands of dollars to attend in person.
* [IPython introduction](http://pycon-2012-notes.readthedocs.org/en/latest/ipython.html) teaches how to get started with  IPython as a convenient way to code and visualize data in a single integrated interface.
* [A matplotlib gallery](http://www.loria.fr/~rougier/coding/gallery/) on top of the [main one](http://matplotlib.org/gallery.html)
* [PEP-8 Programming Style](http://www.python.org/dev/peps/pep-0008/) is a useful set of guidelines for Python coding style.  Python provides a lot of freedom as a language and as programmers we must use exercise it responsibly.  By following these conventions we improve code clarity and give a base understanding that everyone starts from.

### Programming
* [SublimeText](http://www.sublimetext.com/) is a multi-platform text editor that has highlighting for many programming languages including Python.  It also happens to be written in Python.  Version 3 is the way to go.  Be sure to grab [Package Control](https://sublime.wbond.net/) to install plugins, like the excellent on-the-fly code-checking tool, [Anaconda](https://sublime.wbond.net/packages/Anaconda) (no relation to the previously mentioned environment).
* [PyCharm](http://www.jetbrains.com/pycharm/) is an excellent IDE with good tab-completion and debugger tools
* [Spyder](http://code.google.com/p/spyderlib/) mimics a MATLAB-like development enviroment.
* [IPython Notebook](http://ipython.org/) emulates a Mathematica-style environment and is also similar to MATLAB cells.  Run `ipython notebook --pylab=inline` from your shell.  This causes graphics to appear in the document as opposed to in a new window and also starts up Python with Pylab, i.e. NumPy, SciPy, and Matplotlib
* Never use `python` alone, use IPython and IPython QTConsole. They support tab completion and provide many shell commands to let you `ls` or `cd` around through directories.  Run `ipython qtconsole --pylab=inline` from the shell.

## Authors and Contributors
[Jason Su](sujason@stanford.edu), Stanford University (@sujason)

and You!
