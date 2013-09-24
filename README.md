## Welcome to Quantitative
This is the future home of a project that seeks to leverage the vast array of tools available in Python to create an open source framework for the reconstruction and analysis of quantitative MRI methods.  Users are free to use it as is or even to implement own mapping techniques and automatically inherit the framework's other features.
### Features in progress include:
* Automated multi-subject reconstruction of maps
* Image support for DICOM, NIFTI, etc. via [nibabel](http://nipy.sourceforge.net/nibabel/)
* Parallelized computation at the voxel, slice, or subject level
* Resume from crash
* GPU accelerated Stochastic Region of Contraction solver
* Calculation of the Cramer-Rao Lower Bound via [automatic differentiation](https://github.com/LowinData/pyautodiff)

### Implementations in progress:
* DESPOT1 (Deoni et al.)
* DESPOT2-FM (Deoni et al.)
* mcDESPOT (Deoni et al.)
* MPRAGE T1 mapping (Liu et al.)

## Getting Started with Python
### Installation
The best way to install Python for scientific computing is with a pre-made environment or bundle of packages.  There are many competitors but two of the major players are: [Enthought Canopy](https://www.enthought.com/products/canopy/) (formerly Enthought Python Distribution) and [Anaconda](https://store.continuum.io/cshop/anaconda/).  These are both free.  Paid or academic licenses offer access to accelerated libraries or better support.  They help to assemble the most widely used open source community projects including NumPy, Matplotlib, SciPy, and IPython in a one-click install.

Most major Python packages are available from the central package index, [PyPI](https://pypi.python.org/pypi).  New packages can be pulled and installed from PyPI with the shell commands:
* `$ easy_install <package name>`
* `$ pip install <package name>`

Admin privileges are usually required.


### Learning
There is of course a heavy inertia to learning a new programming language, but I believe the effort is well worth it especially if you'd like to move beyond Matlab.  I found the following resources immensely helpful:
* [Dive Into Python](http://www.diveintopython.net)
* [PyVideo](http://www.pyvideo.org) - an amazing resource that catalogs training sessions and talks from past PyCon events.  These typically cost thousands of dollars to attend in person.
* SciPy tutorials
* IPython tutorials
* [PEP-8 Programming Style](http://www.python.org/dev/peps/pep-0008/)

### Programming
* PyCharm is an excellent IDE with good tab-completion and debugger tools
* Spyder attempts to mimic MATLAB
* IPython Notebook emulates a Mathematica-style environment and is also similar to MATLAB cells.  Run "ipython notebook --pylab=inline" from your shell.  This causes graphics to appear in the document as opposed to in a new window and also starts up Python with Pylab, i.e. NumPy, SciPy, and Matplotlib
* Never use "python" alone, use IPython and IPython QTConsole. They support tab completion and provide many shell commands to let you "ls" or "cd" around through directories.  Run "ipython qtconsole --pylab=inline" from the shell.

## Authors and Contributors
[Jason Su](sujason@stanford.edu), Stanford University (@sujason)
You!
