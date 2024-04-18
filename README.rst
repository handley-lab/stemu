=====================================================
stemu: s(t) emulation of smooth functions by stacking
=====================================================
:stemu: s(t) emulation of smooth functions by stacking 
:Author: Harry Bevins & Will Handley
:Version: 0.0.0
:Homepage: https://github.com/handley-lab/stemu
:Documentation: http://stemu.readthedocs.io/

.. image:: https://github.com/handley-lab/stemu/actions/workflows/unittests.yaml/badge.svg?branch=master
   :target: https://github.com/handley-lab/stemu/actions/workflows/unittests.yaml?query=branch%3Amaster
   :alt: Unit test status
.. image:: https://github.com/handley-lab/stemu/actions/workflows/build.yaml/badge.svg?branch=master
   :target: https://github.com/handley-lab/stemu/actions/workflows/build.yaml?query=branch%3Amaster
   :alt: Build status
.. image:: https://codecov.io/gh/handley-lab/stemu/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/handley-lab/stemu
   :alt: Test Coverage Status
.. image:: https://readthedocs.org/projects/stemu/badge/?version=latest
   :target: https://stemu.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. image:: https://badge.fury.io/py/stemu.svg
   :target: https://badge.fury.io/py/stemu
   :alt: PyPi location
.. image:: https://anaconda.org/handley-lab/stemu/badges/version.svg
   :target: https://anaconda.org/handley-lab/stemu
   :alt: Conda location
.. image:: https://zenodo.org/badge/705730277.svg
   :target: https://zenodo.org/doi/10.5281/zenodo.10009816
   :alt: Permanent DOI for this release
.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://github.com/handley-lab/stemu/blob/master/LICENSE
   :alt: License information


A repository for emulation of smooth functions by stacking

UNDER CONSTRUCTION


Features
--------

Installation
------------

``stemu`` can be installed via pip

.. code:: bash

    pip install stemu

via conda

.. code:: bash

    conda install -c handley-lab stemu

or via the github repository

.. code:: bash

    git clone https://github.com/handley-lab/stemu
    cd stemu
    python -m pip install .

You can check that things are working by running the test suite:

.. code:: bash

    python -m pytest
    black .
    isort --profile black .
    pydocstyle --convention=numpy stemu


Dependencies
~~~~~~~~~~~~

Basic requirements:

- Python 3.6+
- `anesthetic <https://pypi.org/project/anesthetic/>`__

Documentation:

- `sphinx <https://pypi.org/project/Sphinx/>`__
- `numpydoc <https://pypi.org/project/numpydoc/>`__

Tests:

- `pytest <https://pypi.org/project/pytest/>`__

Documentation
-------------

Full Documentation is hosted at `ReadTheDocs <http://stemu.readthedocs.io/>`__.  To build your own local copy of the documentation you'll need to install `sphinx <https://pypi.org/project/Sphinx/>`__. You can then run:

.. code:: bash

    python -m pip install ".[all,docs]"
    cd docs
    make html

and view the documentation by opening ``docs/build/html/index.html`` in a browser. To regenerate the automatic RST files run:

.. code:: bash

    sphinx-apidoc -fM -t docs/templates/ -o docs/source/ stemu/

Citation
--------

If you use ``stemu`` to generate results for a publication, please cite
as: ::

   H.T.J. Bevins, W.J. Handley, A. Fialkov, E. de Lera Acedo, K. Javid. globalemu: a novel and robust approach for emulating the sky-averaged 21-cm signal from the cosmic dawn and epoch of reionization, DOI: 10.1093/mnras/stab2737, Mon.Not.Roy.Astron.Soc. 508 (2021) 2, 2923-2936

or using the BibTeX:

.. code:: bibtex

	@article{Bevins:2021eah,
		author = "Bevins, H. T. J. and Handley, W. J. and Fialkov, A. and Acedo, E. de Lera and Javid, K.",
		title = "{globalemu: a novel and robust approach for emulating the sky-averaged 21-cm signal from the cosmic dawn and epoch of reionization}",
		eprint = "2104.04336",
		archivePrefix = "arXiv",
		primaryClass = "astro-ph.CO",
		doi = "10.1093/mnras/stab2737",
		journal = "Mon. Not. Roy. Astron. Soc.",
		volume = "508",
		number = "2",
		pages = "2923--2936",
		year = "2021"
	}


Contributing
------------
There are many ways you can contribute via the `GitHub repository <https://github.com/handley-lab/stemu>`__.

- You can `open an issue <https://github.com/handley-lab/stemu/issues>`__ to report bugs or to propose new features.
- Pull requests are very welcome. Note that if you are going to propose major changes, be sure to open an issue for discussion first, to make sure that your PR will be accepted before you spend effort coding it.
- Adding models and data to the grid. Contact `Will Handley <mailto:wh260@cam.ac.uk>`__ to request models or ask for your own to be uploaded.


Questions/Comments
------------------
