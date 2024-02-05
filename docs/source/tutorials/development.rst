===========
Development
===========

How to contribute
=================

Development of new features should be performed in separate branches
(and/or forks) of the repository in order to ensure that:

- the master branch never contains unfinished/unvalidated code
- unnecessary build notifications to the slack channel are avoided

Opening a pull request for the finished feature allows the code maintainers to
review new code before merging into main. This workflow not only ensures a
certain level of code quality but also encourages discussions about
implementation details and best practices which can be a powerful method of
knowledge exchange for both involved parties.

Automated testing
=================

As a part of the continous integration (CI) pipeline all new code submissions
are valided using a pytest test-suite. MolSimplify currently uses Github
actions to run these checks (see status
`here <https://github.com/hjkgrp/molSimplify/actions/workflows/CI.yaml>`_).

Ideally all code contributions should also contain a simple test case to ensure
that the new feature is not broken by future updates to the code base.

Code quality
============

While measures of code quality (and aesthetics) are highly subjective, a common
standard is useful for the collaborative development of open-source code. Most
Python projects (including molSimplify) follow the recommendations outlined in
`PEP8 <https://peps.python.org/pep-0008/>`_.

Linter
------

The easiest way to ensure your contributions conform to
`PEP8 <https://peps.python.org/pep-0008/>`_ is to install a linter such as
`flake8 <https://flake8.pycqa.org>`_ for your IDE of choice.

Some basic measures of code quality are enforced using flake8 as a part of the
Github actions CI pipeline. The following flake8 error codes will result in a
failed build (where * is a placeholder matching all subclasses of the
errors):

* E9** syntax, io, or indentation error
* F63* syntax error in assertion, comparison, or print
* F7** syntax error in loops or functions
* F82* undefined variables

Pre-commit hooks
----------------

In order to automate the process of code lintering molSimplify also provides a
configuration file for a `pre-commit <https://pre-commit.com/>`_ hook. After
`installing the pre-commit package <https://pre-commit.com/#install>`_
(using pip, conda, or your package manager of choice) the configuration file
can be installed using the command::

    pre-commit install
