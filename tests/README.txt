Instructions for Installing and Running Test Cases for molSimplify
      by Fang Liu  10/24/2017
      edited by Chenru Duan 10/28/2021


I.  The structure of the repo

    |- molSimplify
    |      |- __init__.py
    |      |- Scripts
    |      |- Classes
    |      |_ ...
    |
    |- tests
    |      |-test_case1.py
    |      |-...
    |      |-test_caseN.py
    |       |- inputs
    |      |     |- case1.in
    |      |     |- ...
    |      |     |_ caseN.in
    |      |_ refs
    |            |- case1.xyz
    |            |- case1.report
    |            |_ ...
    |_ setup.py

    Reason for using this structure, please refer to: https://blog.ionelmc.ro/2014/05/25/python-packaging/#the-structure
    Notice that pytest automatically finds test cases by looking at python files
    named with
          test_XXX.py  XXX_test.py

    Under this structure, the test files MUST have unique names

II. Add a test case
  1. Name the test input and output files in a consistent way:
     yourTestCase.in,  yourTestCase.xyz,  yourTestCase.report

  2. Put yourTestCase.in under: tests/inputs/
     Put yourTestCase.xyz,  yourTestCase.report  under: tests/refs/

  3. Create a test python script with the template shown below.
     threshMLBL: the threshold for checking metal-ligand bondlength and defaults to 0.1 A
     threshLG: tolerance for RMSD comparison of Ligand Geometries, and defaults to 0.1 A.
     threshOG: tolerance for RMSD comparison of Overall Geometries, and defaults to 2.0 A
     If you are adding a small test case (e.g. hexachloride), consider
     reducing this parameter. Otherwise it is likely fine the way it is.

############ test_yourTestCase.py  #########
import helperFuncs as hp

def test_example_1(tmpdir):
    testName="yourTestCase"
    threshMLBL = 0.1 #Change this value for your need
    threshLG = 0.1 #Change this value for your need
    threshOG = 2.0 #Change this value for you need
    [passNumAtoms,passMLBL,passLG,passOG,pass_report] = hp.runtest(tmpdir,testName,threshMLBL,threshLG,threshOG)
    assert passNumAtoms
    assert passMLBL
    assert passLG
    assert passOG
    assert pass_report

#############################################

III. Run Test Cases
  1. Tests will be automatically run by GitHub CI. However, it's highly recommended
     that one runs the tests locally before committing or push to repository to
     avoid breaking things

  2. Here are some different ways to run the tests. Assume that the we are at the
     root directory of the repository
     (1) make use of the setup.py:
         python setup.py test
     (2) Use the pytest command:
         pytest

     These will go through all test cases and report whether each is passed. If
     a test fails, pytest will automatically trace back to the function where
     it breaks. Standard output of molSimplify will not be printed unless the
     test fails (then the output will be captured by pytest)

     If one wants to see all the standard output on the screen, we can try:
     (1) Run pytest without capture:
         pytest -s

     (2) Manually run test case under the tests/ directory
         py.test -k <keyword-of-your-test>
         For example, to run all the tests for generation of tetrahedral complexes, do:
         py.test -k tetrahedral
