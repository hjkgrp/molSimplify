import glob
import os
import datetime


def check_existing_tests():
    tests = []
    for fname in glob.glob("test_*.py"):
        tests.append(fname.replace(".py", "").replace("test_", ""))
    print("Existing tests: ", tests)
    return tests


def read_inputs():
    inputs = []
    for fname in glob.glob("inputs/*.in"):
        name = os.path.basename(fname)
        if "_noff" not in name:
            inputs.append(name.replace('.in', ''))
    print("Independent inputs: ", inputs)
    return inputs


def check_refs(inputs):
    testRefDict = dict()
    for name in inputs:
        testRefDict[name] = [False, False, False, False]
    for fname in glob.glob("refs/*.xyz"):
        name = os.path.basename(fname).replace('.xyz', '')
        if "_noff" not in name:
            if name in testRefDict:
                testRefDict[name][0] = True
        else:
            if name.replace("_noff", '') in testRefDict:
                testRefDict[name.replace("_noff", '')][2] = True
    for fname in glob.glob("refs/*.report"):
        name = os.path.basename(fname).replace('.report', '')
        print("report name:", name)
        if "_noff" not in name:
            if name in testRefDict:
                testRefDict[name][1] = True
        else:
            if name.replace("_noff", "") in testRefDict:
                testRefDict[name.replace("_noff", "")][3] = True
    return testRefDict


def writeBasicTest(name):
    testfile = "test_"+name+".py"
    content = '''import helperFuncs as hp

def test_TESTNAME(tmpdir):
    testName="TESTNAME"
    threshMLBL = 0.1
    threshLG =  1.0
    threshOG = 2.0
    [passNumAtoms,passMLBL,passLG,passOG,pass_report,pass_qcin] = hp.runtest(tmpdir,testName,threshMLBL,threshLG,threshOG)
    assert passNumAtoms
    assert passMLBL
    assert passLG
    assert passOG
    assert pass_report
'''
    with open(testfile, "w") as f:
        f.write(content.replace("TESTNAME", name))


def writeNoffTest(name):
    testfile = "test_"+name+".py"

    content = '''
def test_TESTNAME_No_FF(tmpdir):
    testName="TESTNAME"
    threshMLBL = 0.1
    threshLG =  1.0
    threshOG = 2.0
    [passNumAtoms,passMLBL,passLG,passOG,pass_report,pass_qcin] = hp.runtestNoFF(tmpdir,testName,threshMLBL,threshLG,threshOG)
    assert passMLBL
    assert passLG
    assert passOG
    assert pass_report
'''
    with open(testfile, "a") as f:
        f.write(content.replace("TESTNAME", name))


def generateTests():
    inputs = read_inputs()
    testRefDict = check_refs(inputs)
    existingTests = check_existing_tests()
    countNewTests = 0
    for test in testRefDict:
        if test not in existingTests:
            print("-"*80)
            print("New test found! ", test)
            countNewTests += 1
            if testRefDict[test][0] == True and testRefDict[test][1] == True:
                print("Basic Reference results found for test", test)
                writeBasicTest(test)
                if testRefDict[test][2] == True and testRefDict[test][3] == True:
                    print("Reference results found for no-forcefield test", test)
                    writeNoffTest(test)
                    print(
                        "Generated test python script with both basic and no-forcefield mode for ", test)
                else:
                    print(
                        "Generated test python script with ONLY basic mode for ", test)
            else:
                if testRefDict[test][0] == False:
                    print("Basic Reference geometry missing for ", test)
                if testRefDict[test][1] == False:
                    print("Basic Reference report missing for ", test)
                print("WARNING: test python script won't be generated for ", test)
            print("-"*80)
    if countNewTests == 0:
        print("All test files has already been generated. No new tests found.")
    else:
        print(countNewTests, " new tests found and processed!")


if __name__ == "__main__":
    generateTests()
