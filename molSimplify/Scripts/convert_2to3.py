import os
import subprocess


def run_bash(filein):
    print(("Converting: ", filein))
    run_cmd = '2to3 -x unicode -x basestring -x raw_input -p -w %s' % filein
    q = subprocess.Popen(run_cmd, shell=True, stdout=subprocess.PIPE)
    ll = q.communicate()[0].decode("utf-8")
    print(('ll:', ll))


def main():
    basedir = '../'
    for dirpath, dir, files in os.walk(basedir):
        for f in sorted(files):
            if f.split(".")[-1] == 'py':
                run_bash(dirpath + '/' + f)
    print("Done.")


if __name__ == "__main__":
    main()
