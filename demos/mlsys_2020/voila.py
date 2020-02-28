import os
import sys
import subprocess


def main():
    subprocess.run(['voila'] + sys.argv[1:])  #, env=dict(os.environ))


if __name__ == '__main__':
    main()