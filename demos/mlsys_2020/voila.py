import sys
import subprocess


def main():
    subprocess.run(['voila'] + sys.argv[1:])


if __name__ == '__main__':
    main()