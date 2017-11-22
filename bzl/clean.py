import os


def main():
    for root, dirnames, filenames in os.walk('.'):
        for filename in filenames:
            if filename in ['BUILD', 'WORKSPACE']:
                os.unlink(os.path.join(root, filename))


if __name__ == '__main__':
    main()
