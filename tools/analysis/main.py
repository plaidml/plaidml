import argparse
import os
import subprocess


def _valid_path(path):
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError('File not found: {!r}. Use absolute paths.'.format(path))
    return path


def main():

    parser = argparse.ArgumentParser(
        prog='analysis',
        description='Run analysis notebook (more help in Analysis.ipynb)',
        usage='./t2 run //tools/analysis -- [options]')
    parser.add_argument('--ip', help='IP address to bind to', default='127.0.0.1')
    parser.add_argument('--export',
                        nargs='?',
                        const='profiling_notebook.html',
                        default=None,
                        metavar='FILENAME',
                        help='Save notebook via nbconvert instead of launching it')
    parser.add_argument('--export-timeout',
                        type=int,
                        default=480,
                        metavar='T',
                        help='Notebook export times out if any cell runs for more than T seconds')
    parser.add_argument('file',
                        help='The eventlog to process',
                        nargs='?',
                        type=_valid_path,
                        default=os.path.join(os.path.expanduser('~'), 'eventlog.gz'))

    args = parser.parse_args()

    env = os.environ.copy()
    env['PLAIDML_EVENTLOG_FILENAME'] = args.file
    notebook = os.path.join(os.getcwd(), 'tools', 'analysis', 'Analysis.ipynb')
    if not os.path.exists(notebook):
        with open(os.path.join(os.getcwd(), 'MANIFEST'), 'r') as manifest:
            for line in iter(manifest):
                (key, value) = line.split(' ', 1)
                if key == 'com_intel_vertexai/tools/analysis/Analysis.ipynb':
                    notebook = value.rstrip('\n')
                    break
    if args.export:
        notebook = os.path.abspath(os.path.join('tools', 'analysis', 'Analysis.ipynb'))
        timeout_arg = '--ExecutePreprocessor.timeout={}'.format(args.export_timeout)
        cmd = ['jupyter', 'nbconvert', '--execute', '--output', args.export, timeout_arg, notebook]
        subprocess.check_call(cmd, env=env)
    else:
        conda_env = os.getenv('CONDA_DEFAULT_ENV')
        python = os.path.join(conda_env, 'bin', 'python')
        jupyter = os.path.join(conda_env, 'bin', 'jupyter')
        cmd = [python, jupyter, 'notebook', '--ip', args.ip, notebook]
        devnull = open(os.devnull, 'w')
        try:
            subprocess.call(cmd, env=env, stdin=devnull)
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
