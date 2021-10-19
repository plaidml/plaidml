#!/usr/bin/env python
import click
import os
import sh


@click.command()
@click.argument('omz_network')
@click.option('-s', '--skip', is_flag=True, help='skip downloading / optimizing')
@click.option('-bs', '--batch_size', default=1, help='minibatch size')
@click.option('-nt',
              '--num_threads',
              type=int,
              help='number of threads to use, defaults to #hyperthreads')
@click.option('-p', '--precision', default='FP32')  # TODO options
@click.option('--cache_dir', default='~/.cache', help='cache directory')
@click.option('--model_dir', default='~/.cache/omz_models', help='model directory')
@click.option('--report_dir', default='reports', help='cache directory')
def benchmark_app(omz_network, skip, batch_size, num_threads, precision, cache_dir, model_dir,
                  report_dir):
    """Downloads models from openmodelzoo and runs them in benchmark_app, optionally creating a report
  
  \b
  Examples:
    ./benchmark_ov resnet-50-tf
    ./benchmark_ov -bs 16 -nt 28 -p FP16 bert-base-ner
  """
    model_dir = os.path.expanduser(os.path.normpath(model_dir))
    cache_dir = os.path.expanduser(os.path.normpath(cache_dir))
    if not skip:
        sh.mkdir('-p', model_dir)
        print(
            'Downloading {0}'.format(omz_network))  #TODO(brian): make this spit out output easily
        sh.omz_downloader('--cache_dir={0}'.format(cache_dir), name=omz_network, o=model_dir)
        print('Running model optimizer on {0}'.format(omz_network))
        sh.omz_converter(name=omz_network, d=model_dir, o=model_dir, precision=precision)
    else:
        print('Skipping download & conversion')

    print('Benchmarking...')
    model_path = os.path.join(model_dir, 'public', omz_network, precision, omz_network + '.xml')
    ba = sh.Command('benchmark_app')
    ba = ba.bake(m=model_path, api='sync', t='15', b=batch_size, _long_prefix='-', _long_sep=' ')
    if num_threads:
        ba = ba.bake(nstreams=num_threads)
    for line in ba(_iter=True):
        print(line, end='')


if __name__ == '__main__':
    benchmark_app()
