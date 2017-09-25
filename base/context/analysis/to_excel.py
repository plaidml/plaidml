import argparse
import numpy as np
import pandas as pd

import base.context.analysis.util as ca


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Vertex.AI profile data to Excel')
    parser.add_argument('-o', '--out', help='The Excel file to write', required=True)
    parser.add_argument('file', help='a binary profile data file')

    args = parser.parse_args()
    df = ca.cook(ca.events_to_dataframe(ca.read_eventlog(args.file))).astype(str, copy=False)
    df.rename(index=lambda x: str(x), inplace=True)
    df.to_excel(args.out)
