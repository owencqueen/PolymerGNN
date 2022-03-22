import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--values', type=str, nargs = '+')
args = parser.parse_args()
print('Args as list', args.values)