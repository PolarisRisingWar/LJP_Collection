import argparse
import pickle as pk

parser = argparse.ArgumentParser()
parser.add_argument('--law_file')
parser.add_argument('--output_file')
args = parser.parse_args()

k={}
with open(args.law_file) as f:
    l=f.readlines()
    for i in range(len(l)):
        item=l[i]
        k[item.strip()]=i

pk.dump(k,open(args.output_file,'wb'))