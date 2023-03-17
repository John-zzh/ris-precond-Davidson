import os
import yaml
import re
import numpy as np
import argparse

def gen_args():
    parser = argparse.ArgumentParser(description='Davidson')
    parser.add_argument('-f', '--yaml', type=str,   default=None,  help='the yaml file to read')
    args = parser.parse_args()
    return args

args = gen_args()


def main():

    with open(args.yaml, 'r') as y:
        tmp = yaml.safe_load(y)

        func = tmp['functional']
        n_vec = tmp['N_states']
        i_p = tmp['initial_guess'][0] + tmp['preconditioner'][0]
        mol = tmp['molecule']
        N_itr = tmp['N_itr']
        N_mv = tmp['N_mv']
        wall_t = tmp['wall time']
        ip_t = tmp['precondition time'] + tmp['initial guess time']
        ip_t_p = 100*ip_t/tmp['wall time']
        RMSE = tmp['initial-final difference'] if tmp['initial-final difference'] else 0
        overalp = tmp['initial-final overlap']

        print(f"{func:<6} {n_vec:<2} {i_p:<4} {mol:<30} {N_itr:<5} {N_mv:<5}  {wall_t:<7.0f} {ip_t:<7.1f} {ip_t_p:<5.1f} {RMSE:<5.3f}  {overalp:<5.4f}")

if __name__ == "__main__":
    if args.yaml:
        main()
