#!/usr/bin/env python3

import argparse
import os
import sys
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm

default_results_dir = './results'
default_output_file = './stats.csv'
default_instrs = 500
default_printing_period_instrs = 10

default_seed_file = './scripts/seeds.txt'

# No prefetcher
default_lru_binary = 'bin/hashed_perceptron-no-no-no-no-lru-{n_cores}core'
default_hawkeye_binary = 'bin/hashed_perceptron-no-no-no-no-hawkeye_simple-{n_cores}core'

baseline_names = ['lru', 'hawkeye']
baseline_fns = ['lru', 'hawkeye_simple']
baseline_binaries = [default_lru_binary, default_hawkeye_binary]

help_str = {
'help': '''usage: {prog} command [<args>]

Available commands:
    build            Builds ChampSim binaries
    run              Runs ChampSim on specified traces
    eval             Parses and computes metrics on simulation results
    help             Displays this help message. Command-specific help messages
                     can be displayed with `{prog} help command`
'''.format(prog=sys.argv[0]),

'build': '''usage: {prog} build <target> [-c / --cores <core-count-list>]

Description:
    {prog} build <target>
        Builds <target> ChampSim binaries where <target> is one of:

            all            Builds all the binaries
            lru            Builds just the lru binary that uses a least-recently-used eviction policy
            hawkeye        Builds just the Hawkeye byinary that uses a Hawkeye eviction policy

Options:
    -c / --cores <core-count-list>
        Specifies a list of cores to build ChampSim variants. A single core
        version will always be built, but additional versions (e.g. 2-core / 4-core)
        can be listed here (e.g. using -c 2 4). The ChampSim script is tested up
        to 8 cores.

Notes:
    Barring updates to the GitHub repository, this will only need to be done once.
'''.format(prog=sys.argv[0]),

'run': '''usage: {prog} run <execution-traces> [-c / --cores <num-cores>] [--results-dir <results-dir>]
                            [--num-instructions <num-instructions>] [--stat-printing-period <num-instructions>]
                            [--seed-file <seed-file>]

Description:
    {prog} run <execution-traces>
        Runs the base ChampSim binary on the specified execution trace(s). If using
        a multi-core setup, must provide <cores> traces.

Options:
    -c / --cores <num-cores>
        The number of cores that ChampSim will be simulating. Must provide a <cores>
        length list of execution traces to the script. By default, one core is used.
        
    -t / --targets <list-of-targets>
        List of targets to run. By default, it will run all targets: {baseline_names}.
        
    --results-dir <results-dir>
        Specifies what directory to save the ChampSim results file in. This
        defaults to `{default_results_dir}`.

    --num-instructions <num-instructions>
        Number of instructions to run the simulation for. Defaults to
        {default_instrs}M instructions 

    --stat-printing-period <num-instructions>
        Number of instructions to simulate between printing out statistics.
        Defaults to {default_printing_period_instrs}M instructions.

    --seed-file <seed-file>
        Path to seed file to load for ChampSim evaluation. Defaults to {seed_file}.
'''.format(prog=sys.argv[0], default_results_dir=default_results_dir,
    baseline_names=baseline_names,
    default_instrs=default_instrs,
    default_printing_period_instrs=default_printing_period_instrs,
    seed_file=default_seed_file),

'eval': '''usage: {prog} eval [--results-dir <results-dir>] [--output-file <output-file>]

Description:
    {prog} eval
        Runs the evaluation procedure on the ChampSim output found in the specified
        results directory and outputs a CSV at the specified output path.

Options:
    --results-dir <results-dir>
        Specifies what directory the ChampSim results files are in. This defaults
        to `{default_results_dir}`.

    --output-file <output-file>
        Specifies what file path to save the stats CSV data to. This defaults to
        `{default_output_file}`.

Note:
    To get stats comparing performance to a no-prefetcher baseline, it is necessary
    to have run the base ChampSim binary on the same execution trace.

    Without the base data, relative performance data comparing MPKI and IPC will
    not be available and the coverage statistic will only be approximate.
'''.format(prog=sys.argv[0], default_results_dir=default_results_dir, default_output_file=default_output_file),
}



"""
Build
"""
def build_command():
    """Build command
    """
    if len(sys.argv) < 3:
        print(help_str['build'])
        exit(-1)
        
    parser = argparse.ArgumentParser(usage=argparse.SUPPRESS, add_help=False)
    parser.add_argument('target', default=None)
    parser.add_argument('-c', '--cores', type=int, nargs='+', default=[])
    args = parser.parse_args(sys.argv[2:])
    
    print('Building ChampSim versions using args:')
    print('    Target:', args.target)
    print('    Cores :', args.cores)
    
    if args.target not in ['all'] + baseline_names:
        print('Invalid build target')
        exit(-1)

    # Build ChampSims with different replacement policies.
    cores = set([1] + args.cores)

    for name, fn in zip(baseline_names, baseline_fns):
        if args.target == 'all' or name in args.target:
            for c in cores:
                print(f'=== Building {name} ChampSim binary ({fn}), {c} core{"s" if c > 1 else ""} ===')
                os.system(f'./build_champsim.sh hashed_perceptron no no no no {fn} {c}')

                
            
"""
Run
"""
def run_command():
    """Run command
    """
    if len(sys.argv) < 3:
        print(help_str['run'])
        exit(-1)

    parser = argparse.ArgumentParser(usage=argparse.SUPPRESS, add_help=False)
    parser.add_argument('execution_traces', nargs='+', type=str, default=None)
    parser.add_argument('-c', '--cores', type=int, default=1)
    parser.add_argument('-t', '--targets', nargs='+', type=str, default=baseline_names)
    parser.add_argument('--results-dir', default=default_results_dir)
    parser.add_argument('--num-instructions', default=500) #None) #default_spec_instrs if execution_trace[0].isdigit() else default_gap_instrs)
    parser.add_argument('--stat-printing-period', default=default_printing_period_instrs)
    #parser.add_argument('--seed-file', default=default_seed_file)
    parser.add_argument('--name', default='from_file')

    args = parser.parse_args(sys.argv[2:])
    assert len(args.execution_traces) == args.cores, f'Provided {len(args.execution_traces)} traces for a {args.cores} core simulation.'
    
    execution_traces = args.execution_traces

    # Generate results directory
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir, exist_ok=True)
        
    # Generate names for this permutation. (trace names without extensions, joined by hyphen)
    base_traces = '-'.join(
        [''.join(os.path.basename(et).split('.')[:-2]) for et in execution_traces]
    ) 
       
    for name, binary in zip(baseline_names, baseline_binaries):
        binary = binary.format(n_cores = args.cores)
        base_binary = os.path.basename(binary)
        
        # Check if we should actually run this baseline
        if name not in args.targets:
            print(f'Skipping {name} ({binary})')
            continue
        
        if not os.path.exists(binary):
            print(f'{name} ChampSim binary not found, (looked for {binary})')
            exit(-1)

        cmd = '{binary} -stat_printing_period {period}000000 -simulation_instructions {sim}000000 -traces {trace} > {results}/{base_traces}-{base_binary}.txt 2>&1'.format(
            binary=binary,
            period=args.stat_printing_period,
            sim=args.num_instructions,
            trace=' '.join(execution_traces), 
            results=args.results_dir, 
            base_traces=base_traces,
            base_binary=base_binary
        )

        print('Running "' + cmd + '"')
        os.system(cmd)
        
        
"""
Eval
"""
def get_traces_per_cpu(path):
    """Read a single ChampSim output file and get the traces on each CPU.
    """
    traces = {}
    with open(path, 'r') as f:
        for line in f:
            if 'CPU' in line and 'runs' in line:
                core = int(line.split()[1])
                #traces[core] = os.path.basename(line.split()[-1]).split('.')[0] # Trace name - TODO check this works for all traces.
                traces[core] = os.path.basename(line.split()[-1])  # File name
                #traces[core] = line.split()[-1] # Full path to file
    return traces
    
    
def read_file(path, cache_level='LLC'):
    """Read a single ChampSim output file and parse the results.
    """
    #expected_keys = ('trace', 'ipc', 'total_miss', 'useful', 'useless', 'uac_correct', 'iss_prefetches', 'load_miss', 'rfo_miss', 'kilo_inst')
    expected_keys = ('trace', 'ipc', 'kilo_inst', 'load_miss', 'rfo_miss', 'total_miss')
    
    
    #data = defaultdict(lambda: defaultdict(int)) # Indexed by core -> feature
    data = defaultdict(dict)
    
    # Build trace list
    data['trace'] = get_traces_per_cpu(path)
    data['is_homogeneous'] = len(set(data['trace'].values())) == 1
    
    # Build other features
    with open(path, 'r') as f:
        for line in f:
            # Finished CPU indicators
            if 'Finished CPU' in line:
                core = int(line.split()[2])
                data['ipc'][core] = float(line.split()[9])
                data['kilo_inst'][core] = int(line.split()[4]) / 1000
                
            # Region of interest statistics
            if 'CPU' in line and line.split()[0] == 'CPU':
                core = int(line.split()[1])
            if cache_level not in line:
                continue
            line = line.strip()
            if 'LOAD' in line:
                data['load_miss'][core] = int(line.split()[-1])
            elif 'RFO' in line:
                data['rfo_miss'][core] = int(line.split()[-1])
            elif 'TOTAL' in line:
                data['total_miss'][core] = int(line.split()[-1])
            # elif 'USEFUL' in line:
            #     data['useful'][core] = int(line.split()[-6])
            #     data['useless'][core] = int(line.split()[-4])
            #     data['uac_correct'][core] = int(line.split()[-1])
            #     data['iss_prefetches'][core] = int(line.split()[-8])
    
    if not all(key in data for key in expected_keys):
        return None

    return data



def compute_stats(trace, baseline_name=''):
    """Compute additional statistics, after reading the raw
    data from the trace. Return it as a CSV row.
    """
    data = read_file(trace)
    if not data:
        return ''
    
    n_cores = max(data['ipc'].keys()) + 1
    out = ''

    for core in sorted(data['ipc'].keys()):
        trace, ipc, load_miss, rfo_miss, kilo_inst = (
            data['trace'][core], data['ipc'][core], data['load_miss'][core], 
            data['rfo_miss'][core], data['kilo_inst'][core]
        )
        is_homogeneous = data['is_homogeneous']
        
        
        mpki = (load_miss + rfo_miss) / kilo_inst
        out += f'\n{trace},{baseline_name},{core},{n_cores},{is_homogeneous},{mpki},{ipc},{load_miss},{rfo_miss}'
    
    return out



def build_run_statistics(results_dir, output_file):
    """Build statistics for each run, per-core.
    """
    traces = {}
    for fn in os.listdir(results_dir):
        trace = fn.split('-hashed_perceptron-')[0]
        if trace not in traces:
            traces[trace] = {}
        for base_fn in baseline_fns:
            if base_fn == fn.split('-hashed_perceptron-')[1].split('-')[4]:
                traces[trace][base_fn] = os.path.join(results_dir, fn)

                
    
    stats = 'Trace,Baseline,CPU,NumCPUs,HomogeneousMix,MPKI,IPC,LoadMisses,RFOMisses'
    for trace in tqdm(traces, dynamic_ncols=True, unit='trace'):
        d = traces[trace]

        for baseline in d:
            stats += compute_stats(d[baseline], baseline_name=baseline)

    with open(output_file, 'w') as f:
        print(stats, file=f)
    
    
    
def build_trace_statistics(run_stats_file):
    """Build statistics for each trace's fairness,
    using already-computed run statistics.
    """
    columns = ['Trace', 'Baseline', 'NumCPUs', 
               'MinIPC', 'MeanIPC', 'MaxIPC',
               'MinMPKI', 'MeanMPKI', 'MaxMPKI',
               'HomoNormMinIPC', 'HomoNormMeanIPC', 'HomoNormMaxIPC',
               'HomoNormMinMPKI', 'HomoNormMeanMPKI', 'HomoNormMaxMPKI']
    
    trace_df = pd.DataFrame(columns=columns)
    run_df = pd.read_csv(run_stats_file)
    
    # TODO - Clean up loop to do all three groups / uniques at once.
    t = run_df.groupby('Trace')
    for trace in run_df.Trace.unique():
        
        b = t.get_group(trace).groupby('Baseline')
        for baseline in run_df.Baseline.unique():
        
            c = b.get_group(baseline).groupby('NumCPUs')
            for n_cores in run_df.NumCPUs.unique():
                
                try:
                    runs = c.get_group(n_cores)
                except KeyError:
                    print(f'No runs match trace={trace}, baseline={baseline}, n_cores={n_cores}')
                    continue
                
                
                homo = runs[runs.HomogeneousMix == True]
                homo_ipc = homo.IPC.mean() if not homo.empty else np.nan # Average homogeneous IPC (over the cores)
                homo_mpki = homo.MPKI.mean() if not homo.empty else np.nan # Average homogeneous IPC (over the cores)
                    
                # Raw min, mean, max IPCs
                min_ipc, mean_ipc, max_ipc = runs.IPC.min(), runs.IPC.mean(), runs.IPC.max()
                
                # Raw min, mean, max MPKIs
                min_mpki, mean_mpki, max_mpki = runs.MPKI.min(), runs.MPKI.mean(), runs.MPKI.max()
                
                # Min, mean, max IPCs normalized to homogeneous mix
                norm_min_ipc, norm_mean_ipc, norm_max_ipc = min_ipc / homo_ipc, mean_ipc / homo_ipc, max_ipc / homo_ipc
                
                # Min, mean, max MPKIs normalized to homogeneous mix
                norm_min_mpki, norm_mean_mpki, norm_max_mpki = min_mpki / homo_mpki, mean_mpki / homo_mpki, max_mpki / homo_mpki
                
                trace_df.loc[len(trace_df.index)] = [
                    trace, baseline, n_cores, 
                    min_ipc, mean_ipc, max_ipc,
                    min_mpki, mean_mpki, max_mpki,
                    norm_min_ipc, norm_mean_ipc, norm_max_ipc,
                    norm_min_mpki, norm_mean_mpki, norm_max_mpki,
                ]
    
    trace_stats_file = run_stats_file.replace('.csv', '') + '_trace.csv'
    print(f'Saving dataframe to {trace_stats_file}...')
    trace_df.to_csv(trace_stats_file, index=False)
            



def eval_command():
    """Eval command
    """
    parser = argparse.ArgumentParser(usage=argparse.SUPPRESS, add_help=False)
    parser.add_argument('--results-dir', default=default_results_dir)
    parser.add_argument('--output-file', default=default_output_file)

    args = parser.parse_args(sys.argv[2:])
    
    print('=== Building run statistics... ===')
    build_run_statistics(args.results_dir, args.output_file)

    print('=== Building trace statistics... ===')
    build_trace_statistics(args.output_file)



    
"""
Help
"""
def help_command():
    """Help command
    """
    # If one of the available help strings, print and exit successfully
    if len(sys.argv) > 2 and sys.argv[2] in help_str:
        print(help_str[sys.argv[2]])
        exit()
    # Otherwise, invalid subcommand, so print main help string and exit
    else:
        print(help_str['help'])
        exit(-1)

    

"""
Main
"""
commands = {
    'build': build_command,
    'run': run_command,
    'eval': eval_command,
    'help': help_command,
}

def main():
    # If no subcommand specified or invalid subcommand, print main help string and exit
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(help_str['help'])
        exit(-1)

    # Run specified subcommand
    commands[sys.argv[1]]()

if __name__ == '__main__':
    main()
