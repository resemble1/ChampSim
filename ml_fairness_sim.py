#!/usr/bin/env python3

import argparse
import os
import sys

from model import Model

default_results_dir = './results'
default_output_file = './stats.csv'
default_instrs = 500
default_printing_period_instrs = 10

default_seed_file = './scripts/seeds.txt'

# No prefetcher
default_lru_binary = 'bin/hashed_perceptron-no-no-no-no-lru-{n_cores}core'
default_hawkeye_binary = 'bin/hashed_perceptron-no-no-no-no-???-{n_cores}core' # TODO change

baseline_names = ['lru', 'hawkeye']
baseline_fns = ['lru', '???'] # TODO - Obtain Hawkeye binary
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
    if args.target in baseline_fns:
        for name, fn in zip(baseline_names, baseline_fns):
            for c in cores:
                print(f'=== Building {name} ChampSim binary ({fn}), {c} core{"s" if c > 1 else ""} ===')
                os.system(f'./build_champsim.sh hashed_perceptron no no no no {fn} {c}')

                
            
"""
Run
"""
def run_command():
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

#     # Seed file
#     seeds = {}
#     if not os.path.exists(args.seed_file):
#         print('Seed file "' + args.seed_file + '" does not exist')
        
#     else:
#         with open(args.seed_file, 'r') as f:
#             for tr in execution_traces:
#                 for line in f:
#                     line = line.strip()
#                     if line.split()[0] in os.path.basename(tr):
#                         seeds[tr] = line.split()[1]
#                         break
#                 else:
#                     print('Could not find execution trace "{}" in seed file "{}"'.format(tr, args.seed_file))
#                     seeds[tr] = None

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

        # if seeds is not None:
        #     cmd = '{binary} -stat_printing_period {period}000000  -simulation_instructions {sim}000000 -seed {seed} -traces {trace} > {results}/{base_trace}-{base_binary}.txt 2>&1'.format(
        #         binary=binary, warm=args.num_prefetch_warmup_instructions, sim=args.num_instructions,
        #         trace=execution_trace, seed=seed, results=args.results_dir,
        #         period=args.stat_printing_period,
        #         base_trace=os.path.basename(execution_trace), base_binary=os.path.basename(binary))
        # else:
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
def read_file(path, cache_level='LLC'):
    expected_keys = ('ipc', 'total_miss', 'useful', 'useless', 'uac_correct', 'iss_prefetches', 'load_miss', 'rfo_miss', 'kilo_inst')
    data = {}
    with open(path, 'r') as f:
        for line in f:
            if 'Finished CPU' in line:
                data['ipc'] = float(line.split()[9])
                data['kilo_inst'] = int(line.split()[4]) / 1000
            if cache_level not in line:
                continue
            line = line.strip()
            if 'LOAD' in line:
                data['load_miss'] = int(line.split()[-1])
            elif 'RFO' in line:
                data['rfo_miss'] = int(line.split()[-1])
            elif 'TOTAL' in line:
                data['total_miss'] = int(line.split()[-1])
            elif 'USEFUL' in line:
                data['useful'] = int(line.split()[-6])
                data['useless'] = int(line.split()[-4])
                data['uac_correct'] = int(line.split()[-1])
                data['iss_prefetches'] = int(line.split()[-8])

    if not all(key in data for key in expected_keys):
        return None

    return data

def compute_stats(trace, prefetch=None, base=None, baseline_name=None):
    if prefetch is None:
        return None

    pf_data = read_file(prefetch)

    iss_prefetches, uac_correct, useful, useless, ipc, load_miss, rfo_miss, kilo_inst = (
        pf_data['iss_prefetches'], pf_data['uac_correct'], pf_data['useful'], pf_data['useless'], 
        pf_data['ipc'], pf_data['load_miss'], pf_data['rfo_miss'], pf_data['kilo_inst']
    )
    pf_total_miss = load_miss + rfo_miss + useful
    total_miss = pf_total_miss

    pf_mpki = (load_miss + rfo_miss) / kilo_inst

    if base is not None:
        b_data = read_file(base)
        b_total_miss, b_ipc = b_data['total_miss'], b_data['ipc']
        b_load_miss = b_data['load_miss']
        b_rfo_miss = b_data['rfo_miss']
        b_mpki = b_total_miss / kilo_inst

    if useful + useless == 0:
        acc = 'N/A'
    else:
        acc = str(useful / (useful + useless) * 100)
        
    if total_miss == 0 or base is None:
        cov = 'N/A'
        oldcov = 'N/A'
    else:
        #cov = str((b_total_miss - load_miss - rfo_miss) / b_total_miss * 100) # formerly str(useful / total_miss * 100)
        cov = str(((b_load_miss + b_rfo_miss) - (load_miss + rfo_miss)) / (b_load_miss + b_rfo_miss) * 100)
        oldcov = str(str(useful / total_miss * 100))
        
    if iss_prefetches == 0:
        uac = 'N/A'
    else:
        uac = str(uac_correct / iss_prefetches * 100)
        
    if base is not None:
        mpki_improv = str((b_mpki - pf_mpki) / b_mpki * 100)
        ipc_improv = str((ipc - b_ipc) / b_ipc * 100)
    else:
        mpki_improv = 'N/A'
        ipc_improv = 'N/A'

    return '{trace},{baseline_name},{acc},{cov},{oldcov},{uac},{mpki},{mpki_improv},{ipc},{ipc_improv}'.format(
        trace=trace, baseline_name=baseline_name, acc=acc, cov=cov, oldcov=oldcov, uac=uac, mpki=str(pf_mpki),
        mpki_improv=mpki_improv, ipc=str(ipc), ipc_improv=ipc_improv,
    )

def eval_command():
    parser = argparse.ArgumentParser(usage=argparse.SUPPRESS, add_help=False)
    parser.add_argument('--results-dir', default=default_results_dir)
    parser.add_argument('--output-file', default=default_output_file)

    args = parser.parse_args(sys.argv[2:])

    traces = {}
    for fn in os.listdir(args.results_dir):
        trace = fn.split('-hashed_perceptron-')[0]
        if trace not in traces:
            traces[trace] = {}
        if 'from_file' in fn:
            traces[trace]['prefetch'] = os.path.join(args.results_dir, fn)
        else:
            for base_fn in baseline_fns:
                if base_fn == fn.split('-hashed_perceptron-')[1].split('-')[3]:
                    traces[trace][base_fn] = os.path.join(args.results_dir, fn)

    stats = ['Trace,Baseline,Accuracy,Coverage,Coverage_Old,UAC,MPKI,MPKI_Improvement,IPC,IPC_Improvement']
    for trace in traces:
        d = traces[trace]
        print(trace)
        print(d)
        
        baseline_name = 'no' if 'no' in d else 'No Baseline'
        if 'no' in d:
            stats.append(compute_stats(trace, d['no'], baseline_name='no'))
        for pf in d:
            if pf != 'no':
                stats.append(compute_stats(trace, d[pf], d['no'] if 'no' in d else None, baseline_name=pf))
        #if 'no' in d:
        #    stats.append(compute_stats(trace, d['no'], baseline_name='no'))
        #    stats.append(compute_stats(trace, d['prefetch'], d['no'], baseline_name='yours'))
        #else:
        #    stats.append(compute_stats(trace, d['prefetch'], baseline_name='No Baseline'))
        #for fn in baseline_fns:
        #    if fn in d:
        #        trace_stats = None
        #        if fn != 'no' and 'no' in d:
        #            trace_stats = compute_stats(trace, d[fn], d['no'], baseline_name=fn)
        #        if trace_stats is not None:
        #            stats.append(trace_stats)

    with open(args.output_file, 'w') as f:
        print('\n'.join(stats), file=f)

def generate_prefetch_file(path, prefetches):
    with open(path, 'w') as f:
        for instr_id, pf_addr in prefetches:
            print(instr_id, hex(pf_addr)[2:], file=f)

def read_load_trace_data(load_trace, num_prefetch_warmup_instructions):
    
    def process_line(line):
        split = line.strip().split(', ')
        return int(split[0]), int(split[1]), int(split[2], 16), int(split[3], 16), split[4] == '1'

    train_data = []
    eval_data = []
    if load_trace.endswith('.txt'):
        with open(load_trace, 'r') as f:
            for i, line in enumerate(f):
                if line.startswith('***') or line.startswith('Read'):
                    continue
                pline = process_line(line)
                if pline[0] < num_prefetch_warmup_instructions * 1000000:
                    train_data.append(pline)
                else:
                    eval_data.append(pline)
    elif load_trace.endswith('.txt.xz'):
        import lzma
        with lzma.open(load_trace, mode='rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.startswith('***') or line.startswith('Read'):
                    continue
                pline = process_line(line)
                if pline[0] < num_prefetch_warmup_instructions * 1000000:
                    train_data.append(pline)
                else:
                    eval_data.append(pline)
    else:
        print('Unsupported load trace file format')
        exit(-1)

    return train_data, eval_data

def train_command():
    if len(sys.argv) < 3:
        print(help_str['train'])
        exit(-1)
    #'train': '''usage: {prog} train <load-trace> [--model <model-path>] [--generate <prefetch-file>] [--num-prefetch-warmup-instructions <num-warmup-instructions>]

    parser = argparse.ArgumentParser(usage=argparse.SUPPRESS, add_help=False)
    parser.add_argument('load_trace', default=None)
    parser.add_argument('--generate', default=None)
    parser.add_argument('--model', default=None)
    parser.add_argument('--num-prefetch-warmup-instructions', type=int, default=default_warmup_instrs)

    args = parser.parse_args(sys.argv[2:])

    train_data, eval_data = read_load_trace_data(args.load_trace, args.num_prefetch_warmup_instructions)

    model = Model()
    model.train(train_data)

    if args.model is not None:
        model.save(args.model)

    if args.generate is not None:
        prefetches = model.generate(eval_data)
        generate_prefetch_file(args.generate, prefetches)

def generate_command():
    if len(sys.argv) < 3:
        print(help_str['generate'])
        exit(-1)

    #'generate': '''usage: {prog} generate <load-trace> <prefetch-file> [--model <model-path>] [--num-prefetch-warmup-instructions <num-warmup-instructions>]
    parser = argparse.ArgumentParser(usage=argparse.SUPPRESS, add_help=False)
    parser.add_argument('load_trace', default=None)
    parser.add_argument('prefetch_file', default=None)
    parser.add_argument('--model', default=None, required=True)
    parser.add_argument('--num-prefetch-warmup-instructions', default=default_warmup_instrs)

    args = parser.parse_args(sys.argv[2:])

    model = Model()
    model.load(args.model)

    _, data = read_load_trace_data(args.load_trace, args.num_prefetch_warmup_instructions)

    prefetches = model.generate(data)

    generate_prefetch_file(args.prefetch_file, prefetches)

def help_command():
    # If one of the available help strings, print and exit successfully
    if len(sys.argv) > 2 and sys.argv[2] in help_str:
        print(help_str[sys.argv[2]])
        exit()
    # Otherwise, invalid subcommand, so print main help string and exit
    else:
        print(help_str['help'])
        exit(-1)

commands = {
    'build': build_command,
    'run': run_command,
    'eval': eval_command,
    'train': train_command,
    'generate': generate_command,
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
