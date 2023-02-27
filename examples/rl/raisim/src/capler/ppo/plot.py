#!/usr/bin/env python

# Copyright 2020 The DeepGait Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A script for plotting results aggregated over a set of experiments."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

# System modules
import os
import glob
import argparse

# Processing modules
import numpy as np
import pandas as pd
import seaborn as sns

# Data modules
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Plotting modules
import matplotlib.pyplot as plt

# matplotlib configs
plt.ioff()
plt.style.use('seaborn-whitegrid')
plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.serif'] = ['Times New Roman']
tfont = {'size': 20}
lfont = {'size': 16}
afont = {'size': 12}


def tb2data(files):
    events = [os.path.basename(file) for file in files]
    prefix = os.path.commonprefix(events).rpartition('.')[0] + '.'
    _, suffix = os.path.splitext(events[0])
    raw = pd.DataFrame()
    stamps = []
    for file in files:
        stamps.append(int(os.path.basename(file).replace(prefix, "").replace(suffix, "")))
        log = tb2pd(file, stamps[-1])
        if log is not None:
            raw = raw.append(log, ignore_index=True)
    data = pd.pivot_table(raw, values='value', index=['name', 'step'], columns=['stamp'])
    names = raw['name'].unique()
    return names, data


def tb2pd(file, stamp=None):
    event_acc = EventAccumulator(file, {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    })
    event_acc.Reload()
    scalars = event_acc.Tags()["scalars"]
    data = pd.DataFrame({"name": [], "value": [], "step": [], "stamp": []})
    for sig in scalars:
        event_list = event_acc.Scalars(sig)
        values = list(map(lambda x: x.value, event_list))
        step = list(map(lambda x: x.step, event_list))
        frame = {"name": [str(sig)] * len(step), "value": values, "step": step, "stamp": [stamp] * len(step)}
        frame = pd.DataFrame(frame)
        data = pd.concat([data, frame])
    return data


def pd2stats(data, interval=2):
    mean = data.mean(axis=1)
    std = data.std(axis=1)
    upper = mean + interval*std
    lower = mean - interval*std
    return mean, std, lower, upper


def save_runs_figures(name, data, path):
    # Configure and create target path
    output = path + '/' + name
    outdir = os.path.dirname(output)
    try:
        os.makedirs(outdir)
    except OSError:
        print("Cannot create directory: %s" % outdir)
    else:
        print("Creating plots at: %s " % outdir)
    # Generate figure with totality of runs
    plt.subplots()
    sns.lineplot(data=data, dashes=False)
    plt.tick_params(labelsize=afont['size'])
    plt.xlabel('Iterations', **lfont)
    plt.ylabel(name, **lfont)
    plt.title('Runs', **tfont)
    plt.savefig(output + '-runs.svg', dpi=100, quality=100,
                facecolor='w', edgecolor='w', orientation='portrait',
                format='svg', transparent=False, bbox_inches=None,
                pad_inches=0.1, metadata=None)
    # Close all generated figures
    plt.close('all')


def save_statistics_figure(name, data, path):
    # Configure and create target path
    output = path + '/' + name
    outdir = os.path.dirname(output)
    try:
        os.makedirs(outdir)
    except OSError:
        print("Cannot create directory: %s" % outdir)
    else:
        print("Creating plots at: %s " % outdir)
    # Generate figure with confidence bounds
    fig2, ax2 = plt.subplots()
    if isinstance(data, list):
        for datum in data:
            indices = datum.index.values
            mean, std, lower, upper = pd2stats(datum)
            ax2.fill_between(indices, lower, upper, alpha=0.2)
            mean.plot()
    else:
        mean, std, lower, upper = pd2stats(data)
        ax2.fill_between(data.index.values, lower, upper, alpha=0.2)
        mean.plot()
    plt.tick_params(labelsize=afont['size'])
    plt.xlabel('Iterations', **lfont)
    plt.ylabel(name, **lfont)
    plt.title("Statistics", **tfont)
    plt.savefig(output + '-stats.svg', dpi=100, quality=100,
                facecolor='w', edgecolor='w', orientation='portrait',
                format='svg', transparent=False, bbox_inches=None,
                pad_inches=0.1, metadata=None)
    # Close all generated figures
    plt.close('all')


def parse_args():
    """
    Parses CLI arguments applicable for this helper script
    """
    # Create parser instance
    parser = argparse.ArgumentParser(description='A script for plotting CSV data from DeepGait Controller Training.')
    # Define arguments
    parser.add_argument("--input", help='Directory containing CSV files', default='~/.noesis/proc/noesis_rl_train_capler_ppo_example')
    parser.add_argument("--output", help='Output directory for figures.', default='~/.noesis/tmp/capler-ppo-example')
    # Retrieve arguments
    args = parser.parse_args()
    return args


def main(args):

    # Resolve absolute paths
    inpath = os.path.expanduser(args.input)
    outpath = os.path.expanduser(args.output)
    print("TB-Aggr: Input: ", inpath)
    print("TB-Aggr: Output: ", outpath)

    # Set the experiment set
    # experiments = ['/baseline', '/shared-net', '/shared-layer', '/state-dep-stddev']
    # experiments = ['/baseline', '/shared-net', '/shared-layer']
    experiments = ['/tv_-5.0', '/tv_-3.0', '/tv_-2.0', '/tv_-1.0']
    # experiments = ['/' + path for path in os.listdir(inpath)]
    print("TB-Aggr: Search Dirs: ", experiments)

    # Generate the list of experiments and respective runs
    files = [glob.glob(inpath + exp + '/*/logs/*.tfevents.*') for exp in experiments]
    print("TB-Aggr: Found Files:\n", files)

    # Extract pandas from the specific file-set
    # names, data = tb2data(files[0])
    data = [tb2data(file) for file in files]
    names = data[0][0]
    print("TB-Aggr: Data Names:\n", names)

    # Select data subset
    fig_names = [
        'Sampler/mean_episode_length',
        'Sampler/mean_step_reward',
        'Sampler/mean_episode_reward',
        'Sampler/mean_episode_return',
        'Sampler/mean_terminal_value'
    ]

    # Generate SVG figures.
    # save_statistics_figure(names[0], [datum[1].loc[names[0]] for datum in data], outpath)
    for name in fig_names:
        save_statistics_figure(name, [datum[1].loc[name] for datum in data], outpath)


if __name__ == '__main__':
    main(parse_args())

# EOF
