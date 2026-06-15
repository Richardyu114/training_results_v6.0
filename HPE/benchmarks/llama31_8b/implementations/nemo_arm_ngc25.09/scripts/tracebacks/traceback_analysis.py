import os
import sys
import argparse
import csv
import re
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('dir_path')
parser.add_argument('--output', default='traceback_analysis.md', type=str)
parser.add_argument('--aggregate', action='store_true')
parser.add_argument('--no-aggregate', dest='aggregate', action='store_false')
parser.add_argument('--main-thread-tid', default='0x1555551a6740', type=str)
parser.add_argument('--pyspy-max-lines', default=15, type=int)
parser.add_argument('--gdb-max-lines', default=15, type=int)
parser.set_defaults(aggregate=True)
args = parser.parse_args()

def convert_line_to_markdown(line):
    # Leading spaces are stripped and each newline character
    # is converted into two spaces
    return line.lstrip().replace('\n', '  ')

def get_lines_from_thread(log, tid, max_lines, aggregate):
    # Open the provided log, find the thread with the given ID
    # and obtain some number of lines
    with open(log, 'r') as f:
        found_thread = False
        line_count = 0
        lines = ""

        for line in f.readlines():
            if found_thread and line_count < max_lines:
                md_line = convert_line_to_markdown(line)
                if aggregate:
                    # Remove thread IDs (starting with 0x)
                    md_line = re.sub(r'\b0x\w+\b', '', md_line)
                lines += md_line
                line_count += 1

            if line.lower().find(tid.lower()) != -1 and not found_thread:
                found_thread = True

    return '```' + lines + '```'

# Read logs and build dictionary
with os.scandir(args.dir_path) as entries:
    # Read all pyspy logs
    pyspy_logs = [entry.name for entry in entries if entry.name.endswith('.pyspy')]
    pyspy_logs.sort()
    if not pyspy_logs:
        sys.exit("No pyspy logs found!")

    # Assume GDB logs have the same file name as pyspy logs
    gdb_logs = [pyspy_log.split('.')[0] + '.gdb' for pyspy_log in pyspy_logs]

    traceback_dict = {}
    pyspy_dict = defaultdict(list)

    # Go through the logs and retrieve stack traces
    for i, (pyspy_log, gdb_log) in enumerate(zip(pyspy_logs, gdb_logs)):
        rank = int(pyspy_log.split('_')[0].split('rank')[1])

        # Get stack from pyspy log
        pyspy_lines = get_lines_from_thread(args.dir_path + '/' + pyspy_log,
                                            args.main_thread_tid, args.pyspy_max_lines,
                                            args.aggregate)

        # Get stack from GDB log
        gdb_lines = get_lines_from_thread(args.dir_path + '/' + gdb_log,
                                          args.main_thread_tid, args.gdb_max_lines,
                                          args.aggregate)

        if args.aggregate:
            pyspy_dict[pyspy_lines].append(rank)
        else:
            traceback_dict[rank] = [pyspy_lines, gdb_lines]

# Create pandas dataframe
if not args.aggregate:
    from pandas import DataFrame as df
    traceback_df = df.from_dict(traceback_dict, orient='index', columns=['pyspy', 'GDB'])

# Write dataframe out as markdown
with open(args.output, 'w') as md_file:
    if args.aggregate:
        for k, v in pyspy_dict.items():
            # Convert rank list to string, then write stack trace followed by the ranks
            pyspy_dict[k] = [",".join(str(x) for x in v)]
            md_file.write(k + "\n" + pyspy_dict[k][0] + "\n")
    else:
        md_file.write(traceback_df.to_markdown())
