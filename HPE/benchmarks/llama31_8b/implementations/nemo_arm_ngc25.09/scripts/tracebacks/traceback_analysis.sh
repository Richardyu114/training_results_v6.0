#!/bin/bash
set -e

# Parse command line arguments
dir_path=$1
output=${2:-traceback_analysis.md}
aggregate=${3:-true}
main_thread_tid=${4:-0x1555551a6740}
pyspy_max_lines=${5:-15}
gdb_max_lines=${6:-15}

pyspy_tmp_file=$(mktemp)
gdb_tmp_file=$(mktemp)

get_lines_from_thread() {
    local log=$1
    local tid=$2
    local max_lines=$3
    local aggregate=$4
    local tmp_file=$5

    #Explanation:
    # 1. Use lowercase for the whole line:
    # 2. match lowercase $tid and return the next $max_lines if the match was found:
    local awk_command="tolower(\$0) ~ /${tid,,}/{flag=1;next} flag && c++<${max_lines}"
    #awk_command+=" {gsub(/^[\t ]+/, \"\", \$0); gsub(/\b0x[0-9a-fA-F]+\b/, \"\", \$0); printf \"%s \", \$0}"
    # 3. lstrip. gsub is a global substitution. The third argument is the target to store the output of the commend. $0 is the currently processed line
    awk_command+=" {gsub(/^[\t ]+/, \"\", \$0);"
    # 4. remove hexadecimal numbers if $aggregate is true (returns 0 exit code)
    $aggregate && awk_command+=" gsub(/\b0x[0-9a-fA-F]+\b/, \"\", \$0);"
    # 5. Print lines joint in a single line separated by space
    awk_command+=" printf \"%s \", \$0}"                                                                                                                                                                                                    

    rank=$(echo ${log} | sed 's/.*rank\([0-9]*\)_.*/\1/; s/^0*//; s/^$/0/')
    echo ${rank}'|```'$(awk "$awk_command" $log)'```' >> $tmp_file
}

export -f get_lines_from_thread

# Create list of log files
pyspy_logs=($(find "$dir_path" -type f -name "*.pyspy" | sort))
gdb_logs=($(for log in "${pyspy_logs[@]}"; do echo "${log%.*}.gdb"; done))

# Parse log files
parallel_args=" $main_thread_tid $pyspy_max_lines $aggregate $pyspy_tmp_file"
parallel -j 4 --line-buffer 'get_lines_from_thread {}'${parallel_args} ::: ${pyspy_logs[@]} &

if ! $aggregate; then
	parallel_args=" $main_thread_tid $pyspy_max_lines $aggregate $gdb_tmp_file"
	parallel -j 4 --line-buffer 'get_lines_from_thread {}'${parallel_args} ::: ${gdb_logs[@]} &
fi
wait

#Aggregate 
if $aggregate; then

    echo '|                            | ranks             |' > $output
    echo '|---------------------------:|:------------------|' >> $output
	awk -F '|' '{
		if ($2 in keys) {keys[$2] = keys[$2] "," $1}
		else {keys[$2] = $1}
	}
	END {
		for (key in keys) {print key, "|" ,keys[key]}
	}' $pyspy_tmp_file >> $output
else
	sorted1=$(mktemp)
	sorted2=$(mktemp)
	sort -t '|' -k1,1 $pyspy_tmp_file > $sorted1
	sort -t '|' -k1,1 $gdb_tmp_file > $sorted2

    echo '  |     | pyspy                      | GDB                                               |' > $output
    echo '  |----:|:--------------------------:|:--------------------------------------------------|' >> $output
	# IDK why but sorting files numerically before join results in errors. That's why we have another sort here
	join -t '|' -o 1.1,1.2,2.2 $sorted1 $sorted2 | sort -t '|' -k1,1n >> $output
	rm $sorted1 $sorted2

fi

#cleanup
rm $pyspy_tmp_file $gdb_tmp_file
