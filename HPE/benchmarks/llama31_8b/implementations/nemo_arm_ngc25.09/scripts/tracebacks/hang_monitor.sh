#!/usr/bin/env bash

set -eux

# Defines functions to monitor for hangs and launch tracebacks dump script
# Runs the `HANG_MONITOR_EXEC_CMD` command `HANG_MONITOR_TIMEOUT` minutes before the timeout of `SLURM_JOBID` job.

: "${HANG_MONITOR_TIMEOUT:?HANG_MONITOR_TIMEOUT not set}"
: "${HANG_MONITOR_EXEC_CMD:?HANG_MONITOR_EXEC_CMD not set}"
: "${SLURM_JOBID:?SLURM_JOBID not set}"

# Converts timeleft in format HH:MM:SS or MM:SS to seconds
timeleft_to_seconds() {
  timeleft=$1
  timeleft_arr=(`echo $timeleft | tr ':-' ' '`)
  [ "${#timeleft_arr[@]}" -gt 3 ] && echo "Unsupported timeleft. Hang monitor won't be launched" && return
  [ "${#timeleft_arr[@]}" -lt 2 ] && echo "Unsupported timeleft. Hang monitor won't be launched" && return

  # 10# prefix ensures numbers are interpreted in base 10 (e.g. 1:08 would cause 08 to be interpreted in base 8)
  seconds_left=$(( 10#${timeleft_arr[0]}*60 + 10#${timeleft_arr[1]} ))
  if [ "${#timeleft_arr[@]}" -eq 3 ]; then
    seconds_left=$(( seconds_left*60 + 10#${timeleft_arr[2]} ))
  fi
  echo $seconds_left
}

hang_monitor() {
  echo "Launching hang monitor"
  sleep 60  # wait for the job to start

  timeleft=`squeue -j ${SLURM_JOBID} --noheader --format=%L`
  sleep_duration=$(( $(timeleft_to_seconds $timeleft) - 60 * HANG_MONITOR_TIMEOUT ))

  if [ -z "$sleep_duration" ]; then
    echo "Invalid sleep duration. Hang monitor won't be launched"
    return
  fi

  # we run this in a loop in case the job time limit changes
  while [ "$sleep_duration" -gt 0 ]; do
    sleep 60
    timeleft=`squeue -j ${SLURM_JOBID} --noheader --format=%L`
    sleep_duration=$(( $(timeleft_to_seconds $timeleft) - 60 * HANG_MONITOR_TIMEOUT ))
  done

  echo "Launching tracebacks script for timeout ${HANG_MONITOR_TIMEOUT}"
  bash -c "$HANG_MONITOR_EXEC_CMD"
  echo "Tracebacks script done"
}
