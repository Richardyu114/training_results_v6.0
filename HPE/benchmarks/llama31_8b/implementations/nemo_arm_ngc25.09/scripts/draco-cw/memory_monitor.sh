#!/bin/bash

hostname=$(hostname)
NODENAME=$(echo "$hostname" | sed 's/\..*//')
MEMLOGDIR=${LOGDIR}/mem-logs
mkdir -p ${MEMLOGDIR}

shmem_monitor () {
  SHMEM_LOG=${MEMLOGDIR}/${SLURM_JOB_ID}_${NODENAME}_shmem.log
  rm -f ${SHMEM_LOG}
  while true; do
	  df -h /dev/shm | tail -n1 >> ${SHMEM_LOG}
	  sleep 1
  done
}

shmem_nccl_monitor () {
  SHMEM_NCCL_LOG=${MEMLOGDIR}/${SLURM_JOB_ID}_${NODENAME}_shmem_nccl.log
  rm -f ${SHMEM_NCCL_LOG}
  while true; do
  	ls /dev/shm/nccl-* | wc -l >> ${SHMEM_NCCL_LOG}
	sleep 1
  done
}

hostmem_monitor () {
  HOSTMEM_LOG=${MEMLOGDIR}/${SLURM_JOB_ID}_${NODENAME}_hostmem.log
  rm -f ${HOSTMEM_LOG}
  while true; do
	  free -h >> ${HOSTMEM_LOG}
	  sleep 1
  done
}

echo $NODENAME

shmem_monitor & 
shmem_nccl_monitor &
hostmem_monitor 

