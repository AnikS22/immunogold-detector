#!/bin/bash
# Check HPC cluster configuration

echo "=== HPC Partition Information ==="
sinfo -p gpu -o '%partitionname %maxtime %maxcpus %maxmemory'

echo ""
echo "=== CPU Partition Info ==="
sinfo -o '%partitionname %maxtime' | grep -v gpu

echo ""
echo "=== Your Recent Jobs ==="
squeue -u $USER --sort=+time | tail -10
