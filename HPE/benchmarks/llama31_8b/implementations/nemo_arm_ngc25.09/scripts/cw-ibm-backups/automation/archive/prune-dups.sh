#!/bin/bash
shopt -s nullglob
declare -A hash_seen

for f in $(ls -1tr rack-node-database.txt+*); do
    hash=$(md5sum "$f" | awk '{print $1}')
    if [[ -n "${hash_seen[$hash]:-}" ]]; then
        echo "Deleting duplicate: $f"
        rm -f "$f"
    else
        hash_seen[$hash]="$f"
    fi
done
