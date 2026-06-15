# Rack and Node tracker

The Coreweave Slurm on Kubernetes (SUNK) manages nodes dynamically and
independently of the slurm instance.  This means that unlike typical clusters
where there is a static list of nodes that are either UP or DOWN (or DRAINED)
or whatever. When nodes go down (or need to be allocated to a different
customer) they will dynamically disappear from one slurm instance and new nodes
may dynamically appear as the admins bring them online.

This makes it somewhat difficult for us to figure out when a node gets drained
what rack it was in.  The node disappears from the slurm instance, so we can't
query `scontrol show node` to find out anything about the node.

The script in this directory runs in the background and once every five minutes
queries slurm to find out what nodes and racks it knows about and appends any
new nodes into the `rack-node-database.txt`.  That means we can use
`rack-node-database.txt` to find out about drained nodes.

## stopping the current background job

```
$ stop-rack-node-tracker
```

The `rack-node-tracker` is running in the background and wakes up every five
minutes.  When it wakes up it checks for the .running_pid file, and if the
contents don't exist or are different pid, then it shuts itself down.

## starting a new background job (or replacing the current one)

```
$ launch-rack-node-tracker
```

This is safe to do any time.  It launches a new background process.  If there
wals already an existing process it will wake up and shut itself down within
five minutes.


## rack-node-tracker (background script)

Wakes up every five minutes and checks if the list of nodes has changed, if it
does it makes a backup of the old `rack-node-database.txt` and appends any new
nodes.
