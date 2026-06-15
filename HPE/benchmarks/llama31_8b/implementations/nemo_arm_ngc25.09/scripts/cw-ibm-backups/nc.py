#!/usr/bin/env python3
import socket, sys, select

if len(sys.argv) != 2:
    print(f"usage: {sys.argv[0]} /path/to/socket", file=sys.stderr)
    sys.exit(1)

s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect(sys.argv[1])
s.setblocking(False)
sys.stdin = sys.__stdin__
sys.stdout = sys.__stdout__

while True:
    rlist, _, _ = select.select([s, sys.stdin], [], [])
    if s in rlist:
        data = s.recv(4096)
        if not data:
            break
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
    if sys.stdin in rlist:
        data = sys.stdin.buffer.read1(4096)
        if not data:
            s.shutdown(socket.SHUT_WR)
        else:
            s.sendall(data)
