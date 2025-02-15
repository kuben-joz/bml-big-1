#!/bin/bash

echo "* Setting up cluster."

ssh -i root@container2 'echo "- Running code on container2! $(hostname)"'
ssh -i root@container3 'echo "- Running code on container3! $(hostname)"'
ssh -i root@container4 'echo "- Running code on container4! $(hostname)"'

echo "* Done!"

/usr/sbin/sshd -D