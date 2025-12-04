#!/bin/bash
unzip -q /root/gurusmart/$1 -d /workspace/
mkdir -p ~/.ssh
cp /root/gurusmart/.ssh/id_rsa ~/.ssh/id_rsa
cp /root/gurusmart/.ssh/id_rsa.pub ~/.ssh/id_rsa.pub

BRANCH="main"
if [ -n "$2" ]; then
    BRANCH="$2"
fi

GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no" \
    git clone --branch "$BRANCH" git@github.com:atong28/SMART-Moonshot.git /code

pixi install --manifest-path /code/pixi.toml