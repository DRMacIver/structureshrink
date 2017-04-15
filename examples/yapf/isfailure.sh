#!/usr/bin/env bash

set -e -x

cp engine.py target.py

python target.py || exit 1
flake8 target.py || exit 2
yapf -i target.py || exit 3

if 
  flake8 target.py
then
  exit 4
else
  exit 5
fi
