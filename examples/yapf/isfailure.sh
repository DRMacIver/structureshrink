#!/usr/bin/env bash

set -e -x

# This script is organised into a series of passes so that different failure
# modes give different exit statuses. It exits with zero if it exhibits the
# behaviour that the original file is flake8 correct but yapf introduces a
# pep8 error into the result.
# The passes are ordered in order of decreasing speed, so when this fails it
# should fail relatively fast. This partially makes up for the fact that it's
# a really hard condition to satisfy: There's a lot that you can't delete
# because it would cause a variable to be unused, or cause a refernece to
# an undefined variable.

# Fail if this isn't valid Python
python -c 'x = open("engine.py").read(); import ast; ast.parse(x)' || exit 1

# Fail if there are any pep8 errors in it
pep8 engine.py || exit 2

# Do a full flake8 lint (much slower) which will check for valid variable
# defintiions etc.
flake8 engine.py || exit 3

# Actually try to run it.
python engine.py || exit 4

# Run yapf over the file to format it.
yapf -i engine.py || exit 5

if 
  pep8 engine.py
then
  # Good job yapf, you didn't introduce an error
  exit 6
else
  # Our actual desired result: yapf took an apparently valid file and made it
  # invalid.
  exit 0
fi
