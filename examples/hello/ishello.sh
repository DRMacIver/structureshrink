#!/usr/bin/env bash

set -e -x

if
  g++ -O3 -w hello.cpp >compiler.out 2>&1
then
  ulimit -t 1 && ./a.out | grep Hello
else
  if
    grep 'internal compiler error' compiler.out
  then
    exit 101
  else
    exit 1
  fi
fi
