#!/usr/bin/env bash

set -e -x

pdftotext test-case.pdf test-case.txt

grep 'hello world' test-case.txt
