This example takes a file from Hypothesis that yapf
introduces an error into (according to flake8, which isn't
en entirely reliable guide) and shrinks it, subject to
the constraints that the original file remains a valid
Python file which flake8 doesn't complain about.

The difficulty of shrinking through entirely valid steps
is the major source of complexity for this example. In
particular both unused and undefined variable warnings
are easy to introduce and hard to avoid.
