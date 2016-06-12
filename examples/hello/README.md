# Hello World

This is [a rather nasty example from John Regehr](http://blog.regehr.org/archives/1284).

It takes a C++ implementation of Hello World after a preprocessor has been run
on it and attempts to reduce it to a minimal program that when compiled outputs
"Hello". It also notes anything that triggers an internal compiler error along
the way.

Structureshrink does surprisingly well on this example despite not knowing
anything about the structure of C++ programs.

In some sense this isn't surprising because this was the main example I developed
it against! In another it's quite surprising because C++ has a fairly complicated
structure and structureshrink performs a lot of non-trivial transformations on
the shrunk example such as e.g. renaming variables and methods to have shorter
names.
