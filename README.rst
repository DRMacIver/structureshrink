Structured Shrinking
====================

structureshrink is a program and library that attempts to find structure in a
file and uses it to produce smaller examples of the same sort of file.

It considers an example smaller if it contains strictly fewer bytes, or if it
has the same number of bytes but is lexicographically smaller treated as a
sequence of unsigned 8 bit integers (currently structureshrink only does a
small amount to shrink with regards to the latter)

Multishrinking
--------------

Rather than shrinking with respect to a single predicate, structureshrink
insteads shrinks with respect to a *classification*. It takes a function that
maps data to a label. It then tries to produce the smallest example for *each*
label. 

For a command line invocation the label should usually be the exit status of
the program. For library usage it can be any hashable python value.

This has a couple advantages:

1. It allows you to `detect interesting failures that might occur doing the
   course of shrinking <http://blog.regehr.org/archives/1284>`_.
2. Attempts to shrink one label can result in shrinking another label. This
   potentially allows escaping local minima.

Core Algorithm
--------------

At its heart, structureshrink uses a sequence minimisation algorithm. For long
sequences it uses something that is essentially a light optimisation on 
`Delta Debugging <https://en.wikipedia.org/wiki/Delta_Debugging>`_. For shorter
sequences it tries slightly harder to delete *all* sub-intervals of the
sequence (it only does this for shorter sequences because this is an
intrinsically O(n^2) operation).

So far, so uninteresting.

The interesting feature of structureshrink is how this algorithm is applied.

structureshrink extracts a sequence of interesting ngrams from the string. An
ngram is interesting if it appears at least max(2, n) times in the string.

There's no *particularly* principled reason for this choice, except that it
seems to work and it bounds the number of ngrams. A fixed threshold tends to
get a bit excited on highly repetitive data.

We also remember previously useful ngrams and try those even if they don't
satisfy the criterion for the current string. This allows us to infer structure
and then make use of it even when it has become less obvious.

Once we have these ngrams we do several things with them:

1. We split the data by the occurrences of each ngram in it. We then try to
   shrink the sequence of splits by the property that joining by the ngram
   satisfies the desired criterion.
2. We split the data by the occurrences of each ngram in it. We then try to
   shrink the *ngram* (bytewise) such that joining the splits together by that
   ngram satisfies the criterion.

Ngrams are processed in the following order (based on the current best at the
time of calculation. These are not updated as we go):

1. Longer ngrams are processed first
2. Given two ngrams of the same size, process the ones such that smallest
   element of the split is of longest length (because this is the smallest
   amount of data that can be deleted on a successful shrink, so doing these
   first gets us to smaller strings faster).
3. Given two ngrams of the same length and the same smallest split size (this
   usually only happens once the ngrams are length one or two), pick the one
   with the *fewest* splits (because at this stage we're usually more concerned
   with sequence minimizer performance than worst case guarantees).

We also have additional more naive phases:

1. Simply apply the sequence shrinking algorithm bytewise to the data

We go through these phases in sequence. Each time a phase produces useful
changes we go back to the beginning rather than moving on to the next phase.

We stop once no phase produces a change.


Advantages of structureshrink
-----------------------------

Because structureshrink detects the features it needs from the file rather than
proceeding linewise, it is able to cope with a much wider range of file
formats, both text and binary.

It also produces much smaller examples than simple linewise or space wise
deletion - for example I've seen it happily rename variables in a C++ program
despite knowing literally nothing about the grammar of C++.

This can also be a downside, as aggressively minimized programs are not very
readable. To compensate for that structureshrink lets you specify a
preprocessor for formatting your data (e.g. clang-format). This runs before
shrinking, and can also have the advantage that it speeds up the shrink by
removing useless shrinks (at least, as long as your preprocessor is faster than
your test program).


Usage
-----

There's a library and a commannd line tool. Neither are what I would describe
as documented.

To use the command line tool run:

.. code-block::

    python setup.py install

This will require Python 3.4+.

You can now use the structureshrink command.

Usage is that you pass it a file to be shrunk (this will have its contents
replaced and a backup file created) and a command to run. The results will
be in the 'shrinks' directory, one per exit status for the command seen.

Development status
------------------

Somewhere between "Research prototype" and "Usable tool", but much closer to
the first one. It seems to work pretty well, and it's not completely fragile,
but it's definitely rough around the edges. It's certainly not going to
maintain backwards compatibility.

It's not particularly well tested right now (by my standards it barely counts
as tested at all), so it's probably broken in amusing ways.
