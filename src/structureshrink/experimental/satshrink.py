from tempfile import mkstemp
import os
import subprocess
import heapq
from collections import Counter


def sort_key(ls):
    return (len(ls), ls)


def satshrink(initial, criterion):
    if not criterion(initial):
        raise ValueError("Initial example does not satisfy criterion")

    if criterion([]):
        return []

    if len(initial) <= 1:
        return initial

    # Design: We have a graph of len(initial) + nodes labeled  0 through n.
    # We are trying to find a minimal size colouring on it such that every
    # string in the quotient of the graph by that colouring satisfies
    # criterion. We do this by using suffixes of the graphs to look for
    # inconsistencies and then farming off to a SAT solver to do the hard
    # work of resolving them for us.

    n = len(initial)
    nodes = range(n + 1)
    inconsistencies = set()

    def mark_inconsistent(i, j):
        assert i != j
        assert 0 <= i <= len(initial)
        assert 0 <= j <= len(initial)
        inconsistencies.add((i, j))
        inconsistencies.add((j, i))

    def check_consistent(i, j):
        if i == j:
            return True
        if (i, j) in inconsistencies:
            return False
        i, j = sorted((i, j))
        if criterion(initial[:i] + initial[j:]):
            return True
        mark_inconsistent(i, j)
        return False

    # Do a species of delta debugging to populate some initial inconsistencies.
    i = 0
    k = n
    while i < n:
        k = (n - i)
        while k > 0:
            assert i + k <= len(initial)
            if check_consistent(i, i + k):
                i += k
                break
            k //= 2
        else:
            i += 1

    best_result = initial
    prev = -1

    while True:
        assert len(inconsistencies) > prev
        prev = len(inconsistencies)
        colouring = colour_linear_dfa(initial, inconsistencies)
        assert len(colouring) == len(initial) + 1

        for i, j in inconsistencies:
            assert colouring[i] != colouring[j]

        if len(set(colouring)) == len(nodes):
            assert best_result == initial
            # There is no shorter colouring. This is minimal.
            return initial
        # We now have a consistent colouring of our nodes. This means we
        # can build a DFA out of them.
        colours = sorted(set(colouring))
        assert colours == list(range(len(colours)))
        assert len(nodes) == len(colouring)
        dfa = {}
        for i, colour in zip(nodes[:-1], colouring):
            assert colour is not None
            transitions = dfa.setdefault(colour, {})
            character = initial[i]
            nextcolour = colouring[i + 1]
            assert nextcolour in colours
            assert nextcolour is not None
            if character in transitions:
                assert transitions[character] == nextcolour
            else:
                transitions[character] = nextcolour
        assert set(dfa).issubset(set(colours))
        start_state = colouring[0]
        end_state = colouring[n]
        assert start_state != end_state

        # Great! We now have a DFA. Lets build the minimum path through
        # it. We use Dijkstra for this, naturally.
        result = None

        # Queue format: Length of path, value of path, current state
        queue = [(0, [], [start_state])]
        while result is None:
            # This must terminate with reaching an end node so the queue
            # should never be observed empty.
            assert queue
            k, path, states = heapq.heappop(queue)
            assert len(states) == len(path) + 1
            assert len(path) <= len(initial)
            assert k == len(path)
            for character, next_state in sorted(
                dfa[states[-1]].items()
            ):
                if next_state == end_state:
                    result = path + [character]
                    break
                value = (
                    k + 1, path + [character], states + [next_state]
                )
                heapq.heappush(queue, value)
        assert result is not None
        if criterion(result):
            return result
        else:
            # We know that start_state and end_state are separate already.
            assert result
            path = result
            breakdown = {}
            for colour, node in zip(colouring, nodes):
                breakdown.setdefault(colour, []).append(node)
            start_is_bad = False
            for p in breakdown[start_state]:
                if not criterion(initial[p:]):
                    mark_inconsistent(0, p)
                    start_is_bad = True
            if not start_is_bad:
                suffixes = [max(breakdown[c]) for c in colours]
                assert suffixes[end_state] == n
                states = [0]
                for p in path:
                    states.append(dfa[states[-1]][p])
                consistent = 0
                inconsistent = len(path)
                while consistent + 1 < inconsistent:
                    check_consistent = (consistent + inconsistent) // 2
                    prefix = path[:check_consistent]
                    suffix = initial[suffixes[states[check_consistent]]:]
                    if criterion(prefix + suffix):
                        consistent = check_consistent
                    else:
                        inconsistent = check_consistent
                inconsistent_colour = states[inconsistent]
                consistent_colour = states[consistent]
                transition_character = path[consistent]
                sources = [
                    i + 1 for i in breakdown[consistent_colour]
                    if i < n and initial[i] == transition_character
                ]
                assert sources
                targets = breakdown[inconsistent_colour]
                assert targets

                experiment = initial[suffixes[inconsistent_colour]:]

                check = len(inconsistencies)
                for s in sources:
                    for t in targets:
                        assert (s, t) not in inconsistencies
                        if criterion(initial[:s] + experiment) != criterion(
                            initial[:t] + experiment
                        ):
                            mark_inconsistent(s, t)
                assert len(inconsistencies) > check
    assert False


def colour_linear_dfa(sequence, inconsistencies):
    """Given the linear DFA sequence with known node inconsistencies, provide
    a minimal consistent colouring compatible with these."""
    n = len(sequence)

    character_index = {}
    for i, c in enumerate(sequence):
        character_index.setdefault(c, []).append(i)

    nodes = range(n + 1)
    no_colouring = 0
    has_colouring = len(nodes)

    inconsistency_counts = Counter()
    for i, _ in inconsistencies:
        inconsistency_counts[i] += 1
    forced_colouring = {0: 0}

    # First do a greedy colouring that tries to mark a colour for as many
    # nodes as possible. This lets us break symmetries that might otherwise
    # cause us problems.

    if inconsistency_counts:
        while True:
            extra = [
                n for n in nodes if all(
                    (k, n) in inconsistencies for k in forced_colouring)]
            if not extra:
                break
            forced_colouring[
                max(extra, key=lambda s: (inconsistency_counts[s], s))
            ] = len(forced_colouring)

    colouring = None

    while no_colouring + 1 < has_colouring:
        could_have_colouring = (no_colouring + has_colouring) // 2

        # We try to construct a colouring of size maybe_satisfiable that
        # satisfies all our known constraints.

        # First we build the SAT problem. We have one variable for each
        # assignment of a colour to a node.

        clauses = []

        colours = range(could_have_colouring)

        def variable(node, colour):
            return 1 + (node * could_have_colouring) + colour

        # First add the fixed colours.
        for node, colour in forced_colouring.items():
            clauses.append((variable(node, colour),))

        # Everything gets a colour
        for i in nodes:
            clauses.append(tuple(
                variable(i, c)
                for c in colours
            ))

        # Nothing gets more than one colour
        for i in nodes:
            for c1 in colours:
                for c2 in colours:
                    if c1 != c2:
                        clauses.append((
                            -variable(i, c1), -variable(i, c2)
                        ))

        # Now add the known inconsistencies
        for i, j in inconsistencies:
            assert i != j
            for colour in colours:
                clauses.append((
                    -variable(i, colour),
                    -variable(j, colour),
                ))

        # Now we force transition consistency.
        # FIXME: This is a really bad number of clauses and we can do
        # better.
        for indices in character_index.values():
            if len(indices) <= 1:
                continue
            # ¬(c[i, c1] ^ c[j, c1] ^ c[i + 1, c2] ^ ¬c[j + 1, c2]) ->
            # ¬c[i, c1] v ¬c[j, c1] v ¬c[i + 1, c2] v c[j + 1, c2]
            for i in indices:
                for j in indices:
                    if i == j:
                        continue
                    for c1 in colours:
                        for c2 in colours:
                            if c1 == c2:
                                continue
                            clauses.append((
                                -variable(i, c1), -variable(j, c1),
                                -variable(i + 1, c2), variable(j + 1, c2)
                            ))

        # Sort to pass to minisat in a consistent order.
        clauses.sort(key=lambda s: (len(s), s))
        result = minisat(clauses)
        if result is None:
            no_colouring = could_have_colouring
        else:
            has_colouring = could_have_colouring
            colouring = [None] * (n + 1)
            for assignment in result:
                assert assignment != 0
                if assignment < 0:
                    continue
                t = abs(assignment) - 1
                node = t // could_have_colouring
                colour = t % could_have_colouring
                assert node in nodes
                assert colour in colours
                if node in forced_colouring:
                    assert forced_colouring[node] == colour
                colouring[node] = colour
    if colouring is None:
        return list(nodes)
    return colouring


def minisat(clauses):
    # Returns None if unsatisfiable, else a variable assignment.
    if not clauses:
        return []

    n_variables = max(max(abs(c) for c in cs) for cs in clauses)
    n_clauses = len(clauses)

    satfilename = outfilename = None

    try:
        satfd, satfilename = mkstemp(suffix='.sat')
        outfd, outfilename = mkstemp(suffix='.out')
        os.close(outfd)
        satfile = os.fdopen(satfd, mode='w')

        satfile.write(
            'p cnf %d %d\n' % (n_variables, n_clauses)
        )
        for c in clauses:
            satfile.write(
                '%s 0\n' % (' '.join(map(str, c)),)
            )
        satfile.close()

        try:
            subprocess.check_output([
                "minisat", satfilename, outfilename,
            ])
            return None
        except subprocess.CalledProcessError as e:
            # Due to reasons, apparently an exit code of 10 signifies SAT
            if e.returncode == 20:
                return None
            if e.returncode != 10:
                raise
        with open(outfilename, 'r') as o:
            l1, l2 = o.readlines()
        assert l1.strip() == 'SAT'
        result = list(map(int, l2.strip().split()))
        term = result.pop()
        assert term == 0
        assert 0 not in result
        assert all(abs(i) <= n_variables for i in result)
        return result
    finally:
        for f in (satfilename, outfilename):
            if f is not None:
                os.unlink(f)
