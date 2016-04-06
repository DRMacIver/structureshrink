import heapq
from collections import Counter
from structureshrink.utils.minisat import minisat


def sort_key(ls):
    return (len(ls), ls)


def satshrink(initial, criterion):
    prev = None
    while prev != initial:
        prev = initial
        initial = satshrink_step(initial, criterion)
    return initial


def satshrink_step(initial, criterion):
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

    mark_inconsistent(0, n)

    prev = -1

    no_colouring = 1

    while True:
        assert len(inconsistencies) > prev
        prev = len(inconsistencies)

        n_colours = no_colouring + 1

        while True:
            colouring = colour_linear_dfa(initial, inconsistencies, n_colours)
            if colouring is not None:
                break
            no_colouring = n_colours
            if n_colours == n:
                # There is no colouring of the graph with fewer than n states.
                return initial
            elif n_colours * 2 > n:
                n_colours = (1 + n + n_colours) // 2
            else:
                n_colours *= 2
        while no_colouring + 1 < n_colours:
            maybe_n_colours = (n_colours + no_colouring) // 2
            maybe_colouring = colour_linear_dfa(
                initial, inconsistencies, maybe_n_colours)
            if maybe_colouring is not None:
                n_colours = maybe_n_colours
                colouring = maybe_colouring
            else:
                no_colouring = maybe_n_colours

        # We now have a consistent colouring of our nodes. This means we
        # can build a DFA out of them.
        colours = sorted(set(colouring))
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
        visited = set()
        while result is None:
            # This must terminate with reaching an end node so the queue
            # should never be observed empty.
            assert queue
            k, path, states = heapq.heappop(queue)
            assert len(states) == len(path) + 1
            assert len(path) <= len(initial)
            assert k == len(path)
            state = states[-1]
            if state in visited:
                continue
            visited.add(state)
            for character, next_state in sorted(dfa[state].items()):
                if next_state == end_state:
                    result = path + [character]
                    break
                value = (
                    k + 1, path + [character], states + [next_state]
                )
                heapq.heappush(queue, value)
        assert result is not None
        if len(result) < len(initial) and criterion(result):
            return result
        # Our DFA is clearly wrong. We now either find a shrink or improve
        # it, whichever comes first.
        nodes_by_colour = {}
        for node, colour in zip(nodes, colouring):
            nodes_by_colour.setdefault(colour, []).append(node)
        tuples = []
        for ls in nodes_by_colour.values():
            if len(ls) == 1:
                continue
            assert ls == sorted(ls)
            tuples.append((ls[0], ls[-1]))
        # Largest to smallest
        tuples.sort(key=lambda x: x[0] - x[1])
        for i, j in tuples:
            ts = initial[:i] + initial[j:]
            if criterion(ts):
                return ts
            assert (i, j) not in inconsistencies
            mark_inconsistent(i, j)

    assert False


def colour_linear_dfa(sequence, inconsistencies, n_colours):
    """Given the linear DFA sequence with known node inconsistencies, provide
    a minimal consistent colouring compatible with these."""
    n = len(sequence)

    character_index = {}
    for i, c in enumerate(sequence):
        character_index.setdefault(c, []).append(i)

    nodes = range(n + 1)

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

    # First we build the SAT problem. We have one variable for each
    # assignment of a colour to a node.

    clauses = []

    colours = range(n_colours)

    def variable(node, colour):
        return 1 + (node * n_colours) + colour

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

    # Every colour is used
    for c in colours:
        clauses.append(tuple(
            variable(n, c) for n in nodes
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
        return None
    else:
        colouring = [None] * (n + 1)
        for assignment in result:
            assert assignment != 0
            if assignment < 0:
                continue
            t = abs(assignment) - 1
            node = t // n_colours
            colour = t % n_colours
            assert node in nodes
            assert colour in colours
            if node in forced_colouring:
                assert forced_colouring[node] == colour
            colouring[node] = colour
    assert len(colouring) == len(sequence) + 1
    for i, j in inconsistencies:
        assert colouring[i] != colouring[j]
    colours = sorted(set(colouring))
    assert colours == list(range(len(colours)))
    return colouring
