from tempfile import mkstemp
import os
import subprocess
import heapq


def sort_key(ls):
    return (len(ls), ls)


def satshrink(initial, criterion):
    if not criterion(initial):
        raise ValueError("Initial example does not satisfy criterion")

    empty = initial[:0]
    if criterion(empty):
        return empty

    n = len(initial)
    character_index = {}
    for i, c in enumerate(initial):
        character_index.setdefault(c, []).append(i)

    # Design: We have a graph of len(initial) + nodes labeled  0 through n.
    # We are trying to find a minimal size colouring on it such that every
    # string in the quotient of the graph by that colouring satisfies
    # criterion. We do this by using suffixes of the graphs to look for
    # inconsistencies and then farming off to a SAT solver to do the hard
    # work of resolving them for us.

    nodes = range(n + 1)

    # We maintain a list of things which we definitely know aren't the same
    # colour. This speed
    forced_colouring = {}
    forced_colouring[0] = 0
    forced_colouring[n] = 1
    inconsistencies = {(0, n)}

    def fix_colour(i):
        if i in forced_colouring:
            return
        if all((i, k) in inconsistencies for k in forced_colouring):
            forced_colouring[i] = len(forced_colouring)

    def check_consistent(i, j):
        if i == j:
            return True
        if (i, j) in inconsistencies:
            return False
        i, j = sorted((i, j))
        if criterion(initial[:i] + initial[j:]):
            return True
        inconsistencies.add((i, j))
        inconsistencies.add((j, i))
        for k in (i, j):
            fix_colour(k)
        return False

    # Do a species of delta debugging to populate some initial inconsistencies.
    i = 0
    while i < n:
        k = (n - i)
        while k > 0:
            if check_consistent(i, i + k):
                i += k
                break
            else:
                i += 1
        k // 2

    unsatisfiable = 0
    satisfiable = n + 1

    # Invariant: There is no valid colouring of size lo, there is a valid
    # covering of size hi

    best_result = initial

    while True:
        unsatisfiable = max(unsatisfiable, len(forced_colouring) - 1)
        assert unsatisfiable < satisfiable
        if unsatisfiable + 1 == satisfiable:
            break
        maybe_satisfiable = (unsatisfiable + satisfiable) // 2
        # We try to construct a colouring of size maybe_satisfiable that
        # satisfies all our known constraints.

        # First we build the SAT problem. We have one variable for each
        # assignment of a colour to a node.

        clauses = []

        colours = range(maybe_satisfiable)

        def variable(node, colour):
            return 1 + (node * maybe_satisfiable) + colour

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
                            variable(i, c1), -variable(i, c2)
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
            break
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
            unsatisfiable = maybe_satisfiable
        else:
            colouring = [None] * (n + 1)
            for assignment in result:
                assert assignment != 0
                if assignment < 0:
                    continue
                t = abs(assignment) - 1
                node = t // maybe_satisfiable
                colour = t % maybe_satisfiable
                assert node in nodes
                assert colour in colours
                if node in forced_colouring:
                    assert forced_colouring[node] == colour
                colouring[node] = colour
            # We now have a consistent colouring of our nodes. This means we
            # can build a DFA out of them.
            dfa = {}
            for i, colour in enumerate(initial):
                assert colour is not None
                transitions = dfa.setdefault(i, {})
                character = initial[i]
                nextcolour = colouring[i + 1]
                assert nextcolour is not None
                if character in transitions:
                    assert transitions[character] == nextcolour
                else:
                    transitions[character] = nextcolour

            assert len(transitions) == n

            end_state = colouring[n]

            # Great! We now have a DFA. Lets build the minimum path through
            # it. We use Dijkstra for this, naturally.
            result = None

            # Queue format: Length of path, value of path, current state
            queue = [(0, [], 0)]
            while True:
                # This must terminate with reaching an end node so the queue
                # should never be observed empty.
                k, path, state = queue.pop()
                assert state != end_state
                assert k == len(path)
                for character, next_state in sorted(
                    transitions[state].items()
                ):
                    if next_state == end_state:
                        result = next_state
                        break
                    heapq.heappush((
                        k + 1, path + [character], next_state
                    ))
            assert result is not None
            if criterion(result):
                satisfiable = maybe_satisfiable
                if sort_key(result) < sort_key(best_result):
                    best_result = result
            else:
                # Not so consistent after all!
                # We've discovered new transitions that we're not allowed to
                # make. Now let's find out what they are.
                path = result

                # After we have read i charracters from path, we are in
                # states[i]
                states = [0]
                for c in path:
                    states.append(transitions[states[-1]][c])
                colours_to_indices = {}
                for i, colour in enumerate(colouring):
                    colours_to_indices[colour] = i
                assert colours_to_indices[end_state] == n

                works = 0
                does_not_work = states[-1]
                while works + 1 < does_not_work:
                    maybe_works = (works + does_not_work) // 2
                    prefix = path[:maybe_works]
                    state = states[i]
                    suffix = initial[colours_to_indices[state]:]
                    if criterion(prefix + suffix):
                        works = maybe_works
                    else:
                        works = does_not_work
                # So the transition we took works -> does_not_work was bad.
                bad_colour = states[does_not_work]
                coloured_indices = [
                    i for i, c in enumerate(colouring) if c == bad_colour]
                assert len(coloured_indices) > 1
                any_bad = False
                for i in coloured_indices:
                    for j in coloured_indices:
                        if not check_consistent(i, j):
                            any_bad = True
                assert any_bad
                # Anything we previously knew about satisfiability is wrong.
                satisfiable = n + 1
    return best_result


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
