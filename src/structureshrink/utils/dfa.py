from structureshrink.utils.minisat import minisat
import heapq


class _Unknown(object):
    def __repr__(self):
        return "unknown"

    def __bool__(self):
        return False

unknown = _Unknown()


class DFA(object):
    def __init__(self, states):
        states = list(states)
        self.__labels = [label for label, _ in states]
        self.__transitions = [dict(transitions) for _, transitions in states]

    @property
    def start(self):
        return 0

    @property
    def states(self):
        return range(len(self.__labels))

    def label(self, state):
        return self.__labels[state]

    def transition(self, state, c):
        return self.__transitions[state][c]

    def transitions(self, state):
        return sorted(self.__transitions[state].items())

    def __repr__(self):
        return "DFA(%r)" % (list(zip(self.__labels, self.__transitions)),)

    def match(self, string):
        state = self.start
        for c in string:
            try:
                state = self.transition(state, c)
            except KeyError:
                return unknown
        return self.label(state)

    def find_label_matching(self, predicate):
        seen = set()
        queue = [(0, b'', 0)]
        while queue:
            _, path, state = heapq.heappop(queue)
            if state in seen:
                continue
            if predicate(self.label(state)):
                return bytes(path)
            seen.add(state)
            for c, newstate in self.transitions(state):
                newpath = path + bytes([c])
                heapq.heappush(
                    queue, (len(newpath), newpath, newstate)
                )
        raise ValueError("No labels match predicate")


class PTABuilder(object):
    def __init__(self):
        self.initial = [unknown, {}, 0]
        self.states = [self.initial]

    def add(self, string, label):
        state = self.initial
        for c in string:
            try:
                state = state[1][c]
            except KeyError:
                newstate = [unknown, {}, len(self.states)]
                self.states.append(newstate)
                state[1][c] = newstate
                state = newstate
        if state[0] is not unknown and state[0] != label:
            raise ValueError(
                "Label %r conflicts with existing label %r" % (
                    label, state[0]))
        state[0] = label

    def build(self):
        return DFA(
            (label, {c: state[-1] for c, state in transitions.items()})
            for label, transitions, _ in self.states
        )


def build_pta(corpus):
    builder = PTABuilder()
    for string, label in corpus:
        builder.add(string, label)
    return builder.build()


def reduce_dfa(dfa):
    colouring = None
    no_colouring = 0
    has_colouring = len(dfa.states)
    while no_colouring + 1 < has_colouring:
        maybe = (no_colouring + has_colouring) // 2
        attempt = colour_dfa(dfa, maybe)
        if attempt is not None:
            colouring = attempt
            has_colouring = maybe
        else:
            no_colouring = maybe
    if colouring is not None:
        skeleton = [[unknown, {}] for _ in range(has_colouring)]
        for state in dfa.states:
            colour = colouring[state]
            l = dfa.label(state)
            if l is not unknown:
                if skeleton[colour][0] is unknown:
                    skeleton[colour][0] = l
                else:
                    assert skeleton[colour][0] == l
            for c, t in dfa.transitions(state):
                check = skeleton[colour][1].setdefault(c, colouring[t])
                assert check == colouring[t]
        return DFA(skeleton)
    return dfa


def colour_dfa(dfa, n_colours):
    label_witnesses = {}
    for state in dfa.states:
        l = dfa.label(state)
        if l is not unknown:
            label_witnesses.setdefault(l, state)
    forced_colouring = {}
    for c, state in enumerate(sorted(label_witnesses.values())):
        forced_colouring[state] = c
    if len(forced_colouring) > n_colours:
        return None

    colours = range(n_colours)

    def variable(state, colour):
        assert state in dfa.states
        assert colour in colours
        return 1 + n_colours * state + colour

    # Force a particular set of colours for a witness of each label for
    # symetry breaking.
    clauses = [
        (variable(state, colour),)
        for state, colour in forced_colouring.items()
    ]

    # Every variable gets a colour
    for i in dfa.states:
        clauses.append(tuple(
            variable(i, c)
            for c in colours
        ))

    # Every colour gets a variable
    for c in colours:
        clauses.append(tuple(
            variable(i, c)
            for i in dfa.states
        ))

    # Every variable gets at most one colour
    for c1 in colours:
        for c2 in colours:
            if c1 < c2:
                for i in dfa.states:
                    clauses.append((-variable(i, c1), -variable(i, c2)))

    # Two nodes with different labels get distinct colours
    # FIXME: This can be written better.
    for i in dfa.states:
        if dfa.label(i) is not unknown:
            for j in dfa.states:
                if dfa.label(j) is not unknown:
                    if dfa.label(i) != dfa.label(j):
                        for c in colours:
                            clauses.append((
                                -variable(i, c), -variable(j, c)
                            ))

    transitions = {}
    for i in dfa.states:
        for c, j in dfa.transitions(i):
            transitions.setdefault(c, []).append((i, j))

    # If two nodes with a transition on c get the same colour then their target
    # transition for c gets the same colour.
    # FIXME: This too can be better written.
    for ts in transitions.values():
        for u1, v1 in ts:
            for u2, v2 in ts:
                if u1 != u2:
                    for c1 in colours:
                        for c2 in colours:
                            if c1 != c2:
                                clauses.append((
                                    -variable(u1, c1), -variable(u2, c1),
                                    -variable(v1, c2), variable(v2, c2)
                                ))

    clauses.sort()
    solution = minisat(clauses)
    if solution is None:
        return None
    colouring = [None] * len(dfa.states)
    for assignment in solution:
        assert assignment != 0
        if assignment < 0:
            continue
        t = abs(assignment) - 1
        node = t // n_colours
        colour = t % n_colours
        assert node in dfa.states
        assert colour in colours
        if node in forced_colouring:
            assert forced_colouring[node] == colour
        colouring[node] = colour
    return colouring
