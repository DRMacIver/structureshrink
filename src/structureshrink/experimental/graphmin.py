from structureshrink.utils.graphcolouring import colour_graph, \
    colour_graph_with_fixed_vertices


def graphmin(ls, criterion):
    if criterion([]):
        return []
    while True:
        ts = _graphmin_once(ls, criterion)
        if ts == ls:
            return ts
        ls = ts


def _graphmin_once(ls, criterion):
    n = len(ls)
    constraints = set()
    constraints.add((0, n))
    vertices = range(n + 1)
    # Do a species of delta debugging to populate some initial inconsistencies.
    k = n - 1
    while k > 0:
        i = 0
        while i + k <= n:
            ts = ls[:i] + ls[i + k:]
            if criterion(ts):
                return ts
            constraints.add((i, i + k))
            i += k
        k //= 2

    while True:
        initial_constraints = len(constraints)
        colouring = colour_graph(vertices, constraints)
        partitions = {}
        for vertex, colour in colouring.items():
            partitions.setdefault(colour, []).append(vertex)
        partitions = [
            p for p in partitions.values() if 1 < len(p)
        ]
        if not partitions:
            return ls
        checks = sorted(
            [(min(p), max(p)) for p in partitions], key=lambda x: x[0] - x[1]
        )
        for u, v in checks:
            ts = ls[:u] + ls[v:]
            if criterion(ts):
                return ts
            else:
                constraints.add((u, v))
        if len(constraints) == initial_constraints:
            return ls
