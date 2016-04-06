from collections import Counter
from structureshrink.utils.minisat import minisat


def normalize_edges(edges):
    edges = {(u, v) for u, v in edges if u != v}
    edges |= {(v, u) for u, v in edges}
    return edges


def colour_graph(vertices, edges):
    if not edges:
        return {}
    for result in progressive_colour_graph(vertices, edges):
        pass
    return result


def progressive_colour_graph(vertices, edges):
    edges = normalize_edges(edges)
    vertices = sorted(vertices)
    no_colouring = 0
    has_colouring = len(vertices)
    yield {v: i for i, v in enumerate(vertices)}
    if not edges:
        return

    while no_colouring + 1 < has_colouring:
        check_colouring = (no_colouring + has_colouring) // 2
        colouring = colour_graph_with_fixed_vertices(
            vertices, edges, check_colouring)
        if colouring is not None:
            yield colouring
            has_colouring = check_colouring
        else:
            no_colouring = check_colouring


def colour_graph_with_fixed_vertices(vertices, edges, n_colours):
    if n_colours <= 1:
        if edges:
            return None
        else:
            return {v: 0 for v in vertices}

    colours = range(n_colours)
    degrees = Counter()
    for u, _ in edges:
        degrees[u] += 1

    colouring = {}

    while True:
        options = [
            v for v in vertices
            if all((u, v) in edges for u in colouring)
        ]
        if not options:
            break
        fixed = max(options, key=degrees.__getitem__)
        colouring[fixed] = len(colouring)

    if len(colouring) > n_colours:
        return None
    if len(colouring) == len(vertices):
        return colouring

    vertex_to_index = {v: i for i, v in enumerate(vertices)}

    def variable(node, colour):
        return 1 + vertex_to_index[node] * n_colours + colour

    clauses = []

    for vertex, colour in colouring.items():
        clauses.append((variable(vertex, colour),))

    for n in vertices:
        clauses.append(tuple(
            variable(n, c) for c in colours
        ))

    for c in colours:
        clauses.append(tuple(
            variable(n, c) for c in colours
        ))

    for n in vertices:
        for c1 in colours:
            for c2 in colours:
                if c1 < c2:
                    clauses.append((
                        -variable(n, c1), -variable(n, c2)
                    ))
    for i, j in edges:
        for c in colours:
            clauses.append((
                -variable(i, c), -variable(j, c)
            ))
    clauses.sort(key=lambda x: (len(x), x))

    sat = minisat(clauses)
    if sat is None:
        return None
    for variable in sat:
        if variable < 0:
            continue
        variable -= 1
        node = vertices[variable // n_colours]
        colour = variable % n_colours
        if node in colouring:
            assert colouring[node] == colour
        else:
            colouring[node] = colour
    return colouring
