from structureshrink.utils.graphcolouring import colour_graph
from hypothesis import given, strategies as st, assume


@given(st.sets(st.tuples(st.integers(0, 1000), st.integers(0, 1000))))
def test_can_colour_arbitrary_graphs(edges):
    colouring = colour_graph(edges)
    assert colouring is not None
    for i, j in edges:
        if i != j:
            assert colouring[i] != colouring[j]


@given(st.sets(st.integers()), st.sets(st.integers()))
def test_can_two_colour_bipartite_graphs(xs, ys):
    xs -= ys
    assume(xs and ys)
    colouring = colour_graph((x, y) for x in xs for y in ys)
    for x in xs:
        for y in ys:
            assert colouring[x] != colouring[y]
    for ts in (xs, ys):
        assert len({colouring[t] for t in ts}) <= 1
