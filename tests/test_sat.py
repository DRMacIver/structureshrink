from structureshrink.experimental.satshrink import colour_linear_dfa, \
    satshrink
from hypothesis import given, strategies as st, assume, example


def test_colour_everything_the_same():
    colouring = colour_linear_dfa([1, 2, 3], set(), 1)
    assert len(set(colouring)) == 1


def test_colour_two_nodes_different():
    colouring = colour_linear_dfa([1, 2, 3], {(0, 2)}, 2)
    assert len(set(colouring)) == 2


@st.composite
def graph_problem(draw):
    alphabet = st.sampled_from(draw(st.lists(st.integers(), min_size=1)))
    sequence = draw(st.lists(alphabet))
    indices = st.integers(0, len(sequence))
    inconsistencies = draw(
        st.sets(
            st.tuples(indices, indices),
            average_size=max(1, (len(sequence) ** 2) / 2)))
    inconsistencies = {t for t in inconsistencies if t[0] != t[1]}
    inconsistencies |= {(v, u) for u, v in inconsistencies}
    assume(len(inconsistencies) <= 100)
    return sequence, inconsistencies


@given(graph_problem(), st.data())
def test_colour_arbitrary_problem(problem, data):
    sequence, inconsistencies = problem
    return
    colouring = colour_linear_dfa(*problem)
    assert len(colouring) == len(sequence) + 1
    palette = set(colouring)
    len(palette) <= len(sequence) + 1
    assert palette
    assert palette == set(range(max(palette) + 1))

    for u, v in inconsistencies:
        assert 0 <= u <= len(sequence)
        assert 0 <= v <= len(sequence)
        assert colouring[u] != colouring[v]

    for i in range(len(sequence)):
        for j in range(len(sequence)):
            if colouring[i] == colouring[j] and sequence[i] == sequence[j]:
                assert colouring[i + 1] == colouring[j + 1]

    nodes = range(len(sequence) + 1)

    if len(palette) < len(nodes):
        duplicates = [
            (i, j)
            for i in nodes
            for j in nodes
            if i != j
            and colouring[i] == colouring[j]
        ]
        assert duplicates
        u, v = data.draw(st.sampled_from(duplicates))
        inconsistencies.add((u, v))
        newcolouring = colour_linear_dfa(sequence, inconsistencies)
        assert newcolouring[u] != newcolouring[v]
        assert len(set(newcolouring)) >= len(palette)


@given(st.lists(st.integers()), st.integers(0, 10))
def test_find_short_sequence(ls, n):
    assume(n <= len(ls))
    shrunk = satshrink(ls, lambda ls: len(ls) >= n)
    assert len(shrunk) == n


seq = st.lists(st.integers(), min_size=1)


@example(x=[0], y=[0], z=[1, 0])
@given(seq, seq, seq)
def test_find_the_hole(x, y, z):
    def f(ls):
        return ls[:len(x)] == x and ls[-len(z):] == z
    assert f(x + z)
    shrunk = satshrink(x + y + z, f)
    assert len(shrunk) <= len(x + z)
