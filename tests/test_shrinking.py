from structureshrink import Shrinker, Volume
from hypothesis import given, strategies as st, settings, note


@settings(max_examples=10**6, max_iterations=10**6, timeout=-1)
@given(st.data())
def test_partition_by_length(data):
    labels = data.draw(
        st.lists(st.integers(), unique=True, min_size=2), label="labels")

    def classify(s):
        return data.draw(
            st.sampled_from(labels),
            label="classify(%r)" % (s,))
        

    shrinker = Shrinker(
        data.draw(st.binary(min_size=1)), classify, volume=Volume.debug)
    shrinker.output = note

    passes = {}

    def pass_enabled(p):
        try:
            return passes[p]
        except KeyError:
            result = data.draw(st.booleans(), label=p)
            passes[p] = result
            return result

    shrinker.pass_enabled = pass_enabled 

    shrinker.shrink()
