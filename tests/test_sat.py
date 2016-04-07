from structureshrink.experimental.satshrink import satshrink
from hypothesis import given, strategies as st, example, assume


@example(b'\x00\x00\x01', 2)
@example(b'\x00\x00\x00', 1)
@example(b'\0\0', 1)
@given(st.binary(min_size=1), st.integers(0, 10))
def test_satshrink_to_one(b, n):
    assume(n <= len(b))
    x = satshrink(b, lambda s: len(s) >= n)
    assert len(x) == n
