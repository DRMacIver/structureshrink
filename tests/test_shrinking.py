from structureshrink import shrink, Volume
from hypothesis import given, strategies as st
import hashlib


@given(st.binary())
def test_partition_by_length(b):
    shrunk = shrink(b, len)
    assert len(shrunk) == len(b) + 1


@given(st.lists(st.binary(min_size=1, max_size=4), min_size=1, max_size=5))
def test_shrink_to_any_substring(ls):
    shrunk = shrink(
        b''.join(ls), lambda x: sum(l in x for l in ls)
    )
    assert len(shrunk) >= len(ls)


def test_partition_by_last_byte():
    seed = b''.join(bytes([i, j]) for i in range(256) for j in range(256))
    shrunk = shrink(
        seed, lambda s: hashlib.sha1(s).digest()[-1] & 127
    )
    assert len(shrunk) == 128
