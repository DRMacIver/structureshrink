from functools import total_ordering
import heapq
from collections import Counter


def fixate(shrinker):
    """Take a shrinker (either sort) and turn it into one that runs to fixation
    """
    def accept(self, target, criterion):
        self.debug("Running %s to fixation" % (shrinker.__name__,))
        assert target is not None
        prev = None
        while prev != target:
            prev = target
            target = shrinker(self, target, criterion)
        return target
    accept.__name__ = 'fixate(%s)' % (shrinker.__name__)
    return accept


def one_pass_delete(k):
    """Partition shrinker that tries to delete all sequences of length k"""

    def accept(self, target, criterion):
        target = list(target)

        def calc():
            return sorted(
                range(len(target) + 1 - k),
                key=lambda i: shortlex(b''.join(target[i:i+k])), reverse=True)

        indices = calc()

        i = 0
        while i < len(indices):
            j = indices[i]
            assert j + k <= len(target)
            ts = list(target)
            del ts[j:j+k]
            assert len(ts) + k == len(target)
            if criterion(ts):
                target = ts
                indices = calc()
                continue
            i += 1
        return target
    accept.__name__ = "one_pass_delete(%d)" % (k,)
    return accept


def shortlex(b):
    return (len(b), b)


def tokenwise(tokens):
    tokens = sorted(tokens, key=shortlex, reverse=True)

    def accept(partition_shrinker):
        def shrink(self, target, criterion):

            self.debug("Tokenwise for %d tokens" % (len(tokens),))
            assert isinstance(target, bytes)
            used = set()
            assert criterion(target)
            counts = Counter(tokens)
            assert counts[b''] == 0
            interesting = list(counts)

            def make_queue():
                def score(t):
                    p = target.split(t)
                    if len(p) <= 2:
                        return float('inf')
                    return -(len(t) + sum(map(len, p)) / len(p))

                q = [
                    ((score(t)), ReversedKey(shortlex(t)))
                    for t in interesting
                    if t not in used
                ]
                heapq.heapify(q)
                return q
            queue = make_queue()

            while queue:
                t = heapq.heappop(queue)[1].target[1]
                used.add(t)
                ls = target.split(t)
                if len(ls) <= 2:
                    continue
                self.debug("Partioning by %r into %d parts and running %s" % (
                    t, len(ls), partition_shrinker.__name__,
                ))
                ts = partition_shrinker(
                    self, ls, lambda x: criterion(t.join(x)),
                )
                if ts != ls:
                    target = t.join(ts)
                    queue = make_queue()
            return target
        shrink.__name__ = 'tokenwise(%d tokens, %s)' % (
            len(tokens), partition_shrinker.__name__)
        return shrink
    return accept


@total_ordering
class ReversedKey(object):
    def __init__(self, target):
        self.target = target

    def __eq__(self, other):
        return isinstance(other, ReversedKey) and (self.target == other.target)

    def __lt__(self, other):
        return self.target > other.target


def intercalate(parts, t):
    result = []
    for i, p in enumerate(parts):
        if i > 0:
            result.append(t)
        result.append(p)
    return result


EXP_THRESHOLD = 5


def _expmin(self, partition, criterion):
    if criterion([]):
        return []

    def sort_key(b):
        return shortlex(b''.join(b))

    subsets = []
    for s in range(1, 2 ** len(partition) - 1):
        bits = []
        for x in partition:
            if s & 1:
                bits.append(x)
            s >>= 1
            if not s:
                break
        subsets.append(bits)
    subsets.sort(key=lambda b: (len(b), sort_key(b)))
    for bits in subsets:
        if criterion(bits):
            return bits
    return partition


def exp_or_bust(self, partition, criterion):
    partition = list(partition)
    if len(partition) <= EXP_THRESHOLD:
        return _expmin(self, partition, criterion)
    else:
        return partition


def reheating(step):
    def accept(self, partition, criterion):
        k_bound = float('inf')
        while k_bound > 1:
            k = 1
            while k < k_bound:
                self.debug("k=%d" % (k,))
                if len(partition) <= EXP_THRESHOLD:
                    self.explain("Switching to expmin for %d parts" % (
                        len(partition),
                    ))
                    return _expmin(self, partition, criterion)
                shrunk = step(self, partition, criterion, k)
                if shrunk != partition:
                    partition = shrunk
                    if k_bound > 1:
                        k += 1
                else:
                    k_bound = k
                    k = 1
        return partition
    accept.__name__ = "reheating(%s)" % (step.__name__,)
    return accept


def delete_similar(self, partition, criterion, k):
    indices = list(range(len(partition)))
    self.random.shuffle(indices)
    for i in indices:
        count = k
        for j in range(i + 1, len(partition)):
            if partition[j] == partition[i]:
                count -= 1
                if count == 0:
                    shrunk = partition[:i] + partition[j:]
                    assert len(shrunk) < len(partition)
                    if criterion(shrunk):
                        return shrunk
                    else:
                        break
    return partition


def rand_shrink(self, partition, criterion):
    eager = True
    k_bound = len(partition)
    while k_bound > 1:
        k = 1
        while k < k_bound:
            if len(partition) <= EXP_THRESHOLD:
                self.explain("Switching to expmin for %d parts" % (
                    len(partition),
                ))
                return _expmin(self, partition, criterion)
            indices = list(range(0, len(partition) - k))
            self.random.shuffle(indices)
            for i in indices:
                shrunk = partition[:i] + partition[i + k:]
                assert len(shrunk) + k == len(partition)
                self.explain("Deleting block of size %d" % (k,))
                if criterion(shrunk):
                    partition = shrunk
                    if k_bound > 1:
                        if eager:
                            k *= 2
                        else:
                            k += 1
                        break
            else:
                k_bound = k
                k = 1
                eager = False
    return partition


def full_partition_min(self, partition, criterion):
    partition = rand_shrink(self, partition, criterion)
    if len(partition) <= EXP_THRESHOLD:
        return partition

    for k in range(1, 17):
        @fixate
        def run(self, partition, criterion):
            partition = list(partition)
            if len(partition) <= EXP_THRESHOLD:
                return partition
            else:
                return one_pass_delete(k)(self, partition, criterion)
        partition = run(self, partition, criterion)
        if len(partition) <= EXP_THRESHOLD:
            return _expmin(self, partition, criterion)
    return partition


def with_partition(partition_function, partition_shrink):
    def accept(self, string, criterion):
        p = partition_function(string)
        to_s = b''.join
        return to_s(partition_shrink(self, p, lambda p: criterion(to_s(p))))
    accept.__name__ = 'with_partition(%s, %s)' % (
        partition_function.__name__, partition_shrink.__name__
    )
    return accept
