import hashlib
from collections import OrderedDict, Counter
from enum import IntEnum
from functools import cmp_to_key


class Volume(IntEnum):
    quiet = 0
    normal = 1
    debug = 2


def sort_key(s):
    return (len(s), s)


def cache_key(s):
    if len(s) < 20:
        return s
    return hashlib.sha1(s).digest()


ALPHABET = [bytes([b]) for b in range(256)]


class Shrinker(object):

    def __init__(
        self,
        initial, classify, *,
        preprocess=None, shrink_callback=None, printer=None,
        volume=Volume.quiet, principal_only=False,
        passes=None
    ):
        self.__shrink_callback = shrink_callback or (lambda s, r: None)
        self.__printer = printer or (lambda s: None)
        self.__inital = initial
        self.__classify = classify
        self.__preprocess = preprocess or (lambda s: s)
        self.__volume = volume

        self.__cache = {}
        self.__preprocess_cache = {}
        self.__best = OrderedDict()
        self.__regions = {}

        self.shrinks = 0
        preprocessed = self.__preprocess(initial)
        if preprocessed is None:
            raise ValueError('Initial example is rejected by preprocessing')
        label = self.classify(preprocessed)
        self.output('Initial example: %s, labelled %r' % ((
            '%d bytes ' % (len(initial),)
            if initial == preprocessed
            else '%d bytes (%d preprocessed)' % (
                len(initial), len(preprocessed))),
            label))
        self.__initial_label = label
        self.principal_only = principal_only
        self.passes = passes

    def pass_enabled(self, pass_name):
        if self.passes is None or pass_name in self.passes:
            self.output('Running pass %r' % (pass_name,))
            return True
        self.debug('Skipping pass %r' % (pass_name,))
        return False

    def output(self, text):
        if self.__volume >= Volume.normal:
            self.__printer(text)

    def debug(self, text):
        if self.__volume >= Volume.debug:
            self.__printer(text)

    @property
    def best(self):
        return self.__best

    def classify(self, string):
        key = cache_key(string)
        try:
            return self.__cache[key]
        except KeyError:
            pass

        keys = [key]

        preprocessed = self.__preprocess(string)
        if preprocessed is None:
            result = None
        else:
            string = preprocessed
            preprocess_key = cache_key(preprocessed)
            keys.append(preprocess_key)
            try:
                result = self.__cache[preprocess_key]
            except KeyError:
                result = self.__classify(preprocessed)
            if (
                result not in self.best or
                sort_key(string) < sort_key(self.best[result])
            ):
                self.__regions.pop(result, None)
                self.shrinks += 1
                if self.best:
                    if result not in self.best:
                        self.output((
                            'Shrink %d: Discovered new label %r'
                            ' with %d bytes') % (
                                self.shrinks, result, len(string)))
                    else:
                        deletes = len(self.best[result]) - len(string)
                        if deletes == 0:
                            shrink_message = 'lowered %d' % (
                                len([1 for u, v in zip(
                                    string, self.best[result]) if u < v]),)
                        else:
                            shrink_message = 'deleted %d' % (deletes,)

                        self.output(
                            'Shrink %d: Label %r now %d bytes (%s)' % (
                                self.shrinks, result, len(string),
                                shrink_message))
                self.__shrink_callback(string, result)
                self.__best[result] = string
        for k in keys:
            self.__cache[k] = result
        return result

    def __suitable_ngrams(self, label):
        self.debug('Calculating ngrams for %r' % (label,))
        found_ngrams = ngrams(self.best[label])
        self.debug('Found %d ngrams' % len(found_ngrams),)
        return found_ngrams

    def bracket_shrink(self, string, criterion, threshold=1.0):
        prev = None
        while prev != string:
            prev = string
            for l, r in detect_possible_brackets(string):
                intervals = intervals_for_brackets(string, l, r)
                if intervals is None:
                    continue
                intervals.sort(
                    key=lambda x: (x[0] - x[1], x[0]))
                self.debug('Shrinking for bracketed pair %r, %r' % (
                    bytes([l]), bytes([r])
                ))
                changed = True
                while changed:
                    changed = False
                    i = 0
                    while i < len(intervals):
                        u, v = intervals[i]
                        for t in [
                            string[:u] + string[v:],
                            string[:u + 1] + string[v - 1:],
                            string[:u] + string[u + 1:v - 1] + string[v:],
                        ]:
                            if (
                                len(t) < len(string) * threshold and
                                criterion(t)
                            ):
                                string = t
                                intervals = intervals_for_brackets(
                                    string, l, r)
                                changed = True
                                break
                        else:
                            i += 1
                        if intervals is None:
                            break
        return string

    def delete_characters(self, string, criterion):
        counts = Counter(string)
        for c in sorted(range(256), key=counts.__getitem__, reverse=True):
            if c not in string:
                continue
            c = bytes([c])
            t = string.replace(c, b'')
            if criterion(t):
                self.debug('Removed %r' % (c,))
                string = t

    def partition_charwise(self, string, criterion):
        counts = Counter(string)
        alphabet = sorted(counts)
        for c in sorted(alphabet, key=lambda s: (counts[s], s)):
            if c not in string:
                continue
            compressed = bytearray()
            seen_c = False
            for b in string:
                if b == c:
                    if not seen_c:
                        seen_c = True
                        compressed.append(b)
                else:
                    seen_c = False
                    compressed.append(b)
            compressed = bytes(compressed)
            if compressed != string:
                self.debug('Compressing runs of %r' % (bytes([c]),))
                if criterion(compressed):
                    string = compressed
            c = bytes([c])

            partition = string.split(c)
            if len(partition) <= 1:
                continue
            self.debug('Partition by %r into %d parts' % (c, len(partition)))
            shrunk = _ddmin(partition, lambda ls: criterion(c.join(ls)))
            if len(shrunk) < len(partition):
                self.debug('Removed %d parts' % (
                    len(partition) - len(shrunk),))
            t = b''.join(shrunk)
            if criterion(t):
                self.debug('Removed %r entirely' % (c,))
                string = t
            else:
                smaller = {bytes([d]) for d in alphabet if d < c[0]}
                for d in sorted(smaller):
                    t = d.join(shrunk)
                    if criterion(t):
                        self.debug('Replaced %r with %r' % (c, d))
                        string = t
                        break
                else:
                    string = c.join(shrunk)

        return string

    def calculate_partition(self, string, l, r, level):
        labels = []
        count = 0
        bad = False
        for c in string:
            if c == l:
                count += 1
            elif c == r:
                count -= 1
                if count < 0:
                    bad = True
                    break
            labels.append(count >= level)
        if bad:
            return None
        if count != 0:
            return None
        if True not in labels:
            return None
        assert len(labels) == len(string)
        prev_label = None
        current = bytearray()
        partition = []
        for c, label in zip(string, labels):
            if label != prev_label:
                if current:
                    partition.append(bytes(current))
                current.clear()
                current.append(c)
                prev_label = label
            else:
                current.append(c)
        if current:
            partition.append(bytes(current))
        assert b''.join(partition) == string
        assert b'' not in partition
        return partition

    def bracket_partition(self, string, criterion):
        level = 1
        while True:
            initial = string
            brackets = list(detect_possible_brackets(string))
            partitions = []
            for l, r in brackets:
                partition = self.calculate_partition(string, l, r, level)
                if partition is not None:
                    partitions.append((l, r, len(partition)))

            partitions.sort(key=lambda x: x[-1])
            any_partitions = False
            for l, r, _ in partitions:
                partition = self.calculate_partition(string, l, r, level)
                if partition is None:
                    continue
                any_partitions = True
                self.debug(
                    'Partitioning by bracket %r at level %d into %d pieces' % (
                        bytes([l, r]), level, len(partition)))
                string = b''.join(_ddmin(
                    partition, lambda ls: criterion(b''.join(ls))
                ))
            if not any_partitions:
                break
            if string == initial:
                level += 1

    def regions(self, label):
        try:
            return self.__regions[label]
        except KeyError:
            print("Recalculating")
            self.__regions[label] = _calculate_regions(self.best[label])
            return self.__regions[label]

    def shrink(self):
        prev = -1
        while prev != self.shrinks:
            assert self.shrinks > prev
            prev = self.shrinks
            options = list(self.best.items())
            # Always prefer the label we started with, because that's the one
            # the user is most likely to be interested in. Amongst the rest,
            # go for the one that is currently most complicated.
            options.sort(key=lambda lr: sort_key(lr[1]), reverse=True)
            options.sort(key=lambda lr: lr[0] != self.__initial_label)
            for label, current in options:
                if not current:
                    continue
                if self.principal_only and self.__initial_label != label:
                    continue
                if self.classify(b'') == label:
                    continue

                k = 1
                while k > 0:
                    initial_shrinks = self.shrinks

                    i = 0
                    while i + k <= len(self.regions(label)):
                        bitmask = [True] * len(self.best[label])
                        for r in self.regions(label)[i:i+k]:
                            for i in r:
                                bitmask[i] = False
                        assert not all(bitmask)
                        if self.classify(bytes(
                            c for c, i in zip(self.best[label], bitmask)
                            if i
                        )) != label:
                            i += k
                    if self.shrinks == initial_shrinks:
                        k //= 2
                    else:
                        k = max(k, len(self.regions(label)) // 2)


def ngrams(string):
    assert isinstance(string, bytes)
    grams_to_indices = {b'': range(len(string))}
    ngrams = set()
    ngram_counts = Counter()
    c = 0
    while grams_to_indices:
        new_grams_to_indices = {}
        for ng, ls in grams_to_indices.items():
            assert len(ng) == c
            if len(ls) >= 2:
                if ng:
                    ngrams.add(ng)
                    ngram_counts[ng] = len(ls)
                seen = set()
                for i in ls:
                    g = string[i:i + len(ng) + 1]
                    seen.add(g)
                    if len(g) == c + 1:
                        new_grams_to_indices.setdefault(g, []).append(i)
        c += 1
        grams_to_indices = new_grams_to_indices
    for ngram in sorted(ngrams, key=len, reverse=True):
        for t in [ngram[:-1], ngram[1:]]:
            if ngram_counts[t] == ngram_counts[ngram]:
                ngrams.discard(t)
    return sorted(ngrams, key=len, reverse=True)


def score(splitter, string):
    # Lower is better.
    bits = string.split(splitter)
    if not bits:
        return (0, 0)
    else:
        return (-min(map(len, bits)), len(bits))


def _smallmin(string, classify):
    assert len(string) <= 2
    # A bunch of small example optimizations. They're mostly not
    # hit but can be a huge time saver when they are.
    if len(string) <= 2:
        for a in ALPHABET:
            if classify(a):
                return a
        assert len(string) == 2
        for a in ALPHABET:
            for b in ALPHABET:
                c = a + b
                if c >= string:
                    break
                if classify(c):
                    return c


def _bytemin(string, criterion):
    return bytes(_ddmin(list(string), lambda ls: criterion(bytes(ls))))

SMALL = 2


def _ddmin(ls, criterion):
    if not criterion(ls):
        raise ValueError('Initial example does not satisfy condition')
    if criterion([]):
        return []
    k = len(ls) // 2
    while k > 0:
        i = 0
        while k < len(ls) and i + k <= len(ls):
            s = ls[i:i + k]
            assert len(s) < len(ls)
            if criterion(s):
                ls = s
            else:
                s = ls[:i] + ls[i + k:]
                assert len(s) + k == len(ls)
                if criterion(s):
                    ls = s
                else:
                    if k <= SMALL:
                        i += 1
                    else:
                        i += k
        if k <= SMALL:
            k -= 1
        elif k <= 2 * SMALL:
            k = SMALL
        else:
            k //= 2
    return ls


def shrink(*args, **kwargs):
    """Attempt to find a minimal version of initial that satisfies classify."""
    shrinker = Shrinker(*args, **kwargs)
    shrinker.shrink()
    return shrinker.best


def intervals_for_brackets(string, l, r):
    intervals = []
    stack = []
    for i, c in enumerate(string):
        if c == l:
            stack.append(i)
        elif c == r:
            if stack:
                intervals.append((stack.pop(), i + 1))
            else:
                return None
    if stack:
        return None
    return intervals


def detect_possible_brackets(string):
    counts = Counter(string)
    reverse_counts = {}
    for v, n in counts.items():
        if n > 1:
            reverse_counts.setdefault(n, []).append(v)
    return sorted([
        (a, b)
        for ls in reverse_counts.values()
        for a in ls
        for b in ls
        if string.index(a) < string.index(b)
    ], key=lambda x: counts[x[0]], reverse=True)


class denseset(object):
    def __init__(self, start, finish):
        self.start = start
        self.finish = finish
        self.__hash = None

    def __len__(self):
        return self.finish - self.start

    def __iter__(self):
        return iter(range(self.start, self.finish))

    def __eq__(self, other):
        if isinstance(other, denseset):
            return self.start == other.start and self.finish == other.finish
        return frozenset(self) == other

    def __ne__(self, other):
        r = self.__eq__(other)
        if r is NotImplemented:
            return r
        else:
            return not r


def _calculate_regions(string):
    print("Hi?")
    result = set()
    index = {}
    for i, c in enumerate(string):
        index.setdefault(c, []).append(i)
    for c, poses in index.items():
        result.add(frozenset(poses))
        for i, u in enumerate(poses[:-1]):
            result.add(frozenset(denseset(u, poses[i+1])))
    result = list(result)
    print("Calculated", len(result))

    def cmp(r, s):
        if len(r) > len(s):
            return -1
        if len(r) < len(s):
            return 1
        r = sorted(r)
        s = sorted(s)
        for u, v in zip(r, s):
            if string[u] < string[v]:
                return 1
            if string[u] > string[v]:
                return -1
        for u, v in zip(r, s):
            if u < v:
                return -1
            if u > v:
                return 1
        return 0

    result.sort(key=cmp_to_key(cmp))
    print("Sorted")
    return result
