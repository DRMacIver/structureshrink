import hashlib
from collections import OrderedDict, Counter
from enum import IntEnum


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
        volume=Volume.quiet, principal_only=False
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
        self.shrinks = 0
        preprocessed = self.__preprocess(initial)
        if preprocessed is None:
            raise ValueError("Initial example is rejected by preprocessing")
        label = self.classify(preprocessed)
        self.output("Initial example: %s, labelled %r" % ((
            "%d bytes " % (len(initial),)
            if initial == preprocessed
            else "%d bytes (%d preprocessed)" % (
                len(initial), len(preprocessed))),
            label))
        self.__initial_label = label
        self.principal_only = principal_only

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
                self.shrinks += 1
                if self.best:
                    if result not in self.best:
                        self.output((
                            "Shrink %d: Discovered new label %r"
                            " with %d bytes") % (
                                self.shrinks, result, len(string)))
                    else:
                        deletes = len(self.best[result]) - len(string)
                        if deletes == 0:
                            shrink_message = "lowered %d" % (
                                len([1 for u, v in zip(
                                    string, self.best[result]) if u < v]),)
                        else:
                            shrink_message = "deleted %d" % (deletes,)

                        self.output(
                            "Shrink %d: Label %r now %d bytes (%s)" % (
                                self.shrinks, result, len(string),
                                shrink_message))
                self.__shrink_callback(string, result)
                self.__best[result] = string
        for k in keys:
            self.__cache[k] = result
        return result

    def __suitable_ngrams(self, label):
        self.debug("Calculating ngrams for %r" % (label,))
        found_ngrams = ngrams(self.best[label])
        self.debug("Found %d ngrams" % len(found_ngrams),)
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
                self.debug("Shrinking for bracketed pair %r, %r" % (
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
                            string[:u] + string[u+1:v-1]  + string[v:],
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


    def compress_runs(self, string, criterion):
        for c in range(256):
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
                self.debug("Compressing runs of %r" % (bytes([c]),))
                if criterion(compressed):
                    string = compressed
        return string


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
                self.output("Shrinking for label %r from %d bytes" % (
                    label, len(current)))

                if len(current) <= 2:
                    _smallmin(current, lambda b: self.classify(b) == label)
                    continue

                initial_shrinks = self.shrinks

                def criterion(string):
                    return self.classify(string) == label

                self.debug("Compressing runs")
                self.compress_runs(self.best[label], criterion)

                # We do an initial bracket shrink pass with a threshold close
                # but not exactly 1. This catches a lot of potential for coarse
                # deletion of blocks but the fact that we enforce an
                # exponential shrink prevents us from getting distracted by a
                # bunch of tiny shrinks here.
                self.bracket_shrink(
                    self.best[label], criterion, threshold=0.99)

                brackets = list(detect_possible_brackets(self.best[label]))
                brackets.sort(key=lambda s: self.best[label].count(s[0]))
                for l, r in brackets:
                    if intervals_for_brackets(self.best[label], l, r) is None:
                        continue
                    partition = bracket_partition(self.best[label], l, r)
                    self.debug("Partitioning by bracket %r into %d pieces" % (
                        bytes([l, r]), len(partition)
                    ))
                    _ddmin(
                        partition, lambda ls: criterion(b''.join(ls))
                    )

                if initial_shrinks != self.shrinks:
                    continue

                lo = 0
                hi = len(current)
                while lo + 1 < hi:
                    mid = (lo + hi) // 2
                    if self.classify(current[:mid]) == label:
                        hi = mid
                    else:
                        lo = mid

                for ngram in self.__suitable_ngrams(label):
                    initial = self.best[label].split(ngram)
                    if len(initial) < len(ngram) + 1:
                        continue
                    assert len(initial) >= 2
                    self.debug((
                        "Splitting by %r into %d parts. "
                        "Smallest size %d") % (
                            ngram, len(initial), min(map(len, initial))))
                    result = _ddmin(
                        initial,
                        lambda ls: self.classify(ngram.join(ls)) == label
                    )
                    if len(result) < len(initial):
                        self.debug("Split removed %d parts out of %d" % (
                            len(initial) - len(result), len(initial)))

                    initial = result
                    self.debug("Attempting to minimize ngram %r" % (
                        ngram,))
                    result = _bytemin(
                        ngram, lambda ls: self.classify(
                            ls.join(initial)
                        ) == label
                    )
                    if ngram != result:
                        self.debug("Minimized ngram %r to %r" % (
                            ngram, result))

                self.debug("Minimizing bracketwise")
                self.bracket_shrink(
                    self.best[label], lambda c: self.classify(c) == label
                )

                if initial_shrinks != self.shrinks:
                    continue

                self.debug("Minimizing by bytes")
                _bytemin(
                    self.best[label], lambda b: self.classify(b) == label)

                current = self.best[label]
                characters = sorted(set(current))
                self.debug("Minimizing alphabet of %d characters" % (
                    len(characters),
                ))
                for a in characters:
                    if self.best[label].count(a) <= 1:
                        continue
                    for b in characters:
                        current = self.best[label]
                        if a < b:
                            test = self.shrinks
                            self.classify(bytes(
                                a if u == b else u for u in current 
                            ))
                            if self.shrinks != test:
                                self.debug("%r -> %r" % (
                                    bytes([b]), bytes([a])))
                new_size = len(set(self.best[label]))
                if new_size < len(characters):
                    self.debug("Minimized alphabet to %d characters" % (
                        new_size,
                    ))



def ngrams(string):
    assert isinstance(string, bytes)
    grams_to_indices = {b'': range(len(string))}
    scores = {}
    counts = {}
    grams_by_count = {}
    c = 0
    while grams_to_indices:
        for gram, indices in grams_to_indices.items():
            counts[gram] = len(indices)
            if len(indices) > 1:
                # A decent approximation to the size of string when splitting
                # by this ngram. It ignores overlap so isn't quite correct, but
                # is easier and cheaper to calculate.
                scores[gram] = min(b - a for a, b in zip(indices, indices[1:]))
                scores[gram] = min(scores[gram], indices[0])
                scores[gram] = min(scores[gram], len(string) - indices[-1])
                assert scores[gram] >= 0
            scores[gram] = 0
        new_grams_to_indices = {}
        for ng, ls in grams_to_indices.items():
            assert len(ng) == c
            if len(ls) + 1 >= max(2, len(ng)):
                if ng:
                    grams_by_count.setdefault(len(ls), []).append(ng)
                seen = set()
                for i in ls:
                    g = string[i:i+len(ng)+1]
                    seen.add(g)
                    if len(g) == c + 1:
                        new_grams_to_indices.setdefault(g, []).append(i)
        c += 1
        grams_to_indices = new_grams_to_indices
    merge_table = {}
    def find(u):
        trail = []
        while True:
            n = merge_table.setdefault(u, u)
            if n != u:
                assert u not in trail
                trail.append(u)
                u = n
            else:
                break
        for t in trail:
            merge_table[t] = u
        return u

    def union(u, v):
        u = find(u)
        v = find(v)
        if u == v:
            return
        u, v = sorted((u, v), key=lambda x: (len(x), x))
        merge_table[u] = v

    for vs in grams_by_count.values():
        for v in vs:
            find(v)
        vs.sort(key=len)
        for i, u in enumerate(vs):
            if len(u) <= 3:
                continue
            for v in vs[i+1:]:
                if len(v) > len(u) + 1:
                    break
                if u[:-1] == v[1:] or v[:-1] == u[:1] or u in v or v in u:
                    union(u, v)
    grams = list({find(v) for v in merge_table.values()})
    grams.sort(key=lambda s: (len(s), scores[s], -counts[s]), reverse=True)
    return grams


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


def bracket_partition(string, l, r):
    labels = []
    count = 0
    for c in string:
        if c == l:
            count += 1
        elif c == r:
            count -= 1
            assert count >= 0
        labels.append(count > 0)
    assert len(labels) == len(string)
    prev_label = None
    current = bytearray()
    result = []
    for c, label in zip(string, labels):
        if label != prev_label:
            if current:
                result.append(bytes(current))
            current.clear()
            current.append(c)
            prev_label = label
        else:
            current.append(c)
    if current:
        result.append(bytes(current))
    assert b''.join(result) == string
    assert b'' not in result
    return result


def _bytemin(string, criterion):
    return bytes(_ddmin(list(string), lambda ls: criterion(bytes(ls))))

def _ddmin(ls, criterion):
    if not criterion(ls):
        raise ValueError("Initial example does not satisfy condition")
    if criterion([]):
        return []
    prev = None
    heat = 1
    while len(ls) > 1:
        if prev == ls:
            heat *= 2
            if heat >= min(16, len(ls)):
                break 
        prev = ls
        k = len(ls) // 2
        while k > 0:
            if k > heat:
                step = k
            else:
                step = 1
            prev2 = None
            while prev2 != ls:
                prev2 = ls
                i = 0
                while i + k <= len(ls):
                    s = ls[:i] + ls[i + k:]
                    assert len(s) + k == len(ls)
                    if criterion(s):
                        ls = s
                        if i > 0:
                            i -= 1
                    else:
                        i += step
            if k >= heat * 2:
                k //= 2
            elif k > heat:
                k = heat
            else:
                k -= 1
    return ls


def shrink(*args, **kwargs):
    """Attempt to find a minimal version of initial that satisfies classify"""
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
