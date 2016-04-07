import hashlib
from collections import OrderedDict
from enum import IntEnum
import random


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
        volume=Volume.quiet
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
                        self.output(
                            "Shrink %d: Label %r now %d bytes (deleted %d)" % (
                                self.shrinks, result, len(string),
                                len(self.best[result]) - len(string)))
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
                if self.classify(b'') == label:
                    continue

                initial_shrinks = self.shrinks
                self.output("Shrinking for label %r from %d bytes" % (
                    label, len(current)))

                if len(current) <= 2:
                    _smallmin(current, lambda b: self.classify(b) == label)

                lo = 0
                hi = len(current)
                while lo + 1 < hi:
                    mid = (lo + hi) // 2
                    if self.classify(current[:mid]) == label:
                        hi = mid
                    else:
                        lo = mid

                if initial_shrinks != self.shrinks:
                    continue

                for ngram in self.__suitable_ngrams(label):
                    initial = self.best[label].split(ngram)
                    if len(initial) < len(ngram) + 1:
                        continue
                    assert len(initial) > 2
                    self.debug((
                        "Splitting by %r into %d parts. "
                        "Smallest size %d") % (
                            ngram, len(initial), min(map(len, initial))))
                    result = _lsmin(
                        initial,
                        lambda ls: self.classify(ngram.join(ls)) == label
                    )
                    if len(result) < len(initial):
                        self.debug("Split removed %d parts out of %d" % (
                            len(initial) - len(result), len(initial)))

                if initial_shrinks != self.shrinks:
                    continue

                for ngram in self.__suitable_ngrams(label):
                    initial = self.best[label].split(ngram)
                    self.debug("Attempting to minimize ngram %r" % (
                        ngram,))
                    _bytemin(
                        ngram, lambda ls: self.classify(
                            ls.join(initial)
                        ) == label
                    )

                if initial_shrinks != self.shrinks:
                    continue

                self.debug("Minimizing by bytes")
                _bytemin(
                    self.best[label], lambda b: self.classify(b) == label)
                if initial_shrinks != self.shrinks:
                    continue
                width = 16
                while width > 0:
                    i = 0
                    while i + width <= len(self.best[label]):
                        c = self.best[label]
                        d = c[:i] + c[i + width:]
                        self.classify(d)
                        i += 1
                    width -= 1


def ngrams(string):
    assert isinstance(string, bytes)
    grams_to_indices = {b'': range(len(string))}
    grams = []
    scores = {}
    counts = {}
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
            if len(ls) >= max(2, len(ng)):
                grams.append(ng)
                seen = set()
                for i in ls:
                    g = string[i:i+len(ng)+1]
                    seen.add(g)
                    if len(g) == c + 1:
                        new_grams_to_indices.setdefault(g, []).append(i)
                if (
                    len(seen) == 1 and
                    len(new_grams_to_indices[list(seen)[0]]) >= len(ng) + 1
                ):
                    # If the ngram always extends to the same thing, remove it
                    assert grams[-1] == ng
                    grams.pop()
        c += 1
        grams_to_indices = new_grams_to_indices
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


def _bytemin(string, criterion):
    return bytes(_lsmin(list(string), lambda ls: criterion(bytes(ls))))

EXPMIN_THRESHOLD = 6
QUADMIN_THRESHOLD = 8


def _lsmin(ls, criterion):
    if criterion([]):
        return []
    prev = None
    while len(ls) > 8 and ls != prev:
        prev = ls
        ls = _randmin(ls, criterion)
        ls = _ddmin(ls, criterion)
    if EXPMIN_THRESHOLD < len(ls) <= QUADMIN_THRESHOLD:
        ls = _quadmin(ls, criterion)
    if len(ls) <= EXPMIN_THRESHOLD:
        ls = _expmin(ls, criterion)
    return ls


def _randmin(ls, criterion):
    prev = None
    while len(ls) > 6 and ls != prev:
        prev = ls

        i = 0
        while i < len(ls) and len(ls) > 5:
            while True:
                j = random.randint(0, len(ls))
                if abs(j - i) > 2:
                    break
            u, v = sorted((i, j))
            ts = ls[:u] + ls[v:]
            if criterion(ts):
                ls = ts
            else:
                i += 1
    return ls


def subsets(ls):
    assert len(ls) <= 10
    results = []
    for subset in range(2 ** len(ls)):
        results.append([
            t for i, t in enumerate(ls) if subset & (1 << i)
        ])
    results.sort(key=len)
    return results


def _expmin(ls, criterion):
    if not ls:
        return ls
    for i, s in enumerate(subsets(ls)):
        if criterion(s):
            return s
    assert False


def _ddmin(ls, criterion):
    if not criterion(ls):
        raise ValueError("Initial example does not satisfy condition")
    prev = None
    while ls != prev:
        prev = ls
        k = len(ls) // 2
        while k > 0:
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
                        i += max(1, k - 1)
            k //= 2
    return ls


def _quadmin(ls, criterion):
    prev = None
    while ls != prev:
        prev = ls
        width = 32
        while width > 0:
            i = 0
            while i + width <= len(ls):
                ts = ls[:i] + ls[i + width:]
                assert len(ts) < len(ls)
                if criterion(ts):
                    ls = ts
                else:
                    i += 1
            width -= 1

        i = 0
        while i < len(ls):
            j = 0
            while j < i:
                ts = ls[:j] + ls[i:]
                assert len(ts) < len(ls)
                if criterion(ts):
                    ls = ts
                    i = j
                    break
                j += 1
            else:
                i += 1
    return ls


def shrink(*args, **kwargs):
    """Attempt to find a minimal version of initial that satisfies classify"""
    shrinker = Shrinker(*args, **kwargs)
    shrinker.shrink()
    return shrinker.best
