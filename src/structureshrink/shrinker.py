import hashlib
from collections import OrderedDict
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
        self, initial, criterion, preprocess, shrink_callback, printer, volume,
    ):
        self.__shrink_callback = shrink_callback or (lambda s: None)
        self.__printer = printer or (lambda s: None)
        self.__inital = initial
        self.__criterion = criterion
        self.__preprocess = preprocess or (lambda s: s)
        self.__volume = volume

        self.__cache = {}
        self.__preprocess_cache = {}
        self.__best = OrderedDict()
        self.__ngram_scores = {}
        self.__useful_ngrams = set()
        self.shrinks = 0
        preprocessed = self.__preprocess(initial)
        if preprocessed is None:
            raise ValueError("Initial example is rejected by preprocessing")
        label = self.criterion(preprocessed)
        self.output("Initial example: %s, labelled %r" % ((
            "%d initial" % (len(initial),)
            if initial == preprocessed
            else "%d initial (%d preprocessed)" % (
                len(initial), len(preprocessed))),
            label))

    def output(self, text):
        if self.__volume >= Volume.normal:
            self.__printer(text)

    def debug(self, text):
        if self.__volume >= Volume.debug:
            self.__printer(text)

    @property
    def best(self):
        return self.__best

    def criterion(self, string):
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
                result = self.__criterion(preprocessed)
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
        ngrams_by_size = {}
        self.debug("Calculating ngrams for %r" % (label,))

        for gs in [self.__useful_ngrams, ngrams(self.best[label])]:
            for g in gs:
                ngrams_by_size.setdefault(len(g), set()).add(g)
        for k in list(ngrams_by_size):
            ngrams_by_size[k] = list(ngrams_by_size[k])

        for _, grams in reversed(sorted(ngrams_by_size.items())):
            grams = list(filter(None, grams))
            grams.sort(key=lambda s: (score(s, self.best[label]), s))
            for ngram in grams:
                if len(self.best[label].split(ngram)) > 2:
                    yield ngram
        self.debug("Found %d ngrams" % (
            sum(len(v) for v in ngrams_by_size.values()),))

    def shrink(self):
        prev = -1
        while prev != self.shrinks:
            assert self.shrinks > prev
            prev = self.shrinks
            options = list(self.best.items())
            options.sort(
                key=lambda lr: sort_key(lr[1]), reverse=True
            )
            for label, current in options:
                if not current:
                    continue
                if self.criterion(b'') == label:
                    continue

                initial_shrinks = self.shrinks
                self.output("Shrinking for label %r from %d bytes" % (
                    label, len(current)))

                if len(current) <= 2:
                    _smallmin(current, lambda b: self.criterion(b) == label)

                lo = 0
                hi = len(current)
                while lo + 1 < hi:
                    mid = (lo + hi) // 2
                    if self.criterion(current[:mid]) == label:
                        hi = mid
                    else:
                        lo = mid

                if initial_shrinks != self.shrinks:
                    continue

                for ngram in self.__suitable_ngrams(label):
                    initial = self.best[label].split(ngram)
                    assert len(initial) > 2
                    self.debug((
                        "Splitting by %r into %d parts. "
                        "Smallest size %d") % (
                            ngram, len(initial), min(map(len, initial))))
                    final = _lsmin(
                        initial,
                        lambda ls: self.criterion(ngram.join(ls)) == label
                    )
                    if final != initial:
                        self.debug("Deleted %d parts" % (
                            len(initial) - len(final),))
                        self.__useful_ngrams.add(ngram)
                    else:
                        self.__useful_ngrams.discard(ngram)

                if initial_shrinks != self.shrinks:
                    continue

                for ngram in self.__suitable_ngrams(label):
                    initial = self.best[label].split(ngram)
                    self.debug("Attempting to minimize ngram %r" % (
                        ngram,))
                    minigram = _bytemin(
                        ngram, lambda ls: self.criterion(
                            ls.join(initial)
                        ) == label
                    )
                    if minigram != ngram:
                        self.__useful_ngrams.add(minigram)
                        self.debug(
                            "Shrunk ngram %r -> %r" % (ngram, minigram))

                if initial_shrinks != self.shrinks:
                    continue

                self.debug("Minimizing by bytes")
                _bytemin(
                    self.best[label], lambda b: self.criterion(b) == label)
                if initial_shrinks != self.shrinks:
                    continue
                width = 16
                while width > 0:
                    i = 0
                    while i + width <= len(self.best[label]):
                        c = self.best[label]
                        d = c[:i] + c[i + width:]
                        if self.criterion(d) != label:
                            i += 1
                        else:
                            assert d == self.best[label]
                    width -= 1


def ngrams(string):
    assert isinstance(string, bytes)
    grams_to_indices = {b'': range(len(string))}
    grams = []
    c = 0
    while grams_to_indices:
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
    grams.reverse()
    return grams


def score(splitter, string):
    # Lower is better.
    bits = string.split(splitter)
    if not bits:
        return (0, 0)
    else:
        return (-min(map(len, bits)), len(bits))


def _smallmin(string, criterion):
    assert len(string) <= 2
    # A bunch of small example optimizations. They're mostly not
    # hit but can be a huge time saver when they are.
    if len(string) <= 2:
        for a in ALPHABET:
            if criterion(a):
                return a
        assert len(string) == 2
        for a in ALPHABET:
            for b in ALPHABET:
                c = a + b
                if c >= string:
                    break
                if criterion(c):
                    return c


def _bytemin(string, criterion):
    return bytes(_lsmin(list(string), lambda ls: criterion(bytes(ls))))


def _lsmin(ls, criterion):
    if criterion([]):
        return []
    if len(ls) < 8:
        return _quadmin(ls, criterion)
    else:
        result = _ddmin(ls, criterion)
        if len(result) < 8:
            return _quadmin(result, criterion)
        return result


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
                        i += k
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


def shrink(
    initial, criterion, *,
    preprocess=None, shrink_callback=None, printer=None,
    volume=Volume.quiet
):
    """Attempt to find a minimal version of initial that satisfies criterion"""
    shrinker = Shrinker(
        initial, criterion, shrink_callback=shrink_callback,
        printer=printer, preprocess=preprocess, volume=volume,
    )
    shrinker.shrink()
    return shrinker.best