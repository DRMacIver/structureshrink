import hashlib


def sort_key(s):
    return (len(s), s)


def cache_key(s):
    if len(s) < 20:
        return s
    return hashlib.sha1(s).digest()


ALPHABET = [bytes([b]) for b in range(256)]


class Shrinker(object):
    def __init__(
        self, initial, criterion,
        preprocess=None,
        shrink_callback=None, debug=None
    ):
        self.__shrink_callback = shrink_callback or (lambda s: None)
        self.debug = debug or (lambda s: None)
        self.__inital = initial
        self.__criterion = criterion
        self.__preprocess = preprocess or (lambda s: s)

        self.__cache = {}
        self.__preprocess_cache = {}
        self.__best = initial
        self.__ngram_scores = {}
        self.__useful_ngrams = set()
        self.shrinks = 0
        preprocessed = preprocess(initial)
        if preprocessed is None:
            raise ValueError("Initial example is rejected by preprocessing")
        if not self.criterion(preprocessed):
            raise ValueError("Initial example does not satisfy criterion")
        self.__best = preprocessed
        if len(self.__best) != len(initial):
            self.debug("Initial size: %d bytes (%d preprocessed)" % (
                len(initial), len(self.best)))
        else:
            self.debug("Initial size: %d bytes" % (len(initial),))

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
            result = False
        else:
            preprocess_key = cache_key(preprocessed)
            keys.append(preprocess_key)
            try:
                result = self.__cache[preprocess_key]
            except KeyError:
                result = self.__criterion(preprocessed)
            if result and sort_key(string) < sort_key(self.best):
                self.shrinks += 1
                self.debug(
                    "Shrink %d: %d bytes (deleted %d)" % (
                        self.shrinks, len(string),
                        len(self.best) - len(string)))
                self.__shrink_callback(string)
                self.__best = string
        for k in keys:
            self.__cache[k] = result
        return result

    def __suitable_ngrams(self):
        ngrams_by_size = {}
        for gs in (self.__useful_ngrams, ngrams(self.best)):
            for g in gs:
                ngrams_by_size.setdefault(len(g), []).append(g)
        for _, grams in reversed(sorted(ngrams_by_size.items())):
            grams = list(filter(None, grams))
            grams.sort(key=lambda s: (score(s, self.best), s))
            for ngram in grams:
                if len(self.best.split(ngram)) > 2:
                    yield ngram

    def shrink(self):
        prev = None
        while prev != self.best:
            prev = self.best
            for ngram in self.__suitable_ngrams():
                initial = self.best.split(ngram)
                if len(initial) <= 2:
                    self.__demote_ngram(ngram)
                    continue
                self.debug(
                    "Splitting by %r into %d parts. Smallest size %d" % (
                        ngram, len(initial), min(map(len, initial))))
                final = _lsmin(
                    initial, lambda ls: self.criterion(ngram.join(ls))
                )
                if final != initial:
                    self.debug("Deleted %d parts" % (
                        len(initial) - len(final),))
                    self.__useful_ngrams.add(ngram)
                else:
                    self.__useful_ngrams.discard(ngram)

            if prev != self.best:
                continue

            for ngram in self.__suitable_ngrams():
                initial = self.best.split(ngram)
                self.debug("Attempting to to minimize %r" % (ngram,))
                minigram = bytemin(
                    ngram, lambda ls: self.criterion(
                        ls.join(initial)
                    )
                )
                if minigram != ngram:
                    self.__useful_ngrams.add(minigram)
                    self.debug(
                        "Shrunk ngram %r -> %r" % (ngram, minigram))

            if prev != self.best:
                continue
            self.debug("Minimizing by bytes")
            bytemin(self.best, self.criterion)
            if prev != self.best:
                continue
            self.debug("Quadratic minimize by bytes")
            _quadmin(list(self.best), lambda ls: self.criterion(bytes(ls)))


def ngrams(string):
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
                    assert grams.pop()
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


def bytemin(string, criterion):
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
                if criterion(ts):
                    ts = ls
                else:
                    i += 1
            width -= 1

        i = 0
        while i < len(ls):
            j = 0
            while j < i:
                ts = ls[:j] + ls[i:]
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
    preprocess=None, shrink_callback=None, debug=None
):
    """Attempt to find a minimal version of initial that satisfies criterion"""
    shrinker = Shrinker(
        initial, criterion, shrink_callback=shrink_callback,
        debug=debug, preprocess=preprocess
    )
    shrinker.shrink()
    return shrinker.best
