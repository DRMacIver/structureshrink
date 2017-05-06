import hashlib
from collections import OrderedDict, Counter
from enum import IntEnum
from random import Random
import sys
from contextlib import contextmanager


class Volume(IntEnum):
    quiet = 0
    normal = 1
    debug = 2


def cache_key(s):
    if len(s) < 20:
        return s
    return hashlib.sha1(s).digest()


ALPHABET = [bytes([b]) for b in range(256)]

NEWLINE = b'\n'


SHORT = 8


def phase(phase):
    def accept(self, *args, **kwargs):
        with self.in_phase(phase.__name__.replace("_", " ")):
            return phase(self, *args, **kwargs)
    return accept


class Shrinker(object):

    def __init__(
        self,
        initial, classify, *,
        preprocess=None, shrink_callback=None, printer=None,
        volume=Volume.quiet, principal_only=False,
        passes=None, seed=None, preserve_lines=False
    ):
        self.phase_name = None
        self.__indent = 0
        self.__dots = 0
        self.__preserve_lines = preserve_lines
        self.random = Random(seed)
        self.__interesting_ngrams = set()
        self.__shrink_callback = shrink_callback or (lambda s, r: None)
        self.__printer = printer or (lambda s: None)
        self.__inital = initial
        self.__classify = classify
        self.__preprocess = preprocess or (lambda s: s)
        self.__volume = volume
        self.__explain_lines = []
        self.__byte_order = {}
        for c, n in Counter(initial).items():
            self.__byte_order[c] = (-n, c)
        self.__cache = {}
        self.__preprocess_cache = {}
        self.__best = OrderedDict()
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
            return True
        return False

    def output(self, text):
        if self.__volume >= Volume.normal:
            self.__echo(text)

    def explain(self, text):
        self.__clear_explain()
        self.__explain_lines.append(text)

    def debug(self, text):
        if self.__volume >= Volume.debug:
            self.__echo(text)

    @contextmanager
    def context(self, name):
        if self.__volume >= Volume.debug:
            self.debug("Starting: " + name)
            self.__indent += 1
            try:
                yield
            finally:
                self.__indent -= 1
                self.debug("Finishing: " + name)
                if self.__indent == 0:
                    self.debug("")
        else:
            yield

    def __echo(self, text):
        if self.__dots > 0:
            self.__clear_dots()
        self.__printer("  " * self.__indent + text)

    def __clear_dots(self):
        sys.stdout.write('\r')
        sys.stdout.write(' ' * (
            self.__dots + len(self.phase_name or '') + 3))
        sys.stdout.write('\r')
        sys.stdout.flush()
        self.__dots = 0

    @property
    def __byte_sort_key(self):
        return self.__byte_order.__getitem__

    @property
    def best(self):
        return self.__best

    def sort_key(self, s):
        return (len(s), [self.__byte_sort_key(b) for b in s])

    def classify(self, string):
        key = cache_key(string)
        try:
            return self.__cache[key]
        except KeyError:
            pass

        if self.__volume >= Volume.debug:
            if self.__dots == 0 and self.phase_name is not None:
                sys.stdout.write("(%s) " % (self.phase_name,))

            sys.stdout.write(".")
            sys.stdout.flush()
            self.__dots += 1

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
                self.sort_key(string) < self.sort_key(self.best[result])
            ):
                if self.best and (
                    not self.principal_only or result == self.__initial_label
                ):
                    self.shrinks += 1
                    if result not in self.best:
                        self.__process_explain()
                        self.output((
                            'Shrink %d (%s): Discovered new label %r'
                            ' with %d bytes') % (
                                self.shrinks, self.phase_name,
                                result, len(string)))
                    else:
                        deletes = len(self.best[result]) - len(string)
                        if deletes == 0:
                            shrink_message = 'lowered %d' % (
                                len([1 for u, v in zip(
                                    string, self.best[result]) if u != v]),)
                        else:
                            shrink_message = 'deleted %d' % (deletes,)

                        self.__process_explain()
                        self.output(
                            'Shrink %d (%s): Label %r now %d bytes (%s)' % (
                                self.shrinks, self.phase_name,
                                 result, len(string),
                                shrink_message))
                self.__shrink_callback(string, result)
                self.__best[result] = string
        for k in keys:
            self.__cache[k] = result
        return result

    def __process_explain(self):
        for s in self.__explain_lines:
            self.debug(s)
        self.__clear_explain()

    def __clear_explain(self):
        self.__explain_lines.clear()

    def rand_shrink(self, string, criterion):
        eager = True
        k_bound = len(string)
        while k_bound > 1:
            k = 1
            while k < k_bound:
                self.debug("k=%d" % (k,))
                indices = list(range(0, len(string) - k))
                self.random.shuffle(indices)
                for i in indices:
                    shrunk = string[:i] + string[i + k:]
                    assert len(shrunk) + k == len(string)
                    if criterion(shrunk):
                        string = shrunk
                        if eager:
                            k *= 2
                        else:
                            k += 1
                        break
                else:
                    k_bound = k
                    k = 1
                    eager = False
        return string

    def expmin_string(self, string, criterion):
        if criterion(b''):
            return b''

        subsets = set()
        for s in range(1, 2 ** len(string) - 1):
            bits = []
            for x in string:
                if s & 1:
                    bits.append(x)
                s >>= 1
                if not s:
                    break
            subsets.add(bytes(bits))
        subsets = list(subsets)
        subsets.sort(key=shortlex)
        for bits in subsets:
            if criterion(bits):
                return bits
        return string

    def ngram_brute_force(self, string, criterion):
        for k in range(6, 1, -1):
            counts = Counter(string[i:i+k] for i in range(len(string) - k))
            ngrams = [c for c, k in counts.items() if k > 1]
            ngrams.sort(key=lambda n: (counts[n], n), reverse=True)
            for n in ngrams:
                self.debug("Brute forcing %r" % (n,))
                parts = string.split(n)
                if len(parts) <= 2:
                    continue
                n = self.expmin_string(n, lambda m: criterion(m.join(parts)))
                string = n.join(parts)
        return string

    def remove_first_and_last(self, string, criterion):
        self.debug("Pruning around particular bytes")
        prev = None
        while prev != string:
            prev = string
            index = {}
            for i, c in enumerate(string):
                index.setdefault(c, [i, i])[1] = i
            starts = sorted([v[0] for v in index.values()], reverse=True)
            for i in starts:
                attempt = string[i + 1:]
                if criterion(attempt):
                    string = attempt
                    break
            else:
                ends = sorted([v[1] for v in index.values()])
                for i in ends:
                    attempt = string[:i]
                    if criterion(attempt):
                        string = attempt
                        break
        return string

    def __index_ngrams(self, string, i):
        n = len(string)
        c = string[i]
        indices = [[j for j in range(i + 1, n) if string[j] == c]]
        if indices[0]:
            while True:
                k = len(indices)
                new_indices = [
                    j for j in indices[-1]
                    if j + k < n
                    and string[j + k] == string[i + k]
                    and j > i + k
                ]
                if not new_indices:
                    break
                indices.append(new_indices)
        else:
            indices.pop()
        return indices

    def select_ngram(self, string):
        n = len(string)
        counts = Counter(string)
        if not any(v > 1 for v in counts.values()):
            return None
        while True:
            i = self.random.randint(0, len(string) - 1)
            c = string[i]
            if counts[c] == 1:
                continue
            indices = [j for j, d in enumerate(string) if c == d]
            k = 1
            while i + k < n:
                new_indices = [
                    j for j in indices
                    if abs(j - i) > k
                    and j + k < n
                    and string[j + k] == string[i + k]
                ]
                if new_indices:
                    indices = new_indices
                    k += 1
                else:
                    break
            return string[i:i+k]

    def adaptive_greedy_search_pass(self, test_case, predicate):
        indices = list(range(len(test_case)))
        self.random.shuffle(indices)

        def from_indices(ix):
            return [test_case[i] for i in sorted(ix)]

        def index_predicate(ix):
            return predicate(from_indices(ix))

        i = 0
        while i < len(indices):
            def check(k):
                return index_predicate(indices[:i] + indices[i + k:])
            if check(1):
                lo = 1
                hi = 2
                while i + hi <= len(indices) and check(hi):
                    hi *= 2
                if check(hi):
                    indices = indices[:i]
                    break
                while lo + 1 < hi:
                    mid = (lo + hi) // 2
                    if check(mid):
                        lo = mid
                    else:
                        hi = mid
                indices = indices[:i] + indices[i + lo:]
                assert index_predicate(indices)
            i += 1
        return from_indices(indices)

    @phase
    def rand_structure_shrink_string(self, string, criterion):
        done = set()
        failures = 0
        while failures < 100:
            ngram = self.select_ngram(string)
            if ngram is None:
                break
            if ngram in done or len(ngram) <= 1:
                failures += 1
                continue
            done.add(ngram)
            parts = string.split(ngram)
            self.debug("Shrinking by splitting on %r into %d parts" % (
                ngram, len(parts)))
            assert len(parts) > 2
            original = string

            parts = self.adaptive_greedy_search_pass(
                parts, lambda ls: criterion(ngram.join(ls)))

            string = ngram.join(parts)

            if string == original:
                failures += 1
            else:
                failures = 0
        return string

    @contextmanager
    def in_phase(self, phase_name):
        original = self.phase_name
        try:
            self.phase_name = phase_name
            yield
        finally:
            self.phase_name = original
            self.__clear_dots()

    @phase
    def partition_shrink(self, string, criterion):
        partition = []
        i = 0
        n = len(string)
        while i < n:
            indices = self.__index_ngrams(string, i)
            k = max(1, len(indices))
            partition.append(string[i:i+k])
            i += k
        assert b''.join(partition) == string
        return b''.join(self.adaptive_greedy_search_pass(
            partition, lambda ls: criterion(b''.join(ls))
        ))

    @phase
    def structure_shrink_string(self, string, criterion):
        self.debug("Structured shrinking")
        iter_order = list(range(len(string)))
        self.random.shuffle(iter_order)
        for i in iter_order:
            n = len(string)
            if i >= n:
                continue
            indices = self.__index_ngrams(string, i)
            if indices:
                targets = sorted({ix[0] for ix in indices}, reverse=True)
                self.explain(
                    "Trying %d indices for prefixes of %r" % (
                        len(targets), string[i:i+len(indices)],
                    ))

                for j in targets:
                    attempt = string[:i] + string[j:]
                    if criterion(attempt):
                        string = attempt
                        break
        return string

    @phase
    def byte_shrink_string(self, string, criterion):
        self.debug("Shrinking bytewise")
        return bytes(self.adaptive_greedy_search_pass(
            list(string), lambda ls: criterion(bytes(ls))))

    @phase
    def demarcated_shrink(self, string, criterion):
        self.debug("Shrinking demarcated intervals")

        used = set()

        while True:
            counts = Counter(string)
            alphabet = [
                c for c, k in counts.items() if c not in used and k > 1]
            if not alphabet:
                break
            c = min(alphabet, key=lambda b: (counts[b], b))
            used.add(c)
            c = bytes([c])
            parts = string.split(c)
            self.debug("Removing %d intervals demarcated by %r" % (
                len(parts), c,))
            parts = self.adaptive_greedy_search_pass(
                parts, lambda ls: criterion(c.join(ls)))
            string = c.join(parts)
        return string

    def expensive_shrink_string(self, string, criterion):
        if len(string) <= SHORT:
            return string
        for k in range(SHORT, 1, -1):
            self.explain("Deleting intervals of length %d" % (k,))
            assert k > 1
            i = 0
            while i + k <= len(string):
                attempt = string[:i] + string[i+k:]
                if criterion(attempt):
                    string = attempt
                else:
                    i += 1
        self.explain("Deleting pairs")
        i = 0
        while i < len(string):
            for j in range(i + 2, min(i + SHORT, len(string))):
                attempt = string[:i] + string[i + 1:j] + string[j+1:]
                if criterion(attempt):
                    string = attempt
                    break
            else:
                i += 1
        return string

    def shrink_string(self, string, criterion):
        assert criterion(string)
        if len(string) <= 1:
            return string

        if len(set(string)) == 1:
            for i in range(len(string)):
                attempt = string[:i]
                if criterion(attempt):
                    return attempt
            return string

        if len(string) <= 10:
            string = self.byte_shrink_string(string, criterion)
        if len(string) <= 6:
            return self.expmin_string(string, criterion)
        if criterion(b''):
            return b''
        prev = None
        while prev != string:
            prev = string
            string = self.alphabet_minimize(string, criterion)
            string = self.structure_shrink_string(string, criterion)
            string = self.demarcated_shrink(string, criterion)
            string = self.lexicographically_minimize_string(string, criterion)
            if prev != string:
                continue
            string = self.byte_shrink_string(string, criterion)
            string = self.expensive_shrink_string(string, criterion)
        return string

    @phase
    def alphabet_minimize(self, string, criterion):
        alphabet = sorted(set(string), key=self.__byte_sort_key)

        def replace(a, b):
            assert a in string
            return bytes([b if s == a else s for s in string])

        for i, c in enumerate(alphabet):
            self.explain("Removing %r" % (bytes([c]),))
            attempt = bytes([s for s in string if s != c])
            if criterion(attempt):
                string = attempt

        alphabet = sorted(set(string), key=self.__byte_sort_key)

        for i, c in enumerate(alphabet):
            if i > 0 and c in string:
                attempt = replace(c, alphabet[0])
                self.explain("Replacing %r" % (bytes([c]),))
                if criterion(attempt):
                    string = attempt
                    continue
                if criterion(replace(c, alphabet[i - 1])):
                    hi = i - 1
                    lo = 0
                    while lo + 1 < hi:
                        mid = (lo + hi) // 2
                        if criterion(replace(c, alphabet[mid])):
                            hi = mid
                        else:
                            lo = mid
                    d = alphabet[hi]
                    string = replace(c, d)
                    assert c not in string
                    assert criterion(string)
                    self.debug(
                        "Replaced %r with %r" % (bytes([c]), bytes([d])))
        return string

    @phase
    def lexicographically_minimize_string(self, string, criterion):
        n = len(string)

        def lower(u, v):
            c = bytes([min(string[u:v], key=self.__byte_sort_key)])
            r = string[:u] + c * (v - u) + string[v:]
            assert len(r) == len(string)
            return r
        i = 0
        while i < n:
            if criterion(lower(i, i + 1)):
                lo = 1
                hi = 2
                while i + hi <= n and criterion(lower(i, i + hi)):
                    hi *= 2
                if i + hi <= n:
                    while lo + 1 < hi:
                        mid = (lo + hi) // 2
                        if criterion(lower(i, i + mid)):
                            lo = mid
                        else:
                            hi = mid
                string = lower(i, i + lo)
            i += 1
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
            options.sort(key=lambda lr: self.sort_key(lr[1]), reverse=True)
            options.sort(key=lambda lr: lr[0] != self.__initial_label)
            for label, current in options:
                if not current:
                    continue
                if self.principal_only and self.__initial_label != label:
                    continue
                if self.classify(b'') == label:
                    continue
                self.output(
                    'Shrinking for label %r from %d bytes (%d distinct)' % (
                        label, len(current), len(set(current))))

                def criterion(s):
                    assert self.sort_key(s) <= self.sort_key(self.best[label])
                    return self.classify(s) == label

                self.shrink_string(self.best[label], criterion)


def shrink(*args, **kwargs):
    """Attempt to find a minimal version of initial that satisfies classify."""
    shrinker = Shrinker(*args, **kwargs)
    shrinker.shrink()
    return shrinker.best
        

def shortlex(b):
    return (len(b), b)

