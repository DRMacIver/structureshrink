import hashlib
from collections import OrderedDict, Counter, defaultdict
from enum import IntEnum
from random import Random
import sys
from contextlib import contextmanager
import time
from functools import cmp_to_key


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


def phase(name):
    def wrap(function):
        def accept(self, *args, **kwargs):
            with self.in_phase(name):
                return function(self, *args, **kwargs)
        return accept
    return wrap


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
        self.__status_length = 0
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
            if self.phase_name is not None:
                text = "[%s] %s" % (self.phase_name, text)
            self.__echo(text)

    def __echo(self, text):
        self.__clear_status()
        self.__printer("  " * self.__indent + text)

    def __clear_status(self):
        sys.stdout.write('\r')
        sys.stdout.write(' ' * self.__status_length)
        sys.stdout.write('\r')
        self.__status_length = 0

    def __write_status(self, text):
        sys.stdout.write(text)
        self.__status_length += len(text)
        sys.stdout.flush()

    @property
    def __byte_sort_key(self):
        return lambda b: self.__byte_order.setdefault(b, (0, b))

    @property
    def best(self):
        return self.__best

    def sort_key(self, s):
        return (len(s), [self.__byte_sort_key(b) for b in s])

    def classify(self, string):
        if self.__volume >= Volume.debug:
            if self.__status_length == 0 and self.phase_name is not None:
                self.__write_status('(%s) ' % (self.phase_name,))
        key = cache_key(string)
        try:
            return self.__cache[key]
        except KeyError:
            pass

        if self.__volume >= Volume.debug:
            self.__write_status(".")

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

    @phase("targeted")
    def targeted_delete_intervals(self, string, criterion):
        i = 0
        while i < len(string):
            index = {}
            for j in range(i + 1, len(string)):
                index.setdefault(string[j], j)
            for j in sorted(index.values(), reverse=True):
                attempt = string[:i] + string[j:]
                if criterion(attempt):
                    string = attempt
                    break
            else:
                i += 1

    @phase("exhaustive")
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

    def __delete_pass_at(self, i, test_case, predicate):
        def check(k):
            return predicate(test_case[:i] + test_case[i + k:])
        if check(1):
            lo = 1
            hi = 2
            while i + hi <= len(test_case) and check(hi):
                hi *= 2
            if check(hi):
                test_case = test_case[:i]
            else:
                while lo + 1 < hi:
                    mid = (lo + hi) // 2
                    if check(mid):
                        lo = mid
                    else:
                        hi = mid
                test_case = test_case[:i] + test_case[i + lo:]
            assert predicate(test_case)
        return test_case

    def adaptive_greedy_search_pass(self, test_case, predicate):
        indices = list(range(len(test_case)))
        self.random.shuffle(indices)

        def ix_to_t(ix):
            return [test_case[i] for i in sorted(ix)]

        def index_predicate(ix):
            return predicate(ix_to_t(ix))

        i = 0
        while i < len(indices):
            indices = self.__delete_pass_at(i, indices, index_predicate)
            i += 1
        return ix_to_t(indices)

    @contextmanager
    def in_phase(self, phase_name):
        original = self.phase_name
        if original is not None:
            phase_name = "%s > %s" % (original, phase_name)
        self.__clear_status()
        try:
            self.phase_name = phase_name
            yield
        finally:
            self.phase_name = original
            self.__clear_status()

    def __repeatedly_select_index(self, string, shrinker):
        prev = None
        while string != prev:
            prev = string
            string = self.__single_pass_select_index(string, shrinker)
        return string

    def __single_pass_select_index(self, string, shrinker):
        inds = list(range(len(string)))
        self.random.shuffle(inds)
        for i in inds:
            if i < len(string):
                string = shrinker(string, i)
        return string

    def __ngrams_at(self, string, i):
        assert isinstance(string, bytes)

        def count(k):
            return string.count(string[i:i+k])

        if count(1) <= 1:
            return []
        lo = 1
        hi = 1
        while count(hi) > 1 and hi < len(string) - 1:
            hi *= 2
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if count(mid) > 1:
                lo = mid
            else:
                hi = mid

        canon = {}
        for k in range(1, hi):
            t = string[i:i+k]
            canon.setdefault(string.count(t), t)
        return sorted(canon.values(), key=len, reverse=True)

    def __all_ngrams(self, changing_string):
        used_ngrams = set()
        assert isinstance(changing_string(), bytes)

        indices = list(range(len(changing_string())))
        self.random.shuffle(indices)
        for i in indices:
            if i >= len(changing_string()):
                continue
            ngs = self.__ngrams_at(changing_string(), i)
            for ng in ngs:
                if ng not in used_ngrams:
                    used_ngrams.add(ng)
                    yield ng

    @phase("ngram-killing")
    def kill_ngrams(self, string, criterion):
        tokens = self.tokenize(string)
        counts = Counter(tokens)
        targets = [t for t, k in counts.items() if k > 1]
        targets.sort(key=lambda s: len(s) * counts[s])
        new_targets = []
        for t in targets:
            attempt = b''.join([s for s in tokens if s != t])
            if criterion(attempt):
                string = attempt
            else:
                new_targets.append(t)
        for t in new_targets:
            def replace(s):
                return b''.join([s if x == t else x for x in tokens])
            s = self.basic_shrink_string(
                t, lambda s: criterion(replace(s))
            )
            string = replace(s)
        return string

    def basic_shrink_string(self, string, criterion):
        if len(set(string)) == 1:
            for i in range(len(string)):
                attempt = string[:i]
                if criterion(attempt):
                    return attempt
            return string

        threshold = 4
        if criterion(b''):
            return b''
        else:
            string = self.byte_shrink_string(string, criterion)
            if len(string) <= threshold:
                string = self.expmin_string(string, criterion)
            return string

    def structure_shrink_string_2(self, string, criterion):
        cautious = True
        hit_timeout = False
        prev = None
        timeout = 5
        while cautious:
            if hit_timeout:
                timeout *= 2
            elif prev == string:
                cautious = False

            prev = string
            hit_timeout = False

            for token in self.__all_ngrams(lambda: string):
                parts = string.split(token)

                if len(parts) <= 1:
                    continue

                with self.in_phase("structured[%s] > partition[%d] by %r" % (
                    "cautious, %ds" % (
                        timeout,) if cautious else "aggressive",
                    len(parts), token,)
                ):
                    def ls_criterion(ls):
                        return criterion(token.join(ls))

                    if cautious:
                        attempt = list(parts)
                        t = self.random.randint(0, len(attempt) - 1)
                        del attempt[t]
                        if ls_criterion(attempt):
                            parts = attempt
                        else:
                            continue
                    if cautious:
                        parts, timed_out = self.run_with_timeout(
                            self.adaptive_greedy_search_pass,
                            parts, ls_criterion,
                            timeout,
                        )
                        hit_timeout = hit_timeout or timed_out
                    else:
                        parts = self.adaptive_greedy_search_pass(
                            parts, ls_criterion
                        )
                    string = token.join(parts)
        return string

    @phase("bytewise")
    def byte_shrink_string(self, string, criterion):
        return bytes(self.adaptive_greedy_search_pass(
            list(string), lambda ls: criterion(bytes(ls))))

    def demarcated_shrink(self, string, criterion):
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
            with self.in_phase("demarcated %r" % (c,)):
                parts = self.adaptive_greedy_search_pass(
                    parts, lambda ls: criterion(c.join(ls)))
                string = c.join(parts)
        return string

    def expensive_shrink_string(self, string, criterion):
        if len(string) <= SHORT:
            return string
        for k in range(2, SHORT):
            assert k > 1
            with self.in_phase("intervals %d" % (k,)):
                i = 0
                while i + k <= len(string):
                    attempt = string[:i] + string[i+k:]
                    if criterion(attempt):
                        string = attempt
                    else:
                        i += 1

        with self.in_phase("pairs"):
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

    @phase("tokenwise")
    def token_min_string(self, string, criterion):
        tokens = self.tokenize(string)
        self.debug("%d tokens, %d distinct" % (len(tokens), len(set(tokens))))
        k = 1
        while k < len(tokens):
            with self.in_phase("intervals %d" % (k,)):
                i = 0
                while i + k <= len(string):
                    shrunk = tokens[:i] + tokens[i+k:]
                    if criterion(b''.join(shrunk)):
                        tokens = shrunk
                    i += 1
            k += 1
        return b''.join(tokens)

    def tokenize(self, string):
        tokens = []
        i = 0
        while i < len(string):
            lo = 1
            hi = 2

            def check(k):
                if i + k >= len(string):
                    return False
                t = string[i:i+k]
                return t in string[i+k:] or t in string[:i]
            while check(hi):
                hi *= 2
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                if check(mid):
                    lo = mid
                else:
                    hi = mid
            tokens.append(string[i:i+lo])
            i += lo
        assert b''.join(tokens) == string
        return tokens

    def lexicographically_minimize_string(self, string, criterion):
        alphabet = sorted(set(string), key=self.__byte_sort_key)

        i = 0
        while i < len(string):
            for a in alphabet:
                if self.__byte_sort_key(string[i]) >= self.__byte_sort_key(a):
                    break

                def lower(k):
                    return (
                        string[:i] +
                        bytes([
                            min(c, a, key=self.__byte_sort_key)
                            for c in string[i:i+k]]) +
                        string[i+k:])
                k = find_large_n(
                    len(string) - i, lambda k: criterion(lower(k)))
                if k > 0:
                    string = lower(k)
                    break
            i += 1

        return string

    def shrink_string(self, string, criterion):
        assert criterion(string)
        if len(string) <= 1:
            return string

        prev = None
        while prev != string:
            prev = string

            state = ShrinkState(
                string=string, criterion=criterion, sort_key=self.sort_key,
                random=self.random, in_phase=self.in_phase
            )
            state.run()
            string = state.string
            string = self.lexicographically_minimize_string(
                string, criterion)
        return string

        if 6 < len(string) <= 10:
            string = self.byte_shrink_string(string, criterion)
        if len(string) <= 6:
            return self.expmin_string(string, criterion)
        if criterion(b''):
            return b''
        prev = None
        while prev != string:
            prev = string
            string = self.structure_shrink_string_2(string, criterion)
            string = self.byte_shrink_string(string, criterion)
            string = self.token_min_string(string, criterion)
            string = self.kill_ngrams(string, criterion)
            string = self.expensive_shrink_string(string, criterion)
            string = self.alphabet_removal(string, criterion)
            string = self.alphabet_minimize(string, criterion)
            string = self.lexicographically_minimize_string(string, criterion)
        return string

    @phase("alphabet removal")
    def alphabet_removal(self, string, criterion):
        alphabet = sorted(set(string), key=self.__byte_sort_key)

        def replace(a, b):
            assert a in string
            return bytes([b if s == a else s for s in string])

        for i, c in enumerate(alphabet):
            attempt = bytes([s for s in string if s != c])
            if criterion(attempt):
                string = attempt

        return string

    @phase("alphabet minimization")
    def alphabet_minimize(self, string, criterion):
        def replace(a, b):
            assert a in string
            return bytes([b if s == a else s for s in string])

        alphabet = sorted(set(string), key=self.__byte_sort_key)

        for i, c in enumerate(alphabet):
            if i > 0 and c in string:
                attempt = replace(c, alphabet[0])
                with self.in_phase("lowering %r" % (bytes([c]),)):
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

    def run_with_timeout(self, shrinker, test_case, predicate, timeout):
        best = test_case
        here = object()
        start = time.time()

        def wrapped(target):
            nonlocal best
            if time.time() >= start + timeout:
                raise Stop("STOP", here)
            else:
                result = predicate(target)
                if result and self.sort_key(target) < self.sort_key(best):
                    best = target
                return result

        try:
            shrinker(test_case, wrapped)
            return best, False
        except Stop as stop:
            if stop.args[1] is not here:
                raise
            return best, True


class ShrinkState(object):
    def __init__(self, string, criterion, sort_key, random, in_phase):
        self.__index = IndexTree(string)
        self.string = string
        self.__in_phase = in_phase
        self.__random = random
        self.__criterion = criterion
        self.__sort_key = sort_key
        self.__shrink_index = 2
        self.__shrinks = [
            self.opportunistic,
            self.adaptive,
            self.aggressive,
            self.exhaustive(4), self.exhaustive(5), self.exhaustive(6),
            self.exhaustive(7),
        ]
        self.timeout = 10

    def count(self, t):
        return self.__index.count(t)

    def criterion(self, target):
        if self.__criterion(target):
            if self.__sort_key(target) < self.__sort_key(self.string):
                self.string = target
                self.__index = IndexTree(self.string)
            return True
        return False

    def calc_alphabet(self):
        result = sorted(set(self.string))
        result.sort(key=self.count, reverse=True)
        result.sort(key=self.__byte_sort_key)
        return result

    def lexicographically_minimize_string(self):
        alphabet = self.calc_alphabet()

        for c in alphabet:
            if c not in self.string:
                continue

            with self.__in_phase("lower to %r" % (bytes([c]),)):
                string = self.string
                indices = [
                    i for i, d in enumerate(string)
                    if self.__byte_sort_key(d) > self.__byte_sort_key(c)
                ]

                def lower_criterion(ls):
                    seen = set(ls)
                    replaced = bytes([
                        d if i in seen else min(c, d, key=self.__byte_sort_key)
                        for i, d in enumerate(string)
                    ])
                    return self.criterion(replaced)

                shrinker = self.__shrinks[self.__shrink_index]
                shrinker(indices, lower_criterion)

    def __byte_sort_key(self, c):
        return self.__sort_key(bytes([c]))

    def once(self, test_case, predicate):
        i = self.__random.randint(0, len(test_case) - 1)
        attempt = list(test_case)
        del attempt[i]
        if predicate(attempt):
            return attempt
        else:
            return test_case

    def opportunistic(self, test_case, predicate):
        prev = None
        while prev != test_case:
            prev = test_case
            if len(test_case) <= 1:
                return test_case
            elif len(test_case) <= 2:
                for p in test_case:
                    q = [p]
                    if predicate(q):
                        return q
                return test_case
            i = self.__random.randint(1, len(test_case) - 2)
            test_case = self.__adaptive_pass(i, test_case, predicate)
        return test_case

    def __adaptive_interval(self, i, test_case, predicate):
        def deletable(u, v):
            if v > len(test_case):
                return False
            if u < 0:
                return False
            attempt = list(test_case)
            del attempt[u:v]
            return predicate(attempt)

        def check_k(k):
            result = deletable(i, i + k)
            return result

        k = find_large_n(len(test_case) - i, check_k)
        if k == 0:
            return i, i
        upper_bound = i + k
        k = find_large_n(
            i, lambda j: deletable(i - j, upper_bound))
        lower_bound = i - k
        return lower_bound, upper_bound

    def __adaptive_pass(self, i, test_case, predicate):
        u, v = self.__adaptive_interval(i, test_case, predicate)
        attempt = list(test_case)
        del attempt[u:v]
        assert predicate(attempt)
        return attempt

    def adaptive(self, test_case, predicate):
        i = 0
        while i < len(test_case):
            k = find_large_n(
                len(test_case) - i,
                lambda k: predicate(test_case[:i] + test_case[i + k:]))
            if k > 0:
                test_case = test_case[:i] + test_case[i + k:]
            i += 1
        return test_case

    def aggressive(self, test_case, predicate):
        test_case = self.adaptive(test_case, predicate)
        k = 2
        while k < len(test_case):
            with self.__in_phase("%d-intervals" % (k,)):
                i = 0
                while i + k <= len(test_case):
                    attempt = list(test_case)
                    del attempt[i:i+k]
                    assert len(attempt) < len(test_case)
                    if predicate(attempt):
                        test_case = attempt
                    else:
                        attempt = list(test_case)
                        del attempt[i+k-1]
                        del attempt[i]
                        assert len(attempt) < len(test_case)
                        if predicate(attempt):
                            test_case = attempt
                        else:
                            i += 1
            k += 1
        return test_case

    def exhaustive(self, k):
        def accept(test_case, predicate):
            if len(test_case) > k:
                test_case = self.aggressive(test_case, predicate)
            if len(test_case) <= k:
                def sort_key(b):
                    try:
                        return self.__sort_key(b''.join(b))
                    except TypeError:
                        return b

                subsets = []
                for s in range(1, 2 ** len(test_case) - 1):
                    bits = []
                    for x in test_case:
                        if s & 1:
                            bits.append(x)
                        s >>= 1
                        if not s:
                            break
                    subsets.append(bits)
                subsets.sort(key=lambda b: (len(b), sort_key(b)))
                for bits in subsets:
                    if predicate(bits):
                        return bits
                return test_case
            return test_case
        accept.__name__ = 'exhaustive(%d)' % (k,)
        return accept

    def partitions(self):

        with self.__in_phase("bytewise"):
            yield [bytes([c]) for c in self.string]
        used_ngrams = set()

        retry = True
        while retry:
            retry = False
            indices = list(range(len(self.string)))
            self.__random.shuffle(indices)
            for i in indices:
                if i >= len(self.string):
                    continue

                string = self.string

                extend_cache = {}

                def left_extend(t):
                    if i == 0:
                        return t

                    try:
                        return extend_cache[t]
                    except KeyError:
                        pass
                    assert string[i:i+len(t)] == t
                    n = self.count(t)
                    assert n > 1
                    k = find_large_n(
                        i, lambda k: self.count(string[i-k:i] + t) == n)
                    result = string[i-k:i] + t
                    extend_cache[t] = result
                    return result

                def is_good_ngram(k):
                    t = string[i:i+k]
                    if self.count(t) <= 1:
                        return False
                    t = left_extend(t)
                    return t not in used_ngrams

                k = find_large_n(len(string) - i, is_good_ngram)
                if k == 0:
                    continue
                ng = left_extend(string[i:i+k])

                c = self.count(ng)
                equiv = find_large_n(
                    len(ng), lambda k: self.count(ng[:-k]) == c)

                for k in range(len(ng), equiv - 1, -1):
                    t = ng[:k]
                    used_ngrams.add(t)

                if len(ng) <= 1:
                    continue
                partition = self.string.split(ng)
                assert len(partition) > 2
                for i in range(len(partition) - 1):
                    partition[i] += ng

                if len(ng) >= 40:
                    ps = "%s=%s..%d..%s" % (
                        hashlib.sha1(ng).hexdigest()[:8],
                        repr(ng[:8])[1:], len(ng) - 16, repr(ng[-8:])[1:],
                    )
                else:
                    ps = repr(ng)[1:]

                with self.__in_phase("partition %s" % (ps,)):
                    yield partition
                retry = True
                break

        alphabet = set(self.string)
        while alphabet:
            c = min(alphabet, key=self.count)
            alphabet.discard(c)
            if self.count(c) <= 1:
                continue
            c = bytes([c])
            partition = self.string.split(c)
            for i in range(len(partition) - 1):
                partition[i] += c
            with self.__in_phase("partition %r" % (c,)):
                yield partition

    def tokenize(self):
        tokens = []

        cache = {}

        def is_token(t):
            try:
                return cache[t]
            except KeyError:
                pass
            result = self.count(t) > 1
            cache[t] = result
            return result
        seen = set()
        i = 0
        while i < len(self.string):
            lo = 1
            hi = 2

            def check(k):
                if i + k >= len(self.string):
                    return False
                return is_token(self.string[i:i+k])
            while check(hi):
                hi *= 2
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                if check(mid):
                    lo = mid
                else:
                    hi = mid
            t = self.string[i:i+lo]
            tokens.append(t)
            seen.add(t)
            i += lo
        return tokens

    def partition_criterion(self, ls):
        return self.criterion(b''.join(ls))

    def kill_duplicates(self):
        tokens = self.tokenize()
        counts = Counter(tokens)
        targets = list(counts)
        targets.sort(key=lambda t: len(t) * counts[t], reverse=True)
        for t in targets:
            attempt = [s for s in tokens if s != t]
            if self.criterion(b''.join(attempt)):
                tokens = attempt

    def brute(self):
        indices = list(range(len(self.string)))
        self.__random.shuffle(indices)
        orig = self.string
        for i in indices:
            if self.string != orig:
                break
            prev = None
            while prev != self.string:
                prev = self.string

                def contains_at_least(k):
                    s = self.string[i:i+k]
                    inds = self.__index.indices(s)
                    if not inds:
                        return False
                    return inds[-1] >= i + k

                n = find_large_n(len(self.string) - i, contains_at_least)

                canon = {}

                for k in range(1, n):
                    s = self.string[i:i+k]
                    ix = [j for j in self.__index.indices(s) if j >= i + k]
                    assert ix
                    canon.setdefault(ix[0], ix)

                for s in sorted(canon.values(), reverse=True):
                    if self.criterion(self.string[:i] + self.string[s[0]:]):
                        find_large_n(
                            len(s) - 1,
                            lambda k: self.criterion(
                                self.string[:i] + self.string[s[k]:]))
                        break
                else:
                    string = self.string
                    find_large_n(
                        len(self.string) - i,
                        lambda n: self.criterion(string[:i] + string[i+n:])
                    )
            i += 1

    def run(self):
        passnum = 0
        while self.__shrink_index < len(self.__shrinks):
            prev = self.string
            passnum += 1
            shrinker = self.__shrinks[self.__shrink_index]
            with self.__in_phase(
                "%s [pass %s]" % (shrinker.__name__, passnum)
            ):
                for p in self.partitions():
                    assert b''.join(p) == self.string
                    assert self.partition_criterion(p)
                    with self.__in_phase("partition to %d" % (len(p),)):
                        p = shrinker(p, self.partition_criterion)
                    assert b''.join(p) == self.string
                    assert self.partition_criterion(p)

            if len(self.string) == len(prev):
                self.__shrink_index += 1
                passnum = 0


class Stop(Exception):
    pass


def shrink(*args, **kwargs):
    """Attempt to find a minimal version of initial that satisfies classify."""
    shrinker = Shrinker(*args, **kwargs)
    shrinker.shrink()
    return shrinker.best


def shortlex(b):
    return (len(b), b)


class Control(Exception):
    pass


PROBE_UP_TO = 4


def find_large_n(max_n, f):
    """Finds a reasonably large value in [0, max_n] such that f(x) is True.
    f(0) is assumed to be true, and this only guarantees finding the maximum
    if f is monotonic increasing.
    """
    if max_n == 0:
        return max_n
    if not f(1):
        return 0
    lo = 1
    hi = 2

    # Perform an exponential probe upwards to find a value where it becomes
    # false. We expect this to happen very quickly, so we probe at a slightly
    # relaxed rate. This means the first few values we try are 1, 2, 3, 5, 8.
    while hi <= max_n and f(hi):
        lo = hi
        hi *= 2

    if hi > max_n:
        if f(max_n):
            return max_n
        else:
            hi = max_n

    # We now know that f(lo) and not f(hi). We retain this invariant through
    # the loop.

    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if f(mid):
            lo = mid
        else:
            hi = mid
    return lo


class IndexTree(object):
    def __init__(self, string):
        self.string = string
        self.__root = [range(len(self.string)), None]
        self.__counts = {}

    def indices(self, t):
        if isinstance(t, int):
            t = bytes([t])

        target = self.__root

        for k, c in enumerate(t):
            if not target[0]:
                return ()

            if target[1] is None:
                partition = defaultdict(list)
                for i in target[0]:
                    if i + k < len(self.string):
                        partition[self.string[i + k]].append(i)
                target[1] = {
                    k: [tuple(v), None] for k, v in partition.items()
                }
            try:
                target = target[1][c]
            except KeyError:
                return ()
        return target[0]

    def count(self, t):
        if isinstance(t, int):
            return len(self.indices(t))

        try:
            return self.__counts[t]
        except KeyError:
            pass
        c = 0
        last = None
        for i in self.indices(t):
            if last is None or i >= last + len(t):
                c += 1
                last = i
        self.__counts[t] = c
        return c
