import hashlib
from collections import OrderedDict, Counter
from enum import IntEnum
from random import Random
from functools import total_ordering
import heapq


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

NEWLINE = b'\n'


class Shrinker(object):

    def __init__(
        self,
        initial, classify, *,
        preprocess=None, shrink_callback=None, printer=None,
        volume=Volume.quiet, principal_only=False,
        passes=None, seed=None, preserve_lines=False
    ):
        self.__preserve_lines = preserve_lines
        self.__random = Random(seed)
        self.__interesting_ngrams = set()
        self.__shrink_callback = shrink_callback or (lambda s, r: None)
        self.__printer = printer or (lambda s: None)
        self.__inital = initial
        self.__classify = classify
        self.__preprocess = preprocess or (lambda s: s)
        self.__volume = volume
        self.__explain_lines = []

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
            self.__printer(text)

    def explain(self, text):
        self.__clear_explain()
        self.__explain_lines.append(text)

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
                if self.best and (
                    not self.principal_only or result == self.__initial_label
                ):
                    self.shrinks += 1
                    if result not in self.best:
                        self.__process_explain()
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

                        self.__process_explain()
                        self.output(
                            'Shrink %d: Label %r now %d bytes (%s)' % (
                                self.shrinks, result, len(string),
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

    def shrink_string(self, string, criterion):
        if self.__preserve_lines:
            def basic_partition(string):
                return [s + NEWLINE for s in string.split(NEWLINE)]
        else:
            def basic_partition(string):
                return [bytes([c]) for c in string]

        def partition(string):
            return merge_partition(basic_partition(string))

        if criterion(b''):
            return b''

        basic = basic_partition(string)
        if len(basic) <= MAX_K:
            return b''.join(_partymin(
                basic, lambda ls: criterion(b''.join(ls))))

        max_k = 2
        prev = None
        while prev != string or max_k <= MAX_K:
            prev = string

            if self.pass_enabled('partition-shared'):
                self.debug("Partitioning by tokens")

                used = set()

                assert criterion(string)
                tokens = partition(string)
                counts = Counter(tokens)
                assert counts[b''] == 0
                interesting = list(counts)

                def make_queue():
                    q = [
                        ((-len(t), string.count(t)), ReversedKey(shortlex(t)))
                        for t in interesting
                        if t not in used
                    ]
                    heapq.heapify(q)
                    return q
                queue = make_queue()
                while queue:
                    t = heapq.heappop(queue)[1].target[1]
                    used.add(t)
                    ls = string.split(t)
                    if len(ls) > 1:
                        assert t.join(ls) == string
                        ls = intercalate(ls, t)
                        assert b''.join(ls) == string
                        ls = list(filter(None, ls))
                        self.explain((
                            "Partitioning string of length %d by %r "
                            "into %d parts") % (
                            len(string), t, len(ls)))
                        orig = len(ls)
                        ls = _partymin(
                            ls, lambda x: criterion(b''.join(x)),
                            max_k=max_k
                        )
                        new_string = b''.join(ls)
                        assert criterion(new_string)
                        if string != new_string:
                            assert len(ls) < orig
                            self.debug("Reduced to %d/%d parts" % (
                                len(ls), orig))
                            string = new_string
                            queue = make_queue()
                            continue

            if self.pass_enabled('shared-shrink'):
                tokens = partition(string)
                for t, _ in sorted(
                    Counter(tokens).items(),
                    key=lambda x: (x[1], shortlex(x[0])),
                    reverse=True
                ):
                    ls = string.split(t)
                    if len(ls) <= 2:
                        continue
                    self.explain("Shrinking %r in %d parts " % (
                        t, len(ls),
                    ))
                    s = self.shrink_string(t, lambda s: criterion(s.join(ls)))
                    if t != s:
                        self.debug("Shrunk %r to %r" % (t, s))
                    string = s.join(ls)

            if self.pass_enabled('tokenwise'):
                ls = partition(string)
                self.explain("Minimizing tokenwise from %d tokens" % (
                    len(ls),))
                string = b''.join(_partymin(
                    ls, lambda l: criterion(b''.join(l)), max_k=max_k))

            if self.pass_enabled("basic"):
                self.debug("Minimizing %d bytes" % (len(string),))
                party = basic_partition(string)
                string = b''.join(
                    _partymin(
                        party, lambda ls: criterion(b''.join(ls)), max_k=max_k)
                )

            if string == prev:
                max_k += 1
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
                self.output(
                    'Shrinking for label %r from %d bytes (%d distinct)' % (
                        label, len(current), len(set(current))))

                def criterion(s):
                    return self.classify(s) == label

                self.shrink_string(self.best[label], criterion)


def _expmin(ls, criterion, sort_key=None):
    if criterion([]):
        return []
    if sort_key is None:
        def sort_key(b):
            return shortlex(b''.join(b))
    subsets = []
    for s in range(1, 2 ** len(ls) - 1):
        bits = []
        for x in ls:
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
    return ls


EXP_THRESHOLD = 4


MAX_K = 16


def _partymin(ls, criterion, max_k=16):
    if criterion([]):
        return []

    if len(ls) <= EXP_THRESHOLD:
        return _expmin(ls, criterion)

    for k in range(1, max_k + 1):
        def calc():
            return sorted(
                range(len(ls) + 1 - k),
                key=lambda i: shortlex(b''.join(ls[i:i+k])), reverse=True)

        indices = calc()

        prev = None
        while ls != prev:
            prev = ls
            i = 0
            while i < len(indices):
                if (
                    len(ls) <= EXP_THRESHOLD or
                    2 ** len(ls) <= len(ls) * (max_k - k)
                ):
                    return _expmin(ls, criterion)
                j = indices[i]
                assert j + k <= len(ls)
                ts = list(ls)
                del ts[j:j+k]
                assert len(ts) + k == len(ls)
                if criterion(ts):
                    ls = ts
                    indices = calc()
                    continue
                i += 1
    if len(ls) <= EXP_THRESHOLD:
        ls = _expmin(ls, criterion)
    return ls


def shrink(*args, **kwargs):
    """Attempt to find a minimal version of initial that satisfies classify."""
    shrinker = Shrinker(*args, **kwargs)
    shrinker.shrink()
    return shrinker.best


def shortlex(b):
    return (len(b), b)


def tokenize(string):
    return merge_partition([bytes(string[i:i+1]) for i in range(len(string))])


def merge_partition(partition):
    partition = list(partition)
    if len(partition) <= 1:
        return partition
    string = b''.join(partition)

    assert b''.join(partition) == string

    _tokens = {}

    def token_for(s):
        t = _tokens[s]
        assert t == s
        return t

    prev = None
    while partition != prev:
        prev = partition
        bigram_index = {}
        new_partition = []
        for token in partition:
            new_partition.append(token)

            while len(new_partition) >= 2:
                bigram = new_partition[-2] + new_partition[-1]

                def merge_top():
                    t = new_partition.pop()
                    s = new_partition.pop()
                    assert s
                    assert t
                    assert s + t == bigram
                    new_partition.append(token_for(bigram))

                try:
                    bigram = token_for(bigram)
                except KeyError:
                    pass
                else:
                    merge_top()
                try:
                    i = bigram_index[bigram]
                except KeyError:
                    bigram_index[bigram] = len(new_partition) - 2
                    break
                if i + 4 <= len(new_partition):
                    existing_bigram = new_partition[i] + new_partition[i + 1]
                    if existing_bigram == bigram:
                        _tokens[bigram] = bigram
                        merge_top()
                        new_partition[i] = bigram
                        del new_partition[i + 1]
                        continue
                    else:
                        bigram_index[bigram] = len(new_partition) - 2
                break
        assert b''.join(new_partition) == string, string
        partition = new_partition
    return partition


def split_list(ls, t):
    if not ls:
        return []
    parts = [[]]
    for l in ls:
        if l == t:
            parts.append([])
        elif not parts:
            parts.append([l])
        else:
            parts[-1].append(l)
    return parts


def intercalate(parts, t):
    result = []
    for i, p in enumerate(parts):
        if i > 0:
            result.append(t)
        result.append(p)
    return result


@total_ordering
class ReversedKey(object):
    def __init__(self, target):
        self.target = target

    def __eq__(self, other):
        return isinstance(other, ReversedKey) and (self.target == other.target)

    def __lt__(self, other):
        return self.target > other.target
