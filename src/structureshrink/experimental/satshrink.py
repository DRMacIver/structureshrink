from structureshrink.utils import dfa


def satshrink(string, criterion):
    if criterion(b''):
        return b''

    builder = dfa.PTABuilder()
    builder.add(b'', False)
    builder.add(string, True)
    k = len(string) - 1
    while k > 0:
        i = 0
        while i + k <= len(string):
            ts = string[:i] + string[i + k:]
            assert len(ts) < len(string)
            if criterion(ts):
                string = ts
                builder.add(string, True)
            else:
                builder.add(ts, False)
                i += k
        k //= 2
    for i in range(len(string)):
        p = string[:i]
        r = criterion(p)
        builder.add(p, r)
        if r:
            break

    while True:
        minimachine = dfa.reduce_dfa(builder.build())
        path = minimachine.find_label_matching(bool)
        print(path)
        if criterion(path):
            return path
        else:
            builder.add(path, False)
