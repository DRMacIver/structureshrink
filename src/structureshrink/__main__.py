from structureshrink import Shrinker, Volume
import os
from shutil import which
import shlex
import click
import subprocess
import hashlib
import signal
import sys
import time
import random
import traceback


def validate_command(ctx, param, value):
    if value is None:
        return None
    parts = shlex.split(value)
    command = parts[0]

    if os.path.exists(command):
        command = os.path.abspath(command)
    else:
        what = which(command)
        if what is None:
            raise click.BadParameter('%s: command not found' % (command,))
        command = os.path.abspath(what)
    return [command] + parts[1:]


def signal_group(sp, signal):
    gid = os.getpgid(sp.pid)
    assert gid != os.getgid()
    os.killpg(gid, signal)


def interrupt_wait_and_kill(sp):
    if sp.returncode is None:
        # In case the subprocess forked. Python might hang if you don't close
        # all pipes.
        for pipe in [sp.stdout, sp.stderr, sp.stdin]:
            if pipe:
                pipe.close()
        try:
            signal_group(sp, signal.SIGINT)
            for _ in range(10):
                if sp.poll() is not None:
                    return
                time.sleep(0.1)
            signal_group(sp, signal.SIGKILL)
        except ProcessLookupError:
            return


@click.command(
    help="""
structureshrink takes a file and a test command and attempts to produce a
minimal example of every distinct status code it sees out of that command.
(Normally you're only interested in one, but sometimes there are other
interesting behaviours that occur while running it).

Usage is 'structureshrink test filename filenames...'. The file will be
repeatedly overwritten with a smaller version of it, with a backup placed in
the backup file. When the program exits the file will be replaced with the
smallest contents that produce the same exit code as was originally present.

Additional files will not be replaced but will be used as additional examples,
which may discover other interesting files as well as aiding the shrinking
process.
""".strip()
)
@click.option('--debug', default=False, is_flag=True, help=(
    'Emit (extremely verbose) debug output while shrinking'
))
@click.option('--principal', default=False, is_flag=True, help=(
    'When set will only try to shrink examples that classify the same as the '
    'initial example (other values will still be recorded but it will not make'
    ' any deliberate attempts to shrink them).'
))
@click.option(
    '--quiet', default=False, is_flag=True, help=(
        'Emit no output at all while shrinking'))
@click.option(
    '--backup', default='', help=(
        'Name of the backup file to create. Defaults to adding .bak to the '
        'name of the source file'))
@click.option(
    '--shrinks', default='shrinks',
    type=click.Path(file_okay=False, resolve_path=True))
@click.option('--seed', default=None)
@click.option(
    '--preprocess', default=None, callback=validate_command,
    help=(
        "Provide a command that 'normalizes' the input before it is tested ("
        'e.g. a code formatter). If this command returns a non-zero exit code '
        'then the example will be skipped altogether.'))
@click.option(
    '--timeout', default=1, type=click.FLOAT, help=(
        'Time out subprocesses after this many seconds. If set to <= 0 then '
        'no timeout will be used.'))
@click.option(
    '--pass', '-p', 'passes', multiple=True,
    help='Run only a single pass'
)
@click.option('--classify', default=None, callback=validate_command)
@click.argument('test', callback=validate_command)
@click.argument('filename', type=click.Path(
    exists=True, resolve_path=True, dir_okay=False, allow_dash=True
))
@click.argument('filenames', type=click.Path(
    exists=True, resolve_path=True, dir_okay=False, allow_dash=False
), nargs=-1)
def shrinker(
    debug, quiet, backup, filename, test, shrinks, preprocess, timeout,
    classify, filenames, seed, principal, passes
):
    if debug and quiet:
        raise click.UsageError('Cannot have both debug output and be quiet')

    if debug:
        def dump_trace(signum, frame):
            traceback.print_stack()
        signal.signal(signal.SIGQUIT, dump_trace)

    if seed is not None:
        random.seed(seed)

    if not backup:
        backup = filename + os.extsep + 'bak'

    history = os.path.join(shrinks, 'history')

    try:
        os.mkdir(shrinks)
    except OSError:
        pass

    try:
        os.mkdir(history)
    except OSError:
        pass

    try:
        os.remove(backup)
    except FileNotFoundError:
        pass

    seen_output = set()

    def classify_data(string):
        if filename == '-':
            sp = subprocess.Popen(
                test, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                universal_newlines=False,
                preexec_fn=os.setsid,
            )
            try:
                sp.communicate(string, timeout=timeout)
            finally:
                interrupt_wait_and_kill(sp)
            result = sp.returncode
        else:
            try:
                os.rename(filename, backup)
                with open(filename, 'wb') as o:
                    o.write(string)
                sp = subprocess.Popen(
                    test, stdout=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, universal_newlines=False,
                    preexec_fn=os.setsid,
                )
                try:
                    sp.communicate(timeout=timeout)
                except subprocess.TimeoutExpired:
                    return 'timeout'
                finally:
                    interrupt_wait_and_kill(sp)
                return sp.returncode
            finally:
                try:
                    os.remove(filename)
                except FileNotFoundError:
                    pass
                os.rename(backup, filename)
        if classify is None or result is None:
            return result
        else:
            try:
                classify_output = subprocess.check_output(
                    classify, timeout=timeout, stdin=subprocess.DEVNULL)
                classify_return = 0
            except subprocess.CalledProcessError as e:
                classify_output = e.output
                classify_return = e.returncode
            if classify_output and classify_output not in seen_output:
                shrinker.debug(
                    'New classification: %r' % (classify_output,)
                )
                seen_output.add(classify_output)
            return ':%d:%d:%s:' % (
                result, classify_return,
                hashlib.sha1(classify_output).hexdigest()[:8]
                if classify_output else '.'
            )

    timeout *= 10
    if timeout <= 0:
        timeout = None

    if preprocess:
        def preprocessor(string):
            sp = subprocess.Popen(
                preprocess, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                universal_newlines=False,
                preexec_fn=os.setsid,
            )
            try:
                out, _ = sp.communicate(string, timeout=timeout)
                assert isinstance(out, bytes)
                return out
            except subprocess.TimeoutExpired:
                shrinker.debug('Timed out while calling preprocessor')
                return None
            except subprocess.CalledProcessError:
                shrinker.debug('Error while calling preprocessor')
                return None
            finally:
                interrupt_wait_and_kill(sp)
    else:
        preprocessor = None

    if filename == '-':
        initial = sys.stdin.buffer.read()
    else:
        with open(filename, 'rb') as o:
            initial = o.read()

    if debug:
        volume = Volume.debug
    elif quiet:
        volume = Volume.quiet
    else:
        volume = Volume.normal

    def suffixed_name(status):
        if filename == '-':
            base = ''
            ext = 'example'
        else:
            *base, ext = os.path.basename(filename).split(os.extsep, 1)
            base = os.extsep.join(base)
        if base:
            return os.path.extsep.join(((base, '%s' % (status,), ext)))
        else:
            return os.path.extsep.join(((ext, '%s' % (status,))))

    def shrink_callback(string, status):
        with open(os.path.join(shrinks, suffixed_name(status)), 'wb') as o:
            o.write(string)
        with open(
            os.path.join(history, suffixed_name(
                '%d-%s' % (len(string), hashlib.sha1(string).hexdigest()[:12])
            )), 'wb'
        ) as o:
            o.write(string)
    shrinker = Shrinker(
        initial, classify_data, volume=volume,
        shrink_callback=shrink_callback, printer=click.echo,
        preprocess=preprocessor, principal_only=principal,
        passes=passes or None,
    )
    initial_label = shrinker.classify(initial)
    # Go through the old shrunk files. This both reintegrates them into our
    # current shrink state so we can resume and also lets us clear out old bad
    # examples.
    try:
        for f in os.listdir(shrinks):
            path = os.path.join(shrinks, f)
            if not os.path.isfile(path):
                continue
            with open(path, 'rb') as i:
                contents = i.read()
            status = shrinker.classify(contents)
            if suffixed_name(status) != f:
                shrinker.debug('Clearing out defunct %r file' % (f,))
                os.unlink(path)
            else:
                shrinker.debug(
                    'Reusing previous %d byte example for label %r' % (
                        len(contents), status
                    ))
        for f in os.listdir(history):
            path = os.path.join(history, f)
            if not os.path.isfile(path):
                continue
            with open(path, 'rb') as i:
                contents = i.read()
            if principal and len(contents) > len(initial):
                continue
            shrinker.classify(contents)

        for filepath in filenames:
            with open(filepath, 'rb') as i:
                value = i.read()
            shrinker.classify(value)

        if timeout is not None:
            timeout //= 10
        shrinker.shrink()
    finally:
        if filename != '-':
            os.rename(filename, backup)
            with open(filename, 'wb') as o:
                o.write(shrinker.best[initial_label])
        else:
            sys.stdout.buffer.write(shrinker.best[initial_label])


if __name__ == '__main__':
    shrinker()
