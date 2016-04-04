from structureshrink import shrink, Volume
import os
from shutil import which
import click
import subprocess


def validate_command(ctx, param, value):
    if os.path.exists(value):
        return os.path.abspath(value)
    what = which(value)
    if what is None:
        raise click.BadParameter("%s: command not found" % (value,))
    return os.path.abspath(what)


@click.command()
@click.option("--debug", default=False, is_flag=True)
@click.option("--quiet", default=False, is_flag=True)
@click.option("--backup", default='')
@click.option(
    '--shrinks', default="shrinks",
    type=click.Path(file_okay=False, resolve_path=True))
@click.argument('filename', type=click.Path(exists=True, resolve_path=True))
@click.argument('test', callback=validate_command)
def shrinker(debug, quiet, backup, filename, test, shrinks):
    if debug and quiet:
        raise click.UsageError("Cannot have both debug output and be quiet")

    if not backup:
        backup = filename + os.extsep + "bak"

    try:
        os.mkdir(shrinks)
    except OSError:
        pass

    try:
        os.remove(backup)
    except FileNotFoundError:
        pass

    def classify(string):
        try:
            os.rename(filename, backup)
            with open(filename, 'wb') as o:
                o.write(string)
            try:
                subprocess.check_output([test])
                return 0
            except subprocess.CalledProcessError as e:
                return e.returncode
        finally:
            try:
                os.remove(filename)
            except FileNotFoundError:
                pass
            os.rename(backup, filename)

    with open(filename, 'rb') as o:
        initial = o.read()

    if debug:
        volume = Volume.debug
    elif quiet:
        volume = Volume.quiet
    else:
        volume = Volume.normal

    def shrink_callback(string, status):
        *base, ext = os.path.basename(filename).split(os.extsep, 1)
        base = os.extsep.join(base)
        with open(os.path.join(shrinks, os.path.extsep.join((
            base + "-%r" % (status,), ext
        ))), 'wb') as o:
            o.write(string)

    shrink(
        initial, classify, volume=volume,
        shrink_callback=shrink_callback, printer=click.echo
    )


if __name__ == '__main__':
    shrinker()
