from contextlib import contextmanager
import hashlib
from functools import lru_cache
import os


class StringTable(object):

    def __init__(self, path):
        if path == ":memory:":
            path = None
        self.__path = path

    @lru_cache()
    def id_to_string(self, id):
        if self.__path is None:
            return id
        target = os.path.join(self.__path, id)
        with open(target, 'rb') as i:
            return i.read()

    @lru_cache()
    def string_to_id(self, string):
        if self.__path is None:
            return string
        hash = hashlib.sha1(string).hexdigest()[:8]
        target = os.path.join(self.__path, hash)
        if not os.path.exists(target):
            try:
                with open(target, 'xb') as o:
                    o.write(string)
            except FileExistsError:
                pass
        return hash
