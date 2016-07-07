import sqlite3
from contextlib import contextmanager
import hashlib
from functools import lru_cache


class StringTable(object):

    def __init__(self, path):
        self.__database = sqlite3.connect(path)
        with self.__cursor() as c:
            c.execute("""
                create table if not exists string_mapping(
                    id integer primary key,
                    hash text unique,
                    data blob
                )
            """)

    @lru_cache()
    def id_to_string(self, id):
        with self.__cursor() as c:
            c.execute("""select data from string_mapping where id=?""", (id,))
            for (data,) in c:
                return data
            raise KeyError('No string with ID %r' % (id,))

    @lru_cache()
    def string_to_id(self, string):
        with self.__cursor() as c:
            hash = hashlib.sha1(string).hexdigest()
            c.execute('select id from string_mapping where hash=?', (
                hash,))
            for (id,) in c:
                return id
            c.execute('insert into string_mapping (hash, data) values(?, ?)', (
                hash, string))
            return c.lastrowid

    @contextmanager
    def __cursor(self):
        conn = self.__database
        cursor = conn.cursor()
        try:
            try:
                yield cursor
            finally:
                cursor.close()
        except:
            conn.rollback()
            raise
        else:
            conn.commit()
