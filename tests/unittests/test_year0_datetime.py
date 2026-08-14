import datetime
import unittest
from unittest import mock

import bson
import pytz
import tzlocal
from singer import utils

import tap_mongodb.sync_strategies.common as common


class FlakyCursor:
    """A cursor that raises InvalidBSON once, then yields rows on rebuild."""
    def __init__(self, rows, calls, fail_times=1):
        self._rows = rows
        self._calls = calls
        self._fail_times = fail_times
        self._idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._calls['failures'] < self._fail_times:
            self._calls['failures'] += 1
            raise bson.errors.InvalidBSON('year 0 is out of range')
        if self._idx >= len(self._rows):
            raise StopIteration
        row = self._rows[self._idx]
        self._idx += 1
        return row


class TestClampedYearZeroDatetime(unittest.TestCase):
    def test_safe_transform_datetime_emits_year0_for_clamped_min(self):
        # DATETIME_CLAMP decodes a year-0 BSON datetime as naive datetime.min
        clamped = datetime.datetime.min
        result = common.safe_transform_datetime(clamped, ['created'])
        self.assertEqual(result, '0000-01-01T00:00:00.000000Z')

    def test_transform_value_emits_year0_for_clamped_min(self):
        row = {'created': datetime.datetime.min}
        out = {k: common.transform_value(v, [k]) for k, v in row.items()}
        self.assertEqual(out['created'], '0000-01-01T00:00:00.000000Z')

    def test_transform_value_handles_normal_datetime(self):
        normal = datetime.datetime(2020, 6, 15, 12, 0, 0, 123456)
        out = common.transform_value(normal, ['created'])
        expected = tzlocal.get_localzone().localize(normal).astimezone(pytz.UTC)
        self.assertEqual(out, utils.strftime(expected))

    def test_transform_value_handles_aware_datetime(self):
        aware = datetime.datetime(2020, 6, 15, 12, 0, 0, tzinfo=pytz.UTC)
        out = common.transform_value(aware, ['created'])
        self.assertEqual(out, utils.strftime(aware))


class TestFetchRowsWithInvalidBSONRetry(unittest.TestCase):
    def test_retries_once_and_yields_rows(self):
        calls = {'failures': 0, 'builds': 0}

        def build_cursor():
            calls['builds'] += 1
            return FlakyCursor([{'_id': 1}, {'_id': 2}], calls, fail_times=1)

        with mock.patch.object(common.LOGGER, 'critical') as crit:
            rows = list(common.fetch_rows_with_invalid_bson_retry(
                build_cursor, 'db.Coll', 'last_id_fetched', None))

        self.assertEqual(rows, [{'_id': 1}, {'_id': 2}])
        self.assertEqual(calls['builds'], 2)
        self.assertEqual(crit.call_count, 1)

    def test_reraises_after_second_failure(self):
        calls = {'failures': 0, 'builds': 0}

        def build_cursor():
            calls['builds'] += 1
            return FlakyCursor([], calls, fail_times=2)

        with mock.patch.object(common.LOGGER, 'critical') as crit:
            with self.assertRaises(bson.errors.InvalidBSON):
                list(common.fetch_rows_with_invalid_bson_retry(
                    build_cursor, 'db.Coll', 'last_id_fetched', 'abc123'))

        self.assertEqual(calls['builds'], 2)
        self.assertEqual(crit.call_count, 2)

    def test_no_retry_needed_on_clean_cursor(self):
        calls = {'failures': 0, 'builds': 0}

        def build_cursor():
            calls['builds'] += 1
            return FlakyCursor([{'_id': 1}], calls, fail_times=0)

        with mock.patch.object(common.LOGGER, 'critical') as crit:
            rows = list(common.fetch_rows_with_invalid_bson_retry(
                build_cursor, 'db.Coll', 'last_id_fetched', None))

        self.assertEqual(rows, [{'_id': 1}])
        self.assertEqual(calls['builds'], 1)
        crit.assert_not_called()


class TestFullTableSyncWithPoisonRow(unittest.TestCase):
    """sync_collection should survive an InvalidBSON raised mid-iteration
    by rebuilding the cursor once via the retry helper."""

    def _stream(self):
        return {
            'tap_stream_id': 'db-Coll',
            'stream': 'Coll',
            'metadata': [
                {'breadcrumb': (),
                 'metadata': {'database-name': 'db'}}
            ],
        }

    def test_full_table_survives_poison_row(self):
        import tap_mongodb.sync_strategies.full_table as full_table

        good_oid = bson.objectid.ObjectId()
        good_row = {'_id': good_oid, 'created': datetime.datetime.min}
        calls = {'failures': 0}

        def build_flaky(*args, **kwargs):
            return FlakyCursor([good_row], calls, fail_times=1)

        collection = mock.MagicMock()
        collection.find_one.return_value = {'_id': good_oid}
        collection.find.side_effect = build_flaky

        client = mock.MagicMock()
        client.__getitem__.return_value.__getitem__.return_value = collection

        state = {}
        stream = self._stream()

        # Counters are normally seeded by do_sync() before sync_collection runs
        common.COUNTS['db-Coll'] = 0
        common.TIMES['db-Coll'] = 0
        common.SCHEMA_COUNT['db-Coll'] = 0
        common.SCHEMA_TIMES['db-Coll'] = 0

        with mock.patch.object(common.LOGGER, 'critical'):
            full_table.sync_collection(client, stream, state, None)

        # The poison row was skipped past by the retry and the good row synced
        bookmarks = state['bookmarks']['db-Coll']
        self.assertTrue(bookmarks['initial_full_table_complete'])
        self.assertEqual(calls['failures'], 1)
        self.assertEqual(collection.find.call_count, 2)


if __name__ == '__main__':
    unittest.main()
