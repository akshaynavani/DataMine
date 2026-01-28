from datamine.reader import ArrowReader
from datamine.schema import PARTITION_COLUMNS
from datamine.storage import write_table


def _write(samples, tmp_path):
    uri = str(tmp_path / "samples")
    write_table(samples, uri, partition_cols=PARTITION_COLUMNS)
    return uri


def test_scan_with_partition_and_row_filters(samples, tmp_path):
    reader = ArrowReader(_write(samples, tmp_path))
    assert reader.count() == samples.num_rows
    front = reader.scan(cameras=["front"], min_labels=1)
    assert front.num_rows > 0
    assert set(front.column("camera").to_pylist()) == {"front"}
    assert min(front.column("num_labels").to_pylist()) >= 1
    assert front.column_names[0] == "sample_id"


def test_point_lookup_preserves_order(samples, tmp_path):
    reader = ArrowReader(_write(samples, tmp_path))
    wanted = samples.column("sample_id").to_pylist()[5:15][::-1]
    got = reader.get(wanted)
    assert got.column("sample_id").to_pylist() == wanted
    projected = reader.get(wanted[:3], columns=["image_uri"])
    assert projected.column_names == ["sample_id", "image_uri"]


def test_iter_batches_streams_all_rows(samples, tmp_path):
    reader = ArrowReader(_write(samples, tmp_path))
    total = sum(b.num_rows for b in reader.iter_batches(columns=["sample_id"], batch_size=50))
    assert total == samples.num_rows
