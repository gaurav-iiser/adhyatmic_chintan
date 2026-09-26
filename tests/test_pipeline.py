from adhyatmik.pipeline import format_timestamp, slugify


def test_format_timestamp():
    assert format_timestamp(0) == "00:00:00"
    assert format_timestamp(65) == "00:01:05"
    assert format_timestamp(3661) == "01:01:01"


def test_slugify():
    assert slugify("Bh Gita Ch 1 Part 19") == "Bh-Gita-Ch-1-Part-19"
