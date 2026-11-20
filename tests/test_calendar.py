from whymoved.data.calendar import effective_day

# Fri 2021-12-03, Mon 2021-12-06 trading; weekend not.
CAL = ["2021-12-02", "2021-12-03", "2021-12-06", "2021-12-07"]


def test_before_close_same_day():
    # 14:00 UTC = 09:00 ET (winter) -> same day
    assert effective_day("2021-12-03 14:00:00", CAL) == "2021-12-03"


def test_after_close_next_day():
    # 22:00 UTC = 17:00 ET
    assert effective_day("2021-12-03 22:00:00", CAL) == "2021-12-06"


def test_exactly_16_is_next_day():
    assert effective_day("2021-12-02 21:00:00", CAL) == "2021-12-03"


def test_weekend_goes_to_monday():
    assert effective_day("2021-12-04 15:00:00", CAL) == "2021-12-06"


def test_tz_aware_input():
    assert effective_day("2021-12-03T09:30:00-05:00", CAL) == "2021-12-03"


def test_past_calendar_end():
    assert effective_day("2021-12-07 23:00:00", CAL) is None
