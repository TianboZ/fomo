from fomo.sentiment import parse_fear_greed


def test_parse_fear_greed_daily_last_value_wins():
    day = 1_767_225_600_000  # 2026-01-01 00:00 UTC in ms
    payload = {"fear_and_greed_historical": {"data": [
        {"x": day + 86_400_000, "y": 40.0},
        {"x": day, "y": 10.0},
        {"x": day + 3_600_000, "y": 12.5},  # intraday update of the same day
    ]}}
    s = parse_fear_greed(payload)
    assert [d.strftime("%Y-%m-%d") for d in s.index] == ["2026-01-01", "2026-01-02"]
    assert s.tolist() == [12.5, 40.0]
