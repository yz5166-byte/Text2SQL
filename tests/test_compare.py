from text2sql.eval.compare import bird_ex, lenient_ex, values_equal


def test_values_equal_numbers_and_rounding():
    assert values_equal(265, 265.0)
    assert values_equal(0.13675213675213677, 0.1368)  # rounded to 4 decimals
    assert values_equal(0.13675213675213677, 0.14)  # rounded to 2 decimals
    assert not values_equal(0.13675213675213677, 13.68)  # percent scale is a different value
    assert not values_equal(5879.427, 5879.0)  # integer-looking rounding is not accepted
    assert not values_equal(698697005, 698000000)
    assert values_equal("1998", 1998)
    assert values_equal(" Prague ", "prague")
    assert not values_equal(None, 0)


def test_extra_columns_are_allowed():
    gold = [("north Moravia",)]
    pred = [("north Moravia", 409)]
    assert lenient_ex(pred, gold)
    assert not bird_ex(pred, gold)


def test_column_order_and_row_order():
    gold = [("1998-01", 223), ("1998-02", 189)]
    pred = [(189, "1998-02"), (223, "1998-01")]
    assert lenient_ex(pred, gold)


def test_order_matters():
    gold = [("a", 3), ("b", 2), ("c", 1)]
    assert lenient_ex([("a", 3), ("b", 2), ("c", 1)], gold, order_matters=True)
    assert not lenient_ex([("b", 2), ("a", 3), ("c", 1)], gold, order_matters=True)
    assert lenient_ex([("b", 2), ("a", 3), ("c", 1)], gold, order_matters=False)


def test_rows_must_be_jointly_consistent():
    gold = [("x", 1), ("y", 2)]
    pred = [("x", 2), ("y", 1)]  # each column matches alone, rows do not
    assert not lenient_ex(pred, gold)


def test_compare_query_subset():
    gold = [(265, 259)]
    pred = [(265, 259, 6, 0.023166)]
    assert lenient_ex(pred, gold)


def test_row_count_mismatch():
    assert not lenient_ex([("a", 1), ("b", 0)], [("a", 1)])
    assert lenient_ex([], [])
