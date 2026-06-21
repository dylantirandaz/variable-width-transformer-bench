from vwt_bench.widths import uniform_widths, x_shape_widths


def test_uniform_widths() -> None:
    schedule = uniform_widths(num_layers=4, width=32)
    assert schedule.widths == [32, 32, 32, 32]
    assert schedule.square_sum == 4 * 32 * 32


def test_x_shape_widths_are_symmetric_and_narrower_in_middle() -> None:
    schedule = x_shape_widths(
        num_layers=6,
        base_width=96,
        bottleneck_layer_ratio=0.75,
        bottleneck_width_ratio=0.30,
        quantize_to=4,
    )
    assert len(schedule.widths) == 6
    assert schedule.widths[0] == schedule.widths[-1]
    assert min(schedule.widths) == schedule.widths[schedule.bottleneck_layer - 1]
    assert all(width % 4 == 0 for width in schedule.widths)
    assert schedule.average_width < 96
    assert abs(schedule.square_sum - schedule.target_square_sum) / schedule.target_square_sum < 0.20
