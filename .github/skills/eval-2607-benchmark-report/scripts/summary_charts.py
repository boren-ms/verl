"""Add baseline-versus-candidate checkpoint charts to 2607 summary sheets."""

from __future__ import annotations

from collections import OrderedDict

from openpyxl.chart import BarChart, Reference
from openpyxl.chart.label import DataLabelList
from openpyxl.chart.series import DataPoint
from openpyxl.chart.shapes import GraphicalProperties
from openpyxl.utils import get_column_letter

CHART_DATA_MARKER = "__chart_data__"


def _show_axis(axis, *, position: str) -> None:
    axis.delete = False
    axis.axPos = position
    axis.tickLblPos = "nextTo"
    axis.majorTickMark = "out"
    axis.spPr = GraphicalProperties()
    axis.spPr.ln.solidFill = "595959"
    axis.spPr.ln.width = 12700


def _show_zero_crossing_category_axis(axis) -> None:
    axis.delete = False
    axis.title = None
    axis.tickLblPos = "low"
    axis.majorTickMark = "none"
    axis.minorTickMark = "none"
    axis.spPr = GraphicalProperties()
    axis.spPr.ln.solidFill = "595959"
    axis.spPr.ln.width = 12700


def _single_summary_data(worksheet) -> list[tuple[str, list[tuple[str, float, float]]]]:
    candidate_header = str(worksheet.cell(1, 4).value or "Candidate")
    checkpoint = candidate_header.removeprefix("Candidate (").removesuffix(")")
    return [
        (
            str(worksheet.cell(row, 1).value),
            [(checkpoint, worksheet.cell(row, 3).value, worksheet.cell(row, 4).value)],
        )
        for row in range(2, worksheet.max_row + 1)
        if isinstance(worksheet.cell(row, 3).value, (int, float))
        and isinstance(worksheet.cell(row, 4).value, (int, float))
    ]


def _merged_summary_data(worksheet) -> list[tuple[str, list[tuple[str, float, float]]]]:
    groups: OrderedDict[str, list[tuple[str, float, float]]] = OrderedDict()
    for row in range(4, worksheet.max_row + 1):
        checkpoint = worksheet.cell(row, 1).value
        benchmark = worksheet.cell(row, 2).value
        metric = worksheet.cell(row, 3).value
        baseline = worksheet.cell(row, 4).value
        candidate = worksheet.cell(row, 5).value
        if not all(isinstance(value, str) for value in (checkpoint, benchmark, metric)):
            continue
        if not isinstance(baseline, (int, float)) or not isinstance(candidate, (int, float)):
            continue
        groups.setdefault(benchmark, []).append((checkpoint, baseline, candidate))
    return list(groups.items())


def _legacy_summary_data(worksheet) -> list[tuple[str, list[tuple[str, float, float]]]]:
    headers = [worksheet.cell(1, column).value for column in range(1, worksheet.max_column + 1)]
    metric_definition_column = headers.index("Metric definition") + 1
    candidate_columns = [
        column
        for column in range(4, metric_definition_column)
        if str(worksheet.cell(1, column).value).lower() != "delta"
    ]
    output = []
    for row in range(2, worksheet.max_row + 1):
        baseline = worksheet.cell(row, 3).value
        if not isinstance(baseline, (int, float)):
            continue
        points = [
            (str(worksheet.cell(1, column).value), baseline, worksheet.cell(row, column).value)
            for column in candidate_columns
            if isinstance(worksheet.cell(row, column).value, (int, float))
        ]
        if points:
            output.append((str(worksheet.cell(row, 1).value), points))
    return output


def _chart_data(worksheet) -> tuple[int, list[tuple[str, list[tuple[str, float, float]]]]]:
    marker_column = next(
        (column for column in range(1, worksheet.max_column + 1) if worksheet.cell(1, column).value == CHART_DATA_MARKER),
        None,
    )
    if marker_column is not None:
        return marker_column, []
    if worksheet.cell(3, 1).value == "Checkpoint":
        return worksheet.max_column + 10, _merged_summary_data(worksheet)
    if worksheet.cell(1, 1).value == "Benchmark" and str(worksheet.cell(1, 4).value).startswith("Candidate ("):
        return worksheet.max_column + 10, _single_summary_data(worksheet)
    if worksheet.cell(1, 1).value == "Benchmark":
        return worksheet.max_column + 10, _legacy_summary_data(worksheet)
    return worksheet.max_column + 10, []


def apply_summary_charts(worksheet) -> int:
    marker_column, data = _chart_data(worksheet)
    if not data:
        marker_column = next(
            (column for column in range(1, worksheet.max_column + 1) if worksheet.cell(1, column).value == CHART_DATA_MARKER),
            marker_column,
        )
        for row in range(1, worksheet.max_row + 1):
            for column in range(marker_column, min(marker_column + 4, worksheet.max_column) + 1):
                worksheet.cell(row, column).value = None
        if worksheet.cell(3, 1).value == "Checkpoint":
            data = _merged_summary_data(worksheet)
        elif worksheet.cell(1, 1).value == "Benchmark" and str(worksheet.cell(1, 4).value).startswith("Candidate ("):
            data = _single_summary_data(worksheet)
        elif worksheet.cell(1, 1).value == "Benchmark":
            data = _legacy_summary_data(worksheet)
    if not data:
        return 0

    worksheet._charts.clear()
    worksheet.cell(1, marker_column, CHART_DATA_MARKER)
    for column in range(marker_column, marker_column + 4):
        worksheet.column_dimensions[get_column_letter(column)].hidden = True

    if worksheet.cell(3, 1).value == "Checkpoint":
        visible_end_row = max(
            row
            for row in range(4, worksheet.max_row + 1)
            if isinstance(worksheet.cell(row, 1).value, str)
            and isinstance(worksheet.cell(row, 2).value, str)
            and isinstance(worksheet.cell(row, 3).value, str)
        )
    else:
        visible_end_row = max(
            row
            for row in range(2, worksheet.max_row + 1)
            if isinstance(worksheet.cell(row, 1).value, str)
            and isinstance(worksheet.cell(row, 2).value, str)
        )
    chart_start_row = visible_end_row + 3
    helper_row = 2
    worksheet.cell(helper_row, marker_column, "Checkpoint / dataset")
    worksheet.cell(helper_row, marker_column + 1, "Baseline")
    worksheet.cell(helper_row, marker_column + 2, "Candidate")
    worksheet.cell(helper_row, marker_column + 3, "Delta")
    flattened = [
        (
            title if len(points) == 1 else f"{title} | {checkpoint}",
            baseline,
            candidate,
            1 - candidate / baseline,
        )
        for title, points in data
        for checkpoint, baseline, candidate in points
        if baseline != 0
    ]
    for offset, (category, baseline, candidate, delta) in enumerate(flattened, start=1):
        worksheet.cell(helper_row + offset, marker_column, category)
        worksheet.cell(helper_row + offset, marker_column + 1, baseline).number_format = "0.00%"
        worksheet.cell(helper_row + offset, marker_column + 2, candidate).number_format = "0.00%"
        worksheet.cell(helper_row + offset, marker_column + 3, delta).number_format = "0.00%"

    chart = BarChart()
    chart.type = "col"
    chart.grouping = "clustered"
    chart.overlap = 0
    chart.title = None
    chart.height = 10
    chart.width = 20
    chart.y_axis.title = None
    chart.y_axis.numFmt = "0.00%"
    chart.y_axis.majorGridlines = None
    deltas = [delta for _, _, _, delta in flattened]
    minimum = min(deltas)
    maximum = max(deltas)
    span = maximum - minimum
    padding = span * 0.05 if span > 0 else max(abs(minimum), 0.01) * 0.05
    chart.y_axis.scaling.min = min(0, minimum - padding)
    chart.y_axis.scaling.max = max(0, maximum + padding)
    chart.y_axis.crossesAt = 0
    chart.x_axis.title = None
    _show_zero_crossing_category_axis(chart.x_axis)
    _show_axis(chart.y_axis, position="l")
    chart.legend = None
    chart.visible_cells_only = False
    chart.add_data(
        Reference(
            worksheet,
            min_col=marker_column + 3,
            min_row=helper_row,
            max_row=helper_row + len(flattened),
        ),
        titles_from_data=True,
    )
    categories = Reference(
        worksheet,
        min_col=marker_column,
        min_row=helper_row + 1,
        max_row=helper_row + len(flattened),
    )
    chart.set_categories(categories)
    chart.series[0].invertIfNegative = False
    chart.series[0].graphicalProperties.solidFill = "4472C4"
    chart.series[0].dPt = []
    for index, (_, _, _, delta) in enumerate(flattened):
        color = "63BE7B" if delta >= 0 else "F8696B"
        point = DataPoint(idx=index)
        point.invertIfNegative = False
        point.graphicalProperties.solidFill = color
        point.graphicalProperties.line.solidFill = color
        point.graphicalProperties.line.width = 12700
        chart.series[0].dPt.append(point)
    chart.dLbls = DataLabelList()
    chart.dLbls.showVal = True
    chart.dLbls.showCatName = False
    chart.dLbls.showSerName = False
    chart.dLbls.numFmt = "0.00%"
    chart.dLbls.position = "outEnd"

    worksheet.add_chart(chart, f"A{chart_start_row}")
    return 1