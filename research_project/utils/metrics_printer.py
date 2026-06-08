def decorate_metric_line(
        name, 
        metric,
        max_name_length,
        max_metric_length,
        first_line=False
):
    result_str = ""
    if not first_line:
        result_str += "├─" + max_name_length*"─" + "─┼─" + max_metric_length*"─" + "─┤\n"
    str_metric = str(metric)
    result_str += "│ " + (max_name_length-len(name))*" "  + f"{name} │ {str_metric}" + (max_metric_length-len(str_metric))*" " + " │\n"

    return result_str


def decorate_metrics(metrics):
    max_name_length = len("Name")
    max_metric_length = len("Value")
    for name, metric in metrics.items():
        max_name_length = max(max_name_length, len(name))
        max_metric_length = max(max_metric_length, len(str(metric)))

    metric_str = ""

    # Top
    metric_str += "╭─" + max_name_length*"─" + "───" + max_metric_length*"─" + "─╮\n"
    # Title
    title_str = "Metrics"
    metric_str += "│ " + title_str + (max_name_length+max_metric_length+3-len(title_str))*" " + " │\n"
    # Header
    metric_str += "├─" + max_name_length*"─" + "─┬─" + max_metric_length*"─" + "─┤\n"
    metric_str += decorate_metric_line(
        name="Name", 
        metric="Value",
        max_name_length=max_name_length,
        max_metric_length=max_metric_length,
        first_line=True
    )
    for name, metric in metrics.items():
        metric_str += decorate_metric_line(
            name=name, 
            metric=metric,
            max_name_length=max_name_length,
            max_metric_length=max_metric_length
        )

    metric_str += "╰─" + max_name_length*"─" + "─┴─" + max_metric_length*"─" + "─╯"

    return metric_str