def decorate_line(
        name: str, 
        metric: float,
        max_name_length: int,
        max_metric_length: int
) -> str:
    pre_name_spaces = (max_name_length - len(name))*" "
    post_metric_spaces = (max_metric_length - len(str(metric)))*" "

    return f"│ {pre_name_spaces}{name} │ {metric}{post_metric_spaces} │\n"


def decorate_metrics(metrics: dict[str, float]) -> str:
    max_name_length = len("Name")
    max_metric_length = len("Value")
    for name, metric in metrics.items():
        max_name_length = max(max_name_length, len(name))
        max_metric_length = max(max_metric_length, len(str(metric)))

    max_name_horizontal = max_name_length*"─"
    max_metric_horizontal = max_metric_length*"─"
    title = "Metrics"
    title_spaces = (max_name_length + max_metric_length - len(title) + 3)*" "

    lines = [
        f"╭─{max_name_horizontal}───{max_metric_horizontal}─╮\n",
        f"│ {title}{title_spaces} │\n",
        f"├─{max_name_horizontal}─┬─{max_metric_horizontal}─┤\n",  #
        decorate_line("Name", "Value", max_name_length, max_metric_length)
    ]

    for name, metric in metrics.items():
        lines.extend([
            f"├─{max_name_horizontal}─┼─{max_metric_horizontal}─┤\n",
            decorate_line(name, metric, max_name_length, max_metric_length)
        ])

    lines.append(
        f"╰─{max_name_horizontal}─┴─{max_metric_horizontal}─╯"
    )

    return "".join(lines)