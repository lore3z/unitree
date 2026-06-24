import glob
import math
import os
import struct
from collections import defaultdict
from xml.sax.saxutils import escape


LOG_DIR = 'logs/g1_arms'
OUT_DIRS = ['.', 'thesis/images']
RED = '#d62728'
BLUE = '#1f77b4'
GRID = '#cccccc'
TEXT = '#222222'


def get_latest_log(log_dir):
    subdirs = [
        os.path.join(log_dir, name)
        for name in os.listdir(log_dir)
        if os.path.isdir(os.path.join(log_dir, name)) and name != 'exported'
    ]
    if not subdirs:
        return None
    latest_subdir = max(subdirs, key=os.path.getmtime)
    events = glob.glob(os.path.join(latest_subdir, 'events.out.tfevents.*'))
    return events[0] if events else None


def read_tfrecords(path):
    with open(path, 'rb') as handle:
        while True:
            header = handle.read(8)
            if len(header) < 8:
                return
            length = struct.unpack('<Q', header)[0]
            handle.read(4)
            payload = handle.read(length)
            handle.read(4)
            if payload:
                yield payload


def read_varint(buffer, index):
    value = 0
    shift = 0
    while True:
        byte = buffer[index]
        index += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, index
        shift += 7


def skip_field(buffer, index, wire_type):
    if wire_type == 0:
        _, index = read_varint(buffer, index)
        return index
    if wire_type == 1:
        return index + 8
    if wire_type == 2:
        length, index = read_varint(buffer, index)
        return index + length
    if wire_type == 5:
        return index + 4
    raise ValueError(f'Unsupported wire type: {wire_type}')


def parse_summary_value(buffer):
    tag = None
    simple_value = None
    index = 0
    while index < len(buffer):
        key, index = read_varint(buffer, index)
        field = key >> 3
        wire_type = key & 7
        if field == 1 and wire_type == 2:
            length, index = read_varint(buffer, index)
            tag = buffer[index:index + length].decode('utf-8', errors='ignore')
            index += length
        elif field == 2 and wire_type == 5:
            simple_value = struct.unpack_from('<f', buffer, index)[0]
            index += 4
        else:
            index = skip_field(buffer, index, wire_type)
    return tag, simple_value


def parse_summary(buffer):
    values = []
    index = 0
    while index < len(buffer):
        key, index = read_varint(buffer, index)
        field = key >> 3
        wire_type = key & 7
        if field == 1 and wire_type == 2:
            length, index = read_varint(buffer, index)
            values.append(parse_summary_value(buffer[index:index + length]))
            index += length
        else:
            index = skip_field(buffer, index, wire_type)
    return values


def parse_event(buffer):
    step = None
    summaries = []
    index = 0
    while index < len(buffer):
        key, index = read_varint(buffer, index)
        field = key >> 3
        wire_type = key & 7
        if field == 2 and wire_type == 0:
            step, index = read_varint(buffer, index)
        elif field == 5 and wire_type == 2:
            length, index = read_varint(buffer, index)
            summaries.extend(parse_summary(buffer[index:index + length]))
            index += length
        else:
            index = skip_field(buffer, index, wire_type)
    return step, summaries


def collect_scalars(event_path):
    series = defaultdict(list)
    for record in read_tfrecords(event_path):
        step, summaries = parse_event(record)
        if step is None:
            continue
        for tag, value in summaries:
            if tag and value is not None:
                series[tag].append((step, value))
    for tag in list(series.keys()):
        series[tag].sort(key=lambda item: item[0])
    return series


def smooth(values, weight=0.95):
    if not values:
        return []
    smoothed = []
    last = values[0]
    for value in values:
        last = last * weight + (1 - weight) * value
        smoothed.append(last)
    return smoothed


def format_number(value, precision=2):
    if abs(value) >= 1000 or (abs(value) > 0 and abs(value) < 0.01):
        return f'{value:.1e}'
    text = f'{value:.{precision}f}'
    return text.rstrip('0').rstrip('.') if '.' in text else text


def polyline(points, color, width=2.5, opacity=1.0):
    path = ' '.join(f'{x:.2f},{y:.2f}' for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="{width}" stroke-opacity="{opacity}" points="{path}" />'


def line(x1, y1, x2, y2, color, width=1, dash=None, opacity=1.0):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ''
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}" stroke-opacity="{opacity}"{dash_attr} />'


def text(x, y, content, size=14, anchor='middle', rotate=None, weight='normal'):
    transform = f' transform="rotate({rotate:.2f},{x:.2f},{y:.2f})"' if rotate is not None else ''
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="serif" font-size="{size}" '
        f'fill="{TEXT}" text-anchor="{anchor}" font-weight="{weight}"{transform}>'
        f'{escape(content)}</text>'
    )


def series_bounds(series):
    xs = [point[0] for point in series]
    ys = [point[1] for point in series]
    return min(xs), max(xs), min(ys), max(ys)


def render_chart(title, xlabel, ylabel, series, out_path, width=1280, height=720, legend_label='G1 29-DoF (Full body)', smooth_weight=None, y_precision=2):
    margin_left = 90
    margin_right = 40
    margin_top = 70
    margin_bottom = 80
    plot_x0 = margin_left
    plot_y0 = margin_top
    plot_x1 = width - margin_right
    plot_y1 = height - margin_bottom
    plot_w = plot_x1 - plot_x0
    plot_h = plot_y1 - plot_y0

    x_min, x_max, y_min, y_max = series_bounds(series)
    if x_max == x_min:
        x_max += 1
    if y_max == y_min:
        y_max += 1
    y_pad = (y_max - y_min) * 0.08
    y_min -= y_pad
    y_max += y_pad

    def sx(x):
        return plot_x0 + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y):
        return plot_y1 - (y - y_min) / (y_max - y_min) * plot_h

    lines = []
    lines.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white" />')
    lines.append(text(width / 2, 36, title, size=30, weight='normal'))

    grid_count = 5
    for i in range(grid_count + 1):
        x = plot_x0 + i * plot_w / grid_count
        value = x_min + i * (x_max - x_min) / grid_count
        lines.append(line(x, plot_y0, x, plot_y1, GRID, width=1, dash='6,6', opacity=0.85))
        lines.append(text(x, plot_y1 + 28, format_number(value, precision=0 if x_max - x_min > 100 else 1), size=14))

    for i in range(grid_count + 1):
        y = plot_y1 - i * plot_h / grid_count
        value = y_min + i * (y_max - y_min) / grid_count
        lines.append(line(plot_x0, y, plot_x1, y, GRID, width=1, dash='6,6', opacity=0.85))
        lines.append(text(plot_x0 - 12, y + 5, format_number(value, precision=y_precision), size=14, anchor='end'))

    lines.append(line(plot_x0, plot_y0, plot_x0, plot_y1, '#222222', width=1.5))
    lines.append(line(plot_x0, plot_y1, plot_x1, plot_y1, '#222222', width=1.5))
    lines.append(text((plot_x0 + plot_x1) / 2, height - 28, xlabel, size=18))
    lines.append(text(24, (plot_y0 + plot_y1) / 2, ylabel, size=18, anchor='middle', rotate=-90))

    raw_points = [(sx(x), sy(y)) for x, y in series]
    lines.append(polyline(raw_points, RED, width=1.5, opacity=0.18))
    if smooth_weight is not None:
        smoothed = smooth([y for _, y in series], weight=smooth_weight)
        smooth_points = [(sx(x), sy(y)) for (x, _), y in zip(series, smoothed)]
        lines.append(polyline(smooth_points, RED, width=4.0, opacity=0.98))
    else:
        lines.append(polyline(raw_points, RED, width=3.2, opacity=0.98))

    legend_x = plot_x1 - 260
    legend_y = plot_y1 - 58
    lines.append(f'<rect x="{legend_x:.2f}" y="{legend_y:.2f}" width="240" height="46" rx="8" ry="8" fill="white" stroke="#d0d0d0" stroke-width="1" opacity="0.95" />')
    lines.append(line(legend_x + 18, legend_y + 23, legend_x + 64, legend_y + 23, RED, width=4.0, opacity=0.98))
    lines.append(text(legend_x + 72, legend_y + 28, legend_label, size=16, anchor='start'))

    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">' + ''.join(lines) + '</svg>'
    with open(out_path, 'w', encoding='utf-8') as handle:
        handle.write(svg)


def render_grid(title, out_path, metrics, series_map, width=1440, height=1080, legend_label='G1 29-DoF (Full body)'):
    margin_left = 66
    margin_right = 36
    margin_top = 72
    margin_bottom = 54
    gap_x = 28
    gap_y = 38
    plot_w = (width - margin_left - margin_right - gap_x) / 2
    plot_h = (height - margin_top - margin_bottom - 2 * gap_y) / 3

    def subplot_box(index):
        row = index // 2
        col = index % 2
        x0 = margin_left + col * (plot_w + gap_x)
        y0 = margin_top + row * (plot_h + gap_y)
        x1 = x0 + plot_w
        y1 = y0 + plot_h
        return x0, y0, x1, y1

    def sx(x, x_min, x_max, x0, x1):
        return x0 + (x - x_min) / (x_max - x_min) * (x1 - x0)

    def sy(y, y_min, y_max, y0, y1):
        return y1 - (y - y_min) / (y_max - y_min) * (y1 - y0)

    lines = []
    lines.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white" />')
    lines.append(text(width / 2, 34, title, size=26))

    for index, (tag, subplot_title) in enumerate(metrics):
        if tag not in series_map:
            continue
        row = index // 2
        col = index % 2
        series = series_map[tag]
        x0, y0, x1, y1 = subplot_box(index)
        x_min, x_max, y_min, y_max = series_bounds(series)
        if x_max == x_min:
            x_max += 1
        if y_max == y_min:
            y_max += 1
        y_pad = (y_max - y_min) * 0.08
        y_min -= y_pad
        y_max += y_pad

        lines.append(text((x0 + x1) / 2, y0 - 8, subplot_title, size=15, weight='bold'))
        lines.append(line(x0, y0, x0, y1, '#222222', width=1.2))
        lines.append(line(x0, y1, x1, y1, '#222222', width=1.2))

        for tick in range(4):
            frac = tick / 3 if 3 else 0
            x_tick = x0 + frac * (x1 - x0)
            x_value = x_min + frac * (x_max - x_min)
            lines.append(line(x_tick, y0, x_tick, y1, GRID, width=1, dash='5,5', opacity=0.8))
            if row == 2:
                lines.append(text(x_tick, y1 + 18, format_number(x_value, precision=0), size=11))

        for tick in range(4):
            frac = tick / 3 if 3 else 0
            y_tick = y1 - frac * (y1 - y0)
            y_value = y_min + frac * (y_max - y_min)
            lines.append(line(x0, y_tick, x1, y_tick, GRID, width=1, dash='5,5', opacity=0.8))
            if col == 0:
                lines.append(text(x0 - 8, y_tick + 4, format_number(y_value, precision=3), size=11, anchor='end'))

        raw_points = [(sx(x, x_min, x_max, x0, x1), sy(y, y_min, y_max, y0, y1)) for x, y in series]
        lines.append(polyline(raw_points, RED, width=1.3, opacity=0.15))
        lines.append(polyline([(sx(x, x_min, x_max, x0, x1), sy(y, y_min, y_max, y0, y1)) for (x, _), y in zip(series, smooth([y for _, y in series], weight=0.95))], RED, width=2.6, opacity=0.98))
        if index == 0:
            lx = x1 - 178
            ly = y0 + 8
            lines.append(f'<rect x="{lx:.2f}" y="{ly:.2f}" width="170" height="28" rx="6" ry="6" fill="white" stroke="#d0d0d0" stroke-width="1" opacity="0.95" />')
            lines.append(line(lx + 10, ly + 14, lx + 42, ly + 14, RED, width=2.6, opacity=0.98))
            lines.append(text(lx + 48, ly + 18, legend_label, size=11, anchor='start'))
        if row == 2:
            lines.append(text((x0 + x1) / 2, y1 + 40, 'Training Steps', size=12))
        if col == 0:
            lines.append(text(x0 - 42, (y0 + y1) / 2, 'Reward value', size=12, anchor='middle', rotate=-90))

    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">' + ''.join(lines) + '</svg>'
    with open(out_path, 'w', encoding='utf-8') as handle:
        handle.write(svg)


def main():
    event_path = get_latest_log(LOG_DIR)
    if not event_path:
        raise SystemExit(f'No TensorBoard logs found under {LOG_DIR}')

    series_map = collect_scalars(event_path)

    required = {
        'reward': ['Episode/rew_total', 'Train/mean_reward', 'rewards/total'],
        'loss': ['Loss/value_function'],
        'episode': ['Train/mean_episode_length'],
    }

    resolved = {}
    for key, candidates in required.items():
        for candidate in candidates:
            if candidate in series_map:
                resolved[key] = series_map[candidate]
                break
        else:
            raise SystemExit(f'Missing required tag for {key}: {candidates}')

    reward_svg = 'paper_reward_comparison.svg'
    loss_svg = 'paper_loss_convergence.svg'
    episode_svg = 'paper_episode_length_convergence.svg'
    grid_svg = 'paper_subrewards_grid.svg'

    render_chart('Training Reward Comparison', 'Training Steps', 'Total Reward', resolved['reward'], reward_svg, smooth_weight=0.98, y_precision=2)
    render_chart('Value Function Loss Convergence', 'Training Steps', 'Value Loss', resolved['loss'], loss_svg, smooth_weight=0.95, y_precision=3)
    render_chart('Episode Length Convergence', 'Training Steps', 'Mean Episode Length', resolved['episode'], episode_svg, smooth_weight=0.95, y_precision=0)

    metrics = [
        ('Episode/rew_tracking_lin_vel', 'Linear Velocity Tracking'),
        ('Episode/rew_tracking_ang_vel', 'Angular Velocity Tracking'),
        ('Episode/rew_torques', 'Torque Penalty (Energy)'),
        ('Episode/rew_action_rate', 'Action Rate (Smoothness)'),
        ('Episode/rew_orientation', 'Posture Penalty (Orientation)'),
        ('Episode/rew_base_height', 'Base Height Maintenance'),
    ]
    render_grid('29-DOF Subreward Breakdown', grid_svg, metrics, series_map, legend_label='G1 29-DoF (Full body)')

    for out_dir in OUT_DIRS:
        os.makedirs(out_dir, exist_ok=True)
        for svg_name in [reward_svg, loss_svg, episode_svg, grid_svg]:
            source = svg_name
            target = os.path.join(out_dir, svg_name)
            if os.path.abspath(source) != os.path.abspath(target):
                with open(source, 'r', encoding='utf-8') as src_handle, open(target, 'w', encoding='utf-8') as dst_handle:
                    dst_handle.write(src_handle.read())

    print('Generated SVG figures:')
    for svg_name in [reward_svg, loss_svg, episode_svg, grid_svg]:
        print(f'  {svg_name}')


if __name__ == '__main__':
    main()