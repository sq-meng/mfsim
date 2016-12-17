from django.http import HttpResponse, Http404
from django.shortcuts import render
import flexxtools as ft
import numpy as np
from bokeh.plotting import figure, output_file, show
from bokeh.models import Range1d, HoverTool, ColumnDataSource, SingleIntervalTicker
from bokeh.embed import components
from bokeh.layouts import gridplot
from bokeh.resources import CDN, INLINE
from pyclipper import Pyclipper, PFT_NONZERO, PFT_POSITIVE, PT_SUBJECT, PT_CLIP, scale_from_clipper, scale_to_clipper, CT_UNION, CT_INTERSECTION

fmt2 = '{:.2f}'.format
fmt1 = '{:.1f}'.format


def nice_vector(v):
    return '[%s, %s, %s]' % (fmt1(v[0]), fmt1(v[1]), fmt1(v[2]))


def myroot(request):
    return render(request, 'index.html', {})


def multiflexx_sim(request):
    scan_data = request.GET
    try:
        if scan_data['submitted'] == 'Ja':
            submitted = True
        else:
            submitted = False
    except KeyError:
        submitted = False
    scan_rows = []
    script, div = '', ''

    if submitted:
        scan_data_nice = extract_data(scan_data)
        scan_rows = make_scan_rows(scan_data_nice)
        script, div = make_figures(scan_data_nice)
        s2, d2 = make_figures(scan_data_nice)
    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()
    tag_lib = {'scan_data': scan_data, 'scan_rows': scan_rows, 'graph_script': script, 'graph_div': div,
               'js_resources': js_resources, 'css_resources': css_resources, 'submitted': submitted, 's2': s2, 'd2': d2}
    tag_lib.update(scan_data)

    return render(request, 'multiflexx.html', tag_lib)


def extract_data(get):
    floatize = lambda string_list: [float(x) for x in string_list]
    latparam = floatize([get['lat_a'], get['lat_b'], get['lat_c'], get['lat_alpha'], get['lat_beta'], get['lat_gamma']])
    hkl1 = floatize([get['align_h1'], get['align_k1'], get['align_l1']])
    hkl2 = floatize([get['align_h2'], get['align_k2'], get['align_l2']])
    plot_x = floatize([get['plot_h1'], get['plot_k1'], get['plot_l1']])
    plot_y = floatize([get['plot_h2'], get['plot_k2'], get['plot_l2']])
    eis = floatize(get['ei_list'].split(','))
    A3_starts = floatize(get['A3_start_list'].split(','))
    A3_ends = floatize(get['A3_end_list'].split(','))
    A4_starts = floatize(get['A4_start_list'].split(','))
    A4_ends = floatize(get['A4_end_list'].split(','))
    return dict(latparam=latparam, hkl1=hkl1, hkl2=hkl2, plot_x=plot_x, plot_y=plot_y, eis=eis, A3_starts=A3_starts,
                A3_ends=A3_ends, A4_starts=A4_starts, A4_ends=A4_ends)


def make_scan_rows(scan):
    return [dict(ei=fmt2(ei), A3_start=fmt2(A3_start), A3_end=fmt2(A3_end),
                 A4_start=fmt2(A4_start), A4_end=fmt2(A4_end))
            for ei, A3_start, A3_end, A4_start, A4_end in zip(scan['eis'], scan['A3_starts'], scan['A3_ends'], scan['A4_starts'], scan['A4_ends'])]


def make_figures(scan):
    ub_matrix = ft.UBMatrix(scan['latparam'], scan['hkl1'], scan['hkl2'], scan['plot_x'], scan['plot_y'])
    A3_starts, A3_ends, A4_starts, A4_ends = (scan['A3_starts'], scan['A3_ends'], scan['A4_starts'], scan['A4_ends'])
    kis = [ft.e_to_k(e) for e in scan['eis']]
    kfs = [ft.e_to_k(e) for e in ft.EF_LIST]
    unique_kis = sorted(list(set(kis)))
    scan_indexes = [[ind for ind in range(len(kis)) if kis[ind] == ki] for ki in unique_kis]
    colors = ['#FFCE98', '#F6FF8D', '#94FFD5', '#909CFF', '#FF8AD8']

    locuses_dict = {}
    for nth, ki in enumerate(unique_kis):
        clippers = [Pyclipper() for i in range(5)]
        for scan_no in scan_indexes[nth]:
            angles = (A3_starts[scan_no], A3_ends[scan_no], A4_starts[scan_no], A4_ends[scan_no], ub_matrix)
            locuses = [ft.calculate_locus(ki, kf, *angles) for kf in kfs]
            for i in range(5):
                clippers[i].AddPath(scale_to_clipper(locuses[i]), PT_SUBJECT)

        locuses_ki = [scale_from_clipper(clippers[i].Execute(CT_UNION, PFT_NONZERO)) for i in range(5)]
        locuses_ki_x, locuses_ki_y = split_locus_lists(locuses_ki)
        common_locus_x, common_locus_y = find_common_coverage(locuses_ki)
        locuses_dict[ki] = [locuses_ki_x, locuses_ki_y, common_locus_x, common_locus_y]

    p_col = []
    x_axis = np.array(scan['plot_x'])
    y_axis = np.array(scan['plot_y'])
    for ki in locuses_dict.keys():
        TOOLS = "pan,wheel_zoom,reset,save"
        p = figure(plot_width=450, plot_height=500, title='Ei = %s meV' % fmt2(ft.k_to_e(ki)), tools=TOOLS)
        p.xaxis.axis_label = 'x * %s' % nice_vector(x_axis)
        p.yaxis.axis_label = 'y * %s' % nice_vector(y_axis)
        ticker = SingleIntervalTicker(interval=0.5, num_minor_ticks=1)
        p.axis.ticker = ticker
        p.grid.ticker = ticker
        loc_test = locuses_dict[ki]
        efs_str = [fmt1(ft.k_to_e(ki) - ft.k_to_e(kf)) for kf in kfs]
        for i in reversed(range(5)):
            color = colors[i]
            x_list = loc_test[0][i]
            y_list = loc_test[1][i]
            channel = p.patches(x_list, y_list, alpha=0.35, fill_color=color, line_width=1, legend='dE='+efs_str[i])
            set_aspect(p, x_list[0], y_list[0], aspect=ub_matrix.figure_aspect)
        common = p.patches(loc_test[2][0], loc_test[3][0], fill_alpha=0.0, line_width=1.2, legend='Common',
                           line_color='red')
        glyph_dots = plot_lattice_points(p, x_axis, y_axis)
        hover = HoverTool(renderers=[glyph_dots], tooltips=[('coord', '@coord')])
        p.add_tools(hover)
        p_col.append(p)
    for each in p_col:
        each.x_range = p_col[0].x_range
        each.y_range = p_col[0].y_range

    grid = gridplot(p_col, ncols=2)
    script, div = components(grid, CDN)
    return script, div


def split_locus_lists(locuses):
    no_locuses = len(locuses)
    locuses_x, locuses_y = [[] for _ in range(no_locuses)], [[] for _ in range(no_locuses)]
    for i in range(no_locuses):
        locus_ki = locuses[i]
        for path in locus_ki:
            locuses_x[i].append(list(np.array(path)[:, 0]))
            locuses_y[i].append(list(np.array(path)[:, 1]))
    return locuses_x, locuses_y


def find_common_coverage(locuses):
    clipper = Pyclipper()
    current_locus = locuses[0]
    for locus in locuses[1:]:
        clipper.AddPaths(scale_to_clipper(current_locus), PT_SUBJECT)
        clipper.AddPaths(scale_to_clipper(locus), PT_CLIP)
        current_locus = scale_from_clipper(clipper.Execute(CT_INTERSECTION))
        clipper.Clear()
    l_x, l_y = split_locus_lists([current_locus])
    return l_x, l_y


def set_aspect(fig, x, y, aspect=1, margin=0.1):
    """Set the plot ranges to achieve a given aspect ratio.

    Args:
      fig (bokeh Figure): The figure object to modify.
      x (iterable): The x-coordinates of the displayed data.
      y (iterable): The y-coordinates of the displayed data.
      aspect (float, optional): The desired aspect ratio. Defaults to 1.
        Values larger than 1 mean the plot is squeezed horizontally.
      margin (float, optional): The margin to add for glyphs (as a fraction
        of the total plot range). Defaults to 0.1
    """

    xmin = min(xi for xi in x)
    xmax = max(xi for xi in x)
    ymin = min(yi for yi in y)
    ymax = max(yi for yi in y)
    width = (xmax - xmin)*(1+2*margin)
    if width <= 0:
        width = 1.0
    height = (ymax - ymin)*(1+2*margin)
    if height <= 0:
        height = 1.0
    xcenter = 0.5*(xmax + xmin)
    ycenter = 0.5*(ymax + ymin)
    r = aspect*(fig.plot_width/fig.plot_height)
    if width < r*height:
        width = r*height
    else:
        height = width/r
    fig.x_range = Range1d(xcenter-0.5*width, xcenter+0.5*width)
    fig.y_range = Range1d(ycenter-0.5*height, ycenter+0.5*height)


def plot_lattice_points(p, x_axis, y_axis):
    x, y = np.mgrid[-3:3:0.5, -3:3:0.5]
    xr = list(np.reshape(x, -1))
    yr = list(np.reshape(y, -1))
    ttip = []
    for cx, cy in zip(xr, yr):
        ttip.append(nice_vector(cx * x_axis + cy * y_axis))
    source = ColumnDataSource(data=dict(x=xr, y=yr, coord=ttip))
    glyph = p.circle('x', 'y', source=source, size=9, fill_alpha=0.3)
    return glyph
