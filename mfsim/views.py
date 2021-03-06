from django.http import HttpResponse, Http404
from django.shortcuts import render
import flexxtools as ft
import numpy as np
import xtal
from bokeh.plotting import figure, output_file, show
from bokeh.models import Range1d, HoverTool, ColumnDataSource, SingleIntervalTicker, CustomJS, Div, PreText
from bokeh.embed import components
from bokeh.models.widgets import RadioButtonGroup
from bokeh.layouts import gridplot, column, layout
from bokeh.resources import CDN, INLINE
from bokeh.events import MouseMove
from pyclipper import Pyclipper, PFT_NONZERO, PFT_POSITIVE, PT_SUBJECT, PT_CLIP, scale_from_clipper, scale_to_clipper, CT_UNION, CT_INTERSECTION, ClipperException

fmt2 = '{:.2f}'.format
fmt1 = '{:.1f}'.format
fmt0 = '{:.0f}'.format


def bracketed_vector(v):
    return '[%s, %s, %s]' % (fmt1(v[0]), fmt1(v[1]), fmt1(v[2]))


def site_root(request):
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
    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()
    tag_lib = {'scan_data': scan_data, 'scan_rows': scan_rows, 'graph_script': script, 'graph_div': div,
               'js_resources': js_resources, 'css_resources': css_resources, 'submitted': submitted}
    tag_lib.update(scan_data)

    return render(request, 'multiflexx.html', tag_lib)


def float_each(string_list):
    return [float(x) for x in string_list]


def extract_data(get):
    latparam = float_each([get['lat_a'], get['lat_b'], get['lat_c'], get['lat_alpha'], get['lat_beta'], get['lat_gamma']])
    hkl1 = float_each([get['align_h1'], get['align_k1'], get['align_l1']])
    hkl2 = float_each([get['align_h2'], get['align_k2'], get['align_l2']])
    plot_x = float_each([get['plot_h1'], get['plot_k1'], get['plot_l1']])
    plot_y = float_each([get['plot_h2'], get['plot_k2'], get['plot_l2']])
    eis = float_each(get['ei_list'].split(','))
    a3_starts = float_each(get['A3_start_list'].split(','))
    a3_ends = float_each(get['A3_end_list'].split(','))
    a4_starts = float_each(get['A4_start_list'].split(','))
    a4_ends = float_each(get['A4_end_list'].split(','))
    number_points = float_each(get['NP_list'].split(','))
    hm = get['horizontal_magnet']
    if hm != 'no':
        hm_hkl = float_each([get['hm_h'], get['hm_k'],get['hm_l']])
        hm_ssr = float(get['hm_ssr'])
    else:
        hm_hkl = None
        hm_ssr = None
    return dict(latparam=latparam, hkl1=hkl1, hkl2=hkl2, plot_x=plot_x, plot_y=plot_y, eis=eis, A3_starts=a3_starts,
                A3_ends=a3_ends, A4_starts=a4_starts, A4_ends=a4_ends, NPs=number_points, hm=hm, hm_hkl=hm_hkl,
                hm_ssr=hm_ssr)


def make_scan_rows(scan):
    return [dict(ei=fmt2(ei), A3_start=fmt2(A3_start), A3_end=fmt2(A3_end),
                 A4_start=fmt2(A4_start), A4_end=fmt2(A4_end), NP=fmt0(NP))
            for ei, A3_start, A3_end, A4_start, A4_end, NP in zip(scan['eis'], scan['A3_starts'], scan['A3_ends'],
                                                                  scan['A4_starts'], scan['A4_ends'], scan['NPs'])]


def make_figures(scan):
    ub_matrix = ft.UBMatrix(scan['latparam'], scan['hkl1'], scan['hkl2'], scan['plot_x'], scan['plot_y'])
    A3_starts, A3_ends, A4_starts, A4_ends= (scan['A3_starts'], scan['A3_ends'], scan['A4_starts'], scan['A4_ends'])
    NPs = [int(x) for x in scan['NPs']]
    kis = [ft.e_to_k(e) for e in scan['eis']]
    kfs = [ft.e_to_k(e) for e in ft.EF_LIST]
    hm = scan['hm']
    hm_hkl = scan['hm_hkl']
    hm_ssr = scan['hm_ssr']
    unique_kis = sorted(list(set(kis)))
    indexes_of_ki = [[ind for ind in range(len(kis)) if kis[ind] == ki] for ki in unique_kis]
    locus_palette = ['#FFCE98', '#F6FF8D', '#94FFD5', '#909CFF', '#FF8AD8']

    locuses_dict = {}
    scatters_dict = {}
    colors_dict = {}
    senses_dict = {}
    for nth_ki, ki in enumerate(unique_kis):
        clippers = [Pyclipper() for _ in range(ft.EF_CHANNELS)]
        scatter_arrays = [[] for _ in range(ft.EF_CHANNELS)]
        color_arrays = [[] for _ in range(ft.EF_CHANNELS)]
        for scan_no in indexes_of_ki[nth_ki]:
            angles = (A3_starts[scan_no], A3_ends[scan_no], A4_starts[scan_no], A4_ends[scan_no], ub_matrix)
            locuses = [ft.calculate_locus(ki, kf, *angles, no_points=NPs[scan_no]) for kf in kfs]
            scatter_coords = [ft.scatter_coords(ki, kf, *angles, no_points=NPs[scan_no]) for kf in kfs]
            scatter_colors = [ft.scatter_color(ki, kf, *angles, name=hm, ssr=hm_ssr, north=hm_hkl,
                                               no_points=NPs[scan_no]) for kf in kfs]
            if A4_starts[scan_no] > 0:
                senses_dict[ki] = 1
            else:
                senses_dict[ki] = -1

            for i in range(ft.EF_CHANNELS):
                clippers[i].AddPath(scale_to_clipper(locuses[i]), PT_SUBJECT)
                scatter_arrays[i] += scatter_coords[i]
                color_arrays[i] += scatter_colors[i]
        locuses_ki = [scale_from_clipper(clippers[i].Execute(CT_UNION, PFT_NONZERO)) for i in range(ft.EF_CHANNELS)]
        locuses_ki_x, locuses_ki_y = split_locus_lists(locuses_ki)
        scatters_ki_x, scatters_ki_y = split_scatter_lists(scatter_arrays)
        common_locus_x, common_locus_y = find_common_coverage(locuses_ki)
        locuses_dict[ki] = [locuses_ki_x, locuses_ki_y, common_locus_x, common_locus_y]
        scatters_dict[ki] = [scatters_ki_x, scatters_ki_y, color_arrays]

    p_col = []
    plots = []
    x_axis = np.array(scan['plot_x'])
    y_axis = np.array(scan['plot_y'])
    for ki in unique_kis:
        TOOLS = "pan,wheel_zoom,reset,save"
        main_plot = figure(plot_width=700, plot_height=600, title='Ei = %s meV' % fmt2(ft.k_to_e(ki)), tools=TOOLS)
        main_plot.xaxis.axis_label = 'x * %s' % bracketed_vector(x_axis)
        main_plot.yaxis.axis_label = 'y * %s' % bracketed_vector(y_axis)
        ticker = SingleIntervalTicker(interval=0.5, num_minor_ticks=1)
        main_plot.axis.ticker = ticker
        main_plot.grid.ticker = ticker
        locus = locuses_dict[ki]
        efs_str = [fmt1(ft.k_to_e(ki) - ft.k_to_e(kf)) for kf in kfs]
        sources = []
        source_handle = ColumnDataSource(dict(x=[], y=[], colors=[]))
        scatter_off = ColumnDataSource(dict(x=[], y=[], colors=[]))
        for i in reversed(range(ft.EF_CHANNELS)):
            color = locus_palette[i]
            x_list = locus[0][i]
            y_list = locus[1][i]
            main_plot.patches(x_list, y_list, alpha=0.35, fill_color=color, muted_fill_color='black', muted_fill_alpha=0.01,
                                        muted_line_alpha=0.1, line_width=1, legend='dE='+efs_str[i])
            set_aspect(main_plot, x_list[0], y_list[0], aspect=ub_matrix.figure_aspect)
        for i in range(ft.EF_CHANNELS):
            sources.append(ColumnDataSource(dict(x=scatters_dict[ki][0][i], y=scatters_dict[ki][1][i], colors=scatters_dict[ki][2][i])))
        main_plot.circle(x='x', y='y', size=4.5, fill_alpha=1,
                                            visible=True, fill_color='colors', line_alpha=0.2, source=source_handle)
        main_plot.patches(locus[2][0], locus[3][0], fill_alpha=0.0, line_width=1.2, legend='Common',
                           line_color='red', muted_line_alpha=0.0, muted_fill_alpha=0.0)
        glyph_dots = plot_lattice_points(main_plot, x_axis, y_axis)
        main_plot.legend.click_policy = 'mute'
        plot_brillouin_zones(main_plot, x_axis, y_axis)
        cs = sources
        callback = CustomJS(args=dict(s0=cs[0], s1=cs[1], s2=cs[2], s3=cs[3], s4=cs[4], s5=scatter_off, source=source_handle), code="""
                var f = cb_obj.active;                
                switch (f) {
                    case 0:
                        source.data = s0.data;
                        break;
                    case 1:
                        source.data = s1.data;
                        break;
                    case 2:
                        source.data = s2.data;
                        break;
                    case 3:
                        source.data = s3.data;
                        break;
                    case 4:
                        source.data = s4.data;
                        break;
                    case 5:
                        source.data = s5.data;
                        break;
                }
                source.change.emit();       
            """)
        en_buttons = RadioButtonGroup(labels=['2.5', '3.0', '3.5', '4.0', '4.5', 'Off'], active=5, callback=callback)
        en_button_caption = Div()
        en_button_caption.text = """<span style="font-weight: bold;">Active channel:</span>"""
        hover = HoverTool(renderers=[glyph_dots], tooltips=[('Q', '@coord')])
        main_plot.add_tools(hover)
        message_div = Div(width=600, height=200)
        if hm != 'no':
            plot_radar = draw_radar(main_plot, message_div, en_buttons, hm, ki, scan['hkl1'], hm_hkl, hm_ssr, ub_matrix, senses_dict[ki])
            plot_radar.axis.visible = False
            ctrl_col = column([en_button_caption, en_buttons, plot_radar, message_div])
        else:
            plot_radar = draw_radar(main_plot, message_div, en_buttons, hm, ki, scan['hkl1'], scan['hkl1'], 0,
                                    ub_matrix, senses_dict[ki])
            plot_radar.axis.visible = False
            ctrl_col = column([en_button_caption, en_buttons, plot_radar, message_div])
            # ctrl_col = column([en_button_caption, en_buttons, message_div])

        plots.append(main_plot)
        p_col.append([main_plot, ctrl_col])
    for each in plots:
        each.x_range = plots[0].x_range
        each.y_range = plots[0].y_range

    grid = layout(p_col)
    script, div = components(grid)

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


def plot_lattice_points(graph, x_axis, y_axis, bz3d=None):
    x, y = np.mgrid[-10:10:0.5, -10:10:0.5]
    xr = list(np.reshape(x, -1))
    yr = list(np.reshape(y, -1))
    tooltip = []
    fill_alpha = []
    size = []
    low_index_points = []
    for cx, cy in zip(xr, yr):
        coord = cx * x_axis + cy * y_axis
        tooltip.append(bracketed_vector(coord))
        if (coord == coord.astype(int)).all():
            fill_alpha.append(0.3)
            size.append(12)
            low_index_points.append((xr, yr))
        else:
            fill_alpha.append(0)
            size.append(9)
    source = ColumnDataSource(data=dict(x=xr, y=yr, coord=tooltip, fill_alpha=fill_alpha, size=size))
    circles = graph.circle('x', 'y', source=source, size='size', fill_alpha='fill_alpha')
    graph.circle_x(0, 0, size=20, line_color='blue', line_width=1.5, fill_alpha=0)
    return circles


def plot_brillouin_zones(graph, x_axis, y_axis):
    pass


def split_scatter_lists(scatter_arrays):
    scatter_x, scatter_y = [], []
    for i in range(5):
        scatter_array = np.array(scatter_arrays[i])
        scatter_x.append(scatter_array[:, 0])
        scatter_y.append(scatter_array[:, 1])
    return scatter_x, scatter_y


def draw_radar(main_plot, div, en_button, hm_name, ki, hkl1, north, ssr, ub_matrix, sense):
    radar = figure(plot_width=400, plot_height=400, title="Orientations", x_range=[-3, 3], y_range=[-3, 3], tools=[])
    ki_source, kf_source, q_source = initialize_radar(radar, hm_name)

    hkl1s, north_s = ub_matrix.convert(hkl1, 'rs'), ub_matrix.convert(north, 'rs')
    azimuth_offset = ft.azimuthS(north_s, hkl1s) + ssr

    main_plot.js_on_event(MouseMove, CustomJS(args=dict(div=div, en_button=en_button,
        ki_source=ki_source, kf_source=kf_source, q_source=q_source),
        code="""
            ki = {ki};
            hkl1 = {hkl1};
            sense = {sense};
            btn = en_button.active;
            ef_dict = {{0:2.5, 1:3.0, 2:3.5, 3:4.0, 4:4.5, 5:1000}};
            ef = ef_dict[btn];
            kf = etok(ef);
            ps_mat = math.matrix({ps_mat});
            pr_mat = math.matrix({pr_mat});
            p = math.matrix([cb_obj['x'], cb_obj['y'], 0]);
            s = math.multiply(ps_mat, p);
            r = math.multiply(pr_mat, p);
            div.text = 'Q: ' + String([r.get([0]).toFixed(2), r.get([1]).toFixed(2), r.get([2]).toFixed(2)]) + '<br>';
            
            try{{
                [alpha, beta, gamma] = find_triangle(math.norm(s), ki, kf);
                A4 = alpha * sense;
                A3 = -1 * (azimuthS(hkl1, s) + gamma * sense);
                div.text += 'A3: ' + toDeg(A3).toFixed(2) + ', A4: ' + toDeg(A4).toFixed(2);
                offset = {offset};
                ki_az = -A3 + offset + math.pi/2;
                kf_az = ki_az - math.pi + A4;
                ki_source['data']['x'][1] = ki * math.cos(ki_az);
                ki_source['data']['y'][1] = ki * math.sin(ki_az);
                ki_source.change.emit();
                
                kf_source['data']['x'][1] = kf * math.cos(kf_az);
                kf_source['data']['y'][1] = kf * math.sin(kf_az);
                kf_source.change.emit();
                
                q_source['data']['x'][1] = ki * math.cos(ki_az) + kf * math.cos(kf_az);
                q_source['data']['y'][1] = ki * math.sin(ki_az) + kf * math.sin(kf_az);
                q_source.change.emit();
                }}
            catch(except) {{
                div.text += "No channel selected/ Scattering triangle cannot close."
                }}   
            
    
            
        """.format(ps_mat=str(ub_matrix.get_matrix('ps').tolist()),
                   pr_mat=str(ub_matrix.get_matrix('pr').tolist()),
                   ki=ki, hkl1=list(hkl1s), north=north, offset=azimuth_offset, sense=sense
                   )))

    return radar


def initialize_radar(radar, name):
    HM_RED = {
        'HM-1': [[15, 65], [97, 145], [215, 263], [295, 345]],
        'HM-2': [[-10, 10], [170, 190]],
    }
    HM_YELLOW = {
        'HM-2': [[-40, 40], [140, 220]],
    }
    HM_GREEN = {
        'no': [[0, 360]]
    }


    try:
        radar.annular_wedge(x=0, y=0, inner_radius=0.3, outer_radius=2.5,
                            start_angle=[np.radians(x[0]) + np.pi/2 for x in HM_YELLOW[name]],
                            end_angle=[np.radians(x[1]) + np.pi/2 for x in HM_YELLOW[name]],
                            color='orange', alpha=0.4)
    except (IndexError, KeyError):
        pass

    try:
        radar.annular_wedge(x=0, y=0, inner_radius=0.3, outer_radius=2.7,
                            start_angle=[np.radians(x[0]) + np.pi/2 for x in HM_RED[name]],
                            end_angle=[np.radians(x[1]) + np.pi/2 for x in HM_RED[name]],
                            color='red', alpha=0.4)
    except (IndexError, KeyError):
        pass

    try:
        radar.annular_wedge(x=0, y=0, inner_radius=1, outer_radius=2,
                            start_angle=[np.radians(x[0]) + np.pi/2 for x in HM_GREEN[name]],
                            end_angle=[np.radians(x[1]) + np.pi/2 for x in HM_GREEN[name]],
                            color='green', alpha=0.4, line_alpha=0)
    except (IndexError, KeyError):
        pass

    ki_source = ColumnDataSource(dict(x=[0, -1], y=[0, 1]))
    kf_source = ColumnDataSource(dict(x=[0, 1], y=[0, 1]))
    q_source = ColumnDataSource(dict(x=[0, 0], y=[0, 1.41]))

    radar.line('x', 'y', color='blue', source=ki_source, line_width=2, legend="ki")
    radar.line('x', 'y', color='red', source=kf_source, line_width=2, legend="kf")
    radar.line('x', 'y', color='green', source=q_source, line_width=2, legend="Q")

    return ki_source, kf_source, q_source


def handler500(request):
    return render(request, "500.html", {})
