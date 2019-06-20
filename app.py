# all lines except: 78 to 129 and 137 to 189 by Luuk van Bilsen (1265202)
from __future__ import print_function

from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads, DATA

import jinja2
import pandas as pd
import json

from bokeh.embed import json_item
from bokeh.resources import CDN

import os, time
import numpy as np
import holoviews as hv
import networkx as nx

from holoviews import opts, dim
from pathlib import Path

# from bokeh.palettes import all_palettes
from bokeh.layouts import gridplot, row
from math import pi
from bokeh.models import LinearColorMapper, BasicTicker, ColorBar
from holoviews.element.graphs import layout_nodes
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.plotting import figure, show, output_file

from bokeh.palettes import plasma as Plasma256
import statistics
from holoviews import opts

hv.extension('bokeh')

app = Flask(__name__)

template_dir = os.path.join(os.path.dirname(__file__), 'templates')
jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))

# links to go from page to page
@app.route("/")
def index():
    return render_template("mainpage.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/settings")
def settings():
    return render_template("settings.html")


# extra stuff for uploading data-sets
datafiles = UploadSet('data', DATA)
app.config['UPLOADED_DATA_DEST'] = 'static/data'
configure_uploads(app, datafiles)

# upload path
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'any' in request.files:
        filename = datafiles.save(request.files['any'])
        return render_template('mainpage.html')
    return render_template('mainpage.html')


# generator for plot-page, also set some variables
page = jinja_env.get_template('vis.html')
holdFileName = 'nodeDummy'
file = f'static/data/{holdFileName}.csv'

# route to the page with visualizations and plot selection
@app.route("/visualization", methods=['GET', 'POST'])
def root():
    global holdFileName
    global file
    arr = os.listdir(app.config['UPLOADED_DATA_DEST'])
    if request.method == 'POST':
        holdFileName = request.form.get('set')
    file = f'static/data/{holdFileName}'
    return page.render(resources=CDN.render(), selected=holdFileName, arr=arr, holdFileName=holdFileName, file=file)


# Node-link diagram             by student1 (0000000)   &    student2 (0000000)
@app.route('/plot')
def plot():

    df_data = pd.read_csv(file, sep=';', header=0, index_col=False)

    p = []
    d = []
    e = []
    f = []
    hold = df_data.shape[0]

    # loop that sets values first in lists for columns
    l = 0
    i = 0
    while i < hold:
        # Fromnames + delete row of names once listed
        b = list(df_data.columns.values)
        del b[0]
        a1 = len(b) - (i - 1)
        a = list(a1 * (df_data.iloc[i, 0],))
        del a[:1]

        # Tonames + delete names that are already linked
        p = b
        del p[:(i)]

        # weights + delete weights that are already linked
        c = df_data.iloc[0:, (i + 1)].tolist()
        del c[:(i)]

        # remove people linked to themselves
        # for ele in c:
        #     if ele == 1:
        #         c.remove(ele)

        e = list(e + a)
        d = list(d + p)
        f = list(f + c)
        i += 1

    # df from which the plot will be made
    df_plot = pd.DataFrame(columns=['from', 'to', 'weight'])

    # puts said lists in columns
    df_plot['from'] = e
    df_plot['to'] = d
    df_plot['weight'] = f

    df_plot = df_plot.loc[df_plot['weight'] != 0.0]
    check = df_plot['from'] == df_plot['to']
    check2 = df_plot['weight'] == 1

    df_plot = df_plot[(check == False) & (check2 == False)]
    df_plot.reset_index()

    graph = hv.Graph(df_plot)
    graph.opts(width=900, height=900, show_frame=False, edge_color='weight',
               xaxis=None, yaxis=None, node_size=10, edge_line_width='weight')

    # layout of graph
    layout_nodes(graph, layout=nx.layout.fruchterman_reingold_layout, kwargs={'weight': 'weight'})

    holder = graph
    renderer = hv.renderer('bokeh')
    k = renderer.get_plot(holder).state

    k.plot_width = 700
    k.plot_height = 700

#  graph = from_networkx(G, nx.spring_layout, scale=2, center=(0, 0)

    return json.dumps(json_item(k))

# check viewbox html om div.

# Adjacency Matrix          by student1 (0000000)   &    student2 (0000000)
@app.route('/plot2')
def plot2():

    df_data = pd.read_csv(file, sep=';', header=0, index_col=False)

    x = []
    namesTo = []
    namesFrom = []
    namesWeight = []
    hold = df_data.shape[0]

    i = 0
    while i < hold:
        namesVal = list(df_data.columns.values)
        del namesVal[0]
        namesTimes = len(namesVal)
        namesHold = list(namesTimes * (df_data.iloc[i, 0],))

        x = namesVal

        c = df_data.iloc[:, (i + 1)].tolist()

        namesFrom = list(namesFrom + namesHold)
        namesTo = list(namesTo + x)
        namesWeight = list(namesWeight + c)

        i += 1

    df_plot = pd.DataFrame(columns=['from', 'to', 'weight'])

    df_plot['from'] = namesFrom
    df_plot['to'] = namesTo
    df_plot['weight'] = namesWeight

    data = dict(
        xname=namesFrom,
        yname=namesTo,
        weights=namesWeight
    )

    p = figure(x_axis_location="above",
               x_range=list(reversed(df_plot['from'].unique())), y_range=(df_plot['from'].unique()),
               tooltips=[('names', '@yname, @xname'), ('weight', '@weights'), ])

    p.plot_width = 700
    p.plot_height = 700
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "6pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = np.pi / 3

    p.rect('xname', 'yname', 0.9, 0.9, source=data, alpha='weights', line_color=None,
           hover_line_color='black', color='blue', hover_color='blue')

    return json.dumps(json_item(p))


@app.route('/plot3')
def plot3():

    df_data = pd.read_csv(file, sep=';', header=0, index_col=False)
    hold = df_data.shape[0]

    amount = []
    colorMeans = []

    i = 1
    while i <= hold:
        namesVal = df_data.iloc[:, i].tolist()
        j = 0
        length = len(namesVal)
        while j < length:
            if namesVal[j] == 0.0 or namesVal[j] == 1.0:
                namesVal.remove(namesVal[j])
                length = length - 1
                continue
            j += 1
        namesTimes = len(namesVal)
        amount.append(namesTimes)
        mean = statistics.mean(namesVal)  # fout op grotere sets 100
        colorMeans.append(mean)
        i += 1
    names = list(df_data.columns.values)
    del names[0]

    df_plot = pd.DataFrame(columns=['names', 'amount', 'colorMeans', 'colorCodes'])

    # puts said lists in columns
    df_plot['names'] = names
    df_plot['amount'] = amount
    df_plot['colorMeans'] = colorMeans

    count_row = df_plot.shape[0]
    df_plot['colorCodes'] = '#0C0786'

    df_plot = df_plot.sort_values('colorMeans', ascending=True)
    df_plot['colorCodes'] = Plasma256(count_row)

    ###
    # if count_row < 255:
    #    df_plot['colorCodes'] = Plasma256(count_row)
    # else:
    #    colorHold = np.ceil(count_row/255)
    #    yourList = Plasma256(255)
    #    longlist = list(np.repeat(yourList, colorHold))
    #    differ = len(longlist) - count_row
    #    del longlist[-differ:]
    #    df_plot['colorCodes'] = longlist
    ###

    df_plot = df_plot.reset_index()
    del df_plot['index']

    color = df_plot['colorCodes']
    names = df_plot['names'].values.tolist()

    TOOLS = [HoverTool()]

    sorted_links = sorted(names, key=lambda x: df_plot['colorMeans'][names.index(x)])
    q = figure(x_range=sorted_links, plot_height=650, plot_width=1400, tools=TOOLS, title="Barchart")

    q.vbar(x=names, top=amount, width=0.7, color=color)
    q.xgrid.grid_line_color = None
    q.y_range.start = 0
    q.xaxis.major_label_orientation = pi / 4

    color_mapper = LinearColorMapper(palette="Plasma256", low=df_plot['colorMeans'].min(),
                                     high=df_plot['colorMeans'].max())
    color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(),
                         location=(0, 0))

    q.add_layout(color_bar, 'left')

    return json.dumps(json_item(q))


@app.route('/plot4')
def plot4():

    df_data = pd.read_csv(file, sep=';', header=0, index_col=False)
    if not Path(df_data[:-4] + '.out.csv').is_file():
        df_data = pd.read_csv(df_data, sep=';', header=0, index_col=False)

        p = []
        d = []
        e = []
        f = []
        hold = df_data.shape[0]

        # loop that sets values first in lists for columns
        i = 0
        while i < hold:

            # Fromnames + delete row of names once listed
            b = list(df_data.columns.values)
            del b[0]
            a1 = len(b) - i
            a = list(a1 * (df_data.iloc[i, 0],))
            del a[:1]

            # Tonames + delete names that are already linked
            p = b
            del p[:(i + 1)]

            # weights + delete weights that are already linked
            c = df_data.iloc[:, 1].tolist()
            del c[:(i + 1)]

            # remove people linked to themselves
            for ele in c:
                if ele == 1:
                    c.remove(ele)

            e = list(e + a)
            d = list(d + p)
            f = list(f + c)

            i += 1

        # df from which the plot will be made
        df_plot = pd.DataFrame(columns=['from', 'to', 'weight'])

        # puts said lists in columns
        df_plot['from'] = e
        df_plot['to'] = d
        df_plot['weight'] = f

        # delete edges with weight 0
        df_plot = df_plot.loc[df_plot['weight'] != 0.0]

        df_plot.to_csv(df_data[:-4] + '.out.csv', sep='\t', encoding='utf-8', index=False)
    else:
        df_plot = pd.read_csv(df_data[:-4] + '.out.csv', sep='\t', encoding='utf-8', index_col=False)

    chord = hv.Chord(df_plot)
    chord.opts(
        opts.Chord(cmap='Category20', edge_cmap='Category20', edge_color=dim('weight').str(),
                   labels='name', node_color=dim('index').str()))

    holder2 = chord
    renderer = hv.renderer('bokeh')
    m = renderer.get_plot(holder2).state

    return json.dumps(json_item(m))


if __name__ == '__main__':
    app.run(debug=True)
