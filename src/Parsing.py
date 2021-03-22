import json
from datetime import datetime, timedelta
import pathlib
import pandas as pd
import networkx as nx
from statistics import median, mean
from itertools import combinations
from minepy import MINE
import warnings

warnings.simplefilter("ignore", UserWarning)
from sklearn.metrics import mutual_info_score


def loadTraces(pathToTraces):
    def loadJson(link):
        with open(link) as f:
            data = json.load(f)
        return data

    operations = sorted(list(map(lambda x: x.name, list(pathToTraces.glob('**'))[1:])))

    traces = {}
    for operation in operations:
        pathToOperation = pathToTraces / operation
        pathes = sorted(list(pathToOperation.glob('*.json')))
        traces[operation] = {}
        traces[operation]['id'] = list(map(lambda x: x.name[:x.name.find('.json')], pathes))
        traces[operation]['data'] = list(map(lambda x: loadJson(x), pathes))

    return operations, traces


def loadMetrics(pathToData):
    pathToMetrics = pathToData / 'fixed_metrics'
    nodeNames = sorted(list(map(lambda x: x.name[:x.name.find('_')], list(pathToMetrics.glob('*.csv')))))

    nodes = {}
    for name in nodeNames:
        nodes[name] = {}
        nodes[name]['data'] = pd.read_csv(pathToMetrics / (name + '_metrics.csv'))

    for name in nodeNames:
        nodes[name]['data']['now'] = nodes[name]['data']['now'].map(
            lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S CEST'))

    metrics = list(nodes[nodeNames[0]]['data'].keys())
    metrics.remove('now')
    metrics.remove('load.cpucore')  # always == 8
    metrics = sorted(metrics)

    return nodeNames, metrics, nodes


def parseTrace(operation, df, graph):
    G = graph
    for item in df['children']:
        trace = {}
        trace['operation'] = operation
        trace['host'] = item.get('info').get('host')
        trace['name'] = item.get('info').get('name')
        trace['service'] = item.get('info').get('service')
        trace['project'] = item.get('info').get('project')
        trace['startTimestamp'] = datetime.strptime(
            item.get('info').get('meta.raw_payload.' + item.get('info').get('name') + '-start').get('timestamp'),
            '%Y-%m-%dT%H:%M:%S.%f')
        endTimestamp = item.get('info').get('meta.raw_payload.' + item.get('info').get('name') + '-stop',
                                            {'timestamp': 'Null'}).get('timestamp')
        if endTimestamp != 'Null':
            trace['endTimestamp'] = datetime.strptime(endTimestamp, '%Y-%m-%dT%H:%M:%S.%f')
            trace['duration'] = trace['endTimestamp'] - trace['startTimestamp']
        else:
            trace['endTimestamp'] = 'Null'
            trace['duration'] = 'Null'
        trace['trace_id'] = item.get('trace_id')
        trace['parent_id'] = item.get('parent_id')
        trace['base_id'] = item.get('info').get('meta.raw_payload.' + item['info']['name'] + '-start').get('base_id')
        trace['isRoot'] = trace['parent_id'] == trace['base_id']

        G.add_nodes_from([(trace['trace_id'], trace)])
        if not (trace['isRoot']):
            G.add_edge(trace['parent_id'], trace['trace_id'])

        if len(item['children']) != 0:
            G = parseTrace(operation, item, G)

    return G


# fix non-endTimestamp problem
def fixTraces(operations, traces):
    for operation in operations:
        for trace in traces[operation]['graph']:
            spans = trace.nodes(data=True)
            for span in spans:
                if span[1]['endTimestamp'] == 'Null':
                    children = list(nx.descendants(trace, span[0]))
                    if children == []:
                        continue
                    endTimestamp = span[1]['startTimestamp']
                    for child in children:
                        time = spans[child]['endTimestamp']
                        if time != 'Null':
                            endTimestamp = max(endTimestamp, time)
                    span[1]['endTimestamp'] = endTimestamp
                    span[1]['duration'] = span[1]['endTimestamp'] - span[1]['startTimestamp']

    return traces


def createWindowing(windowSize, overlapping):
    n_s = int(windowSize * (1 - overlapping))
    windows = []

    timeStart = datetime.strptime('2019-11-19 17:38:38', '%Y-%m-%d %H:%M:%S')
    timeEnd = datetime.strptime('2019-11-20 01:30:00', '%Y-%m-%d %H:%M:%S')

    time = timeStart
    while time + timedelta(seconds=windowSize) <= timeEnd:
        windows.append([time + timedelta(seconds=1), time + timedelta(seconds=windowSize)])
        time += timedelta(seconds=n_s)

    ds = pd.DataFrame({'window': windows})

    return windows, ds


# create label from features
def combineLabel(features, combination):
    label = features[0]
    for i in combination:
        label = label + '_' + features[i]
    return label


def createModes():
    features_p = ['host_1', 'operation_1', 'name_1', 'service_1', 'project_1']
    features = ['host_2', 'operation_2', 'name_2', 'service_2', 'project_2']
    featuresNonCommunication = ['host', 'operation', 'name', 'service', 'project']
    columns = []
    columns.append(featuresNonCommunication[0])
    columns.append(features_p[0] + '->' + features[0])
    for l in range(1, len(features)):
        for combination in combinations(list(range(1, len(features))), l):
            label_r = combineLabel(featuresNonCommunication, list(combination))
            columns.append(label_r)
            label_r = combineLabel(features, list(combination))
            if len(features_p) != 0:
                label_l = combineLabel(features_p, list(combination))
                columns.append(label_l + '->' + label_r)

    modes = {}
    for i in range(len(columns)):
        k = (i // 2 + 1, i // 2 + 17)[i % 2]
        modes[k] = {'name': columns[i], 'combinations': []}

    return modes


def createColumns(pathToTraces, operations, nodeNames, metrics, traces, modes, ds):
    def addCombinationToMode(i, label):
        k = (i // 2 + 1, i // 2 + 17)[i % 2]
        if label not in modes.get(k).get('combinations'):
            modes[k]['combinations'].append(label)
            modes[k]['combinations'].append(label + '__duration')

    def addCombintaionToColumns(label):
        if label not in list(ds.keys()):
            ds[label] = 0
            ds[label + '__duration'] = 0

    # get all possible combinations of two types of aggregation
    for operation in operations:
        for trace in traces[operation]['graph']:
            spans = trace.nodes(data=True)
            for span in spans:
                i = 0
                features_p = []
                if not (span[1]['isRoot']):
                    span_p = spans[list(trace.predecessors(span[0]))[0]]
                    features_p = [span_p['host'], span_p['operation'], span_p['name'], span_p['service'],
                                  span_p['project']]
                features = [span[1]['host'], span[1]['operation'], span[1]['name'], span[1]['service'],
                            span[1]['project']]
                addCombintaionToColumns(features[0])
                addCombinationToMode(i, features[0])
                i += 1
                if len(features_p) != 0:
                    addCombintaionToColumns(features_p[0] + '->' + features[0])
                    addCombinationToMode(i, features_p[0] + '->' + features[0])
                i += 1
                for l in range(1, len(features)):
                    for combination in combinations(list(range(1, len(features))), l):
                        label_r = combineLabel(features, list(combination))
                        addCombintaionToColumns(label_r)
                        addCombinationToMode(i, label_r)
                        i += 1
                        if len(features_p) != 0:
                            label_l = combineLabel(features_p, list(combination))
                            addCombintaionToColumns(label_l + '->' + label_r)
                            addCombinationToMode(i, label_l + '->' + label_r)
                        i += 1

    # save JSON of modes
    with open(pathToTraces / 'modes.json', 'w') as f:
        json.dump(modes, f)

    # Metrics columns
    for metric in metrics:
        for name in nodeNames:
            ds[name + '_' + metric] = 0.0

    # MI columns
    for p in range(len(metrics)):
        for l in range(p, len(metrics)):
            for i in range(len(nodeNames)):
                t = (0, 1)[p == l]
                for j in range(i + t, len(nodeNames)):
                    ds['MI' + '_' + nodeNames[i] + '_' + metrics[p] + '_' + nodeNames[j] + '_' + metrics[l]] = 0.0

    return ds


def computeMedianOfMetric(windowSize, overlapping, nodeNames, metrics, windows, nodes, ds):
    n_s = int(windowSize * (1 - overlapping))
    f = 0
    k = 0
    while f < len(windows):
        for metric in metrics:
            for name in nodeNames:
                m = median(list(nodes[name]['data'][metric])[k:k + windowSize])
                # m = mean(list(nodes[name]['data'][metric])[k:k + windowSize])
                ds.at[f, name + '_' + metric] = m
        k += n_s
        f += 1

    return ds


def computeMI(windowSize, overlapping, nodeNames, metrics, windows, nodes, ds):
    n_s = int(windowSize * (1 - overlapping))
    f = 0
    k = 0
    while f < len(windows):
        for p in range(len(metrics)):
            for l in range(p, len(metrics)):
                for i in range(len(nodeNames)):
                    t = (0, 1)[p == l]
                    for j in range(i + t, len(nodeNames)):
                        mi = mutual_info_score(list(nodes[nodeNames[i]]['data'][metrics[p]])[k:k + windowSize],
                                               list(nodes[nodeNames[j]]['data'][metrics[l]])[k:k + windowSize])
                        # mine = MINE(alpha=0.6, c=15, est="mic_approx")
                        # mine.compute_score(list(nodes[nodeNames[i]]['data'][metrics[p]])[k:k + windowSize],
                        #                   list(nodes[nodeNames[j]]['data'][metrics[l]])[k:k + windowSize])
                        # mi = mine.mic()
                        ds.at[f, 'MI' + '_' + nodeNames[i] + '_' + metrics[p] + '_' + nodeNames[j] + '_' + metrics[
                            l]] = mi
        k += n_s
        f += 1

    return ds


def collectData(operations, windows, traces, ds):
    # find index of window
    def findIndex(time):
        for i in range(len(windows)):
            if windows[i][0] <= time < (windows[i][1] + timedelta(seconds=1)):
                return i
        return -1

    def increaseNumberAndDuration(row, column, duration):
        ds.at[row, column + '__duration'] += duration
        ds.at[row, column] += 1

    def fillWindow(i_s, i_e, span, column):
        if (i_s == i_e):
            increaseNumberAndDuration(i_s, column,
                                      (span['endTimestamp'] - span['startTimestamp']) // timedelta(microseconds=1))
        else:
            if (i_e == -1):
                increaseNumberAndDuration(i_s, column, (
                        windows[i_s][1] + timedelta(seconds=1) - span['startTimestamp']) // timedelta(
                    microseconds=1))
            else:
                increaseNumberAndDuration(i_s, column, (
                        windows[i_s][1] + timedelta(seconds=1) - span['startTimestamp']) // timedelta(
                    microseconds=1))
                increaseNumberAndDuration(i_e, column,
                                          (span['endTimestamp'] - windows[i_e][0]) // timedelta(microseconds=1))
                for i in range(1, i_e - i_s):
                    increaseNumberAndDuration(i_s + i, column, (
                            windows[i_s + i][1] + timedelta(seconds=1) - windows[i_s + i][0]) // timedelta(
                        microseconds=1))

    for operation in operations:
        for trace in traces[operation]['graph']:
            spans = trace.nodes(data=True)
            for span in spans:
                i_s, i_e = findIndex(span[1]['startTimestamp']), -1
                if span[1]['endTimestamp'] != 'Null':
                    i_e = findIndex(span[1]['endTimestamp'])
                features = [span[1]['host'], span[1]['operation'], span[1]['name'], span[1]['service'],
                            span[1]['project']]
                fillWindow(i_s, i_e, span[1], features[0])
                features_p = []
                if not (span[1]['isRoot']):
                    span_p = spans[list(trace.predecessors(span[0]))[0]]
                    features_p = [span_p['host'], span_p['operation'], span_p['name'], span_p['service'],
                                  span_p['project']]
                if len(features_p) != 0:
                    fillWindow(i_s, i_e, span[1], features_p[0] + '->' + features[0])
                for l in range(1, len(features)):
                    for combination in combinations(list(range(1, len(features))), l):
                        label_r = combineLabel(features, list(combination))
                        fillWindow(i_s, i_e, span[1], label_r)
                        if len(features_p) != 0:
                            label_l = combineLabel(features_p, list(combination))
                            fillWindow(i_s, i_e, span[1], label_l + '->' + label_r)

    return ds


def saveData(overlapping, pathToTraces, ds):
    title = ('non', str(int(overlapping * 100)) + '%')[overlapping != 0]

    ds.to_csv(pathToTraces / ('parsed_traces_with_' + title + '_overlapping.csv'), index=False)


def main(windowSize=60, overlapping=0):
    assert 0 < windowSize < 28282
    assert 0 <= overlapping < 1

    relativePathToData = 'data/sequential_data'
    pathToData = pathlib.Path().absolute().parent / relativePathToData
    pathToTraces = pathToData / 'traces'

    operations, traces = loadTraces(pathToTraces)
    nodeNames, metrics, nodes = loadMetrics(pathToData)

    for operation in operations:
        traces[operation]['graph'] = list(
            map(lambda x: parseTrace(operation, x, nx.DiGraph()), traces[operation]['data']))

    traces = fixTraces(operations, traces)
    windows, ds = createWindowing(windowSize, overlapping)
    modes = createModes()
    ds = createColumns(pathToTraces, operations, nodeNames, metrics, traces, modes, ds)
    ds = computeMedianOfMetric(windowSize, overlapping, nodeNames, metrics, windows, nodes, ds)
    ds = computeMI(windowSize, overlapping, nodeNames, metrics, windows, nodes, ds)
    ds = collectData(operations, windows, traces, ds)

    saveData(overlapping, pathToTraces, ds)


main()
