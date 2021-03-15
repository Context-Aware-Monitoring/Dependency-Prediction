import json
from datetime import datetime, timedelta
import pathlib
import pandas as pd
import networkx as nx
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
        pathes = list(pathToOperation.glob('*.json'))
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


def getLabel5(span):
    return span['host'] + '_' + span['operation'] + '_' + span['name'] + '_' + span['service'] + '_' + span['project']


def getLabel2(span):
    return span['host'] + '_' + span['name']


def createColumns(operations, nodeNames, metrics, traces, ds):
    # aggregation graphs (2 and 5 features) columns
    combinations = []
    for operation in operations:
        for trace in traces[operation]['graph']:
            spans = trace.nodes(data=True)
            for span in spans:
                if span[1]['isRoot']:
                    continue
                span_p = getLabel5(spans[list(trace.predecessors(span[0]))[0]])
                span_c = getLabel5(span[1])
                if span_c != span_p:
                    communication = span_p + '->' + span_c
                    if communication not in combinations:
                        combinations.append(communication)
                        ds[communication] = 0
                        ds[communication + '__duration'] = 0

                span_p = getLabel2(spans[list(trace.predecessors(span[0]))[0]])
                span_c = getLabel2(span[1])
                if span_c != span_p:
                    communication = span_p + '->' + span_c
                    if communication not in combinations:
                        combinations.append(communication)
                        ds[communication] = 0
                        ds[communication + '__duration'] = 0

    # counting and duration (host_name) columns
    combinations = []
    for operation in operations:
        for trace in traces[operation]['graph']:
            spans = trace.nodes(data=True)
            for span in spans:
                combination = getLabel2(span[1])
                if combination not in combinations:
                    combinations.append(combination)
                    ds[combination] = 0
                    ds[combination + '__duration'] = 0

    # communications between hosts (for plots)
    for i in range(len(nodeNames)):
        for j in range(i + 1, len(nodeNames)):
            ds[nodeNames[i] + '_' + nodeNames[j] + '__total_number'] = 0
            ds[nodeNames[i] + '_' + nodeNames[j] + '__total_duration'] = 0

    # total counting and duration per host columns (for plots)
    for host in nodeNames:
        ds[host + '__total_number'] = 0
        ds[host + '__total_duration'] = 0

    # MI columns
    for k in range(len(metrics)):
        for l in range(k, len(metrics)):
            for i in range(len(nodeNames)):
                t = (0, 1)[k == l]
                for j in range(i + t, len(nodeNames)):
                    ds['MI' + '_' + nodeNames[i] + '_' + nodeNames[j] + '_' + metrics[k] + '_' + metrics[l]] = 0.0

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
                        ds.at[f, 'MI' + '_' + nodeNames[i] + '_' + nodeNames[j] + '_' + metrics[p] + '_' + metrics[
                            l]] = mi
        k += n_s
        f += 1

    return ds


def collectData(operations, nodeNames, windows, traces, ds):
    # find index of window
    def findIndex(time):
        for i in range(len(windows)):
            if (time <= windows[i][1]) and (time >= windows[i][0]):
                return i
        return -1

    # get communication label with f function
    def getCommunication(span, trace, f):
        if span[1]['isRoot']:
            return False, ''
        spans = trace.nodes(data=True)
        span_p = f(spans[list(trace.predecessors(span[0]))[0]])
        span_c = f(span[1])
        if span_c != span_p:
            communication = span_p + '->' + span_c
            return True, communication
        return False, ''

    # get communication label between hosts
    def getCommunicationHosts(span, trace):
        if span[1]['isRoot']:
            return False, ''
        spans = trace.nodes(data=True)
        span_p = spans[list(trace.predecessors(span[0]))[0]]['host']
        span_c = span[1]['host']
        if span_c != span_p:
            if nodeNames.index(span_p) < nodeNames.index(span_c):
                communication = span_p + '_' + span_c
                return True, communication
            communication = span_c + '_' + span_p
            return True, communication
        return False, ''

    def increaseNumberAndDuration(row, column, duration, totalLabel):
        if totalLabel:
            ds.at[row, column + '__total_duration'] += duration
            ds.at[row, column + '__total_number'] += 1
        else:
            ds.at[row, column + '__duration'] += duration
            ds.at[row, column] += 1

    def fillWindow(i_s, i_e, span, column, totalLabel=False):
        if (i_s == i_e):
            increaseNumberAndDuration(i_s, column, (span['endTimestamp'] - span['startTimestamp']).microseconds,
                                      totalLabel)
        else:
            if (i_e == -1):
                increaseNumberAndDuration(i_s, column, (windows[i_s][1] - span['startTimestamp']).microseconds,
                                          totalLabel)
            else:
                increaseNumberAndDuration(i_s, column, (windows[i_s][1] - span['startTimestamp']).microseconds,
                                          totalLabel)
                increaseNumberAndDuration(i_e, column, (windows[i_e][1] - span['endTimestamp']).microseconds,
                                          totalLabel)
                for i in range(1, i_e - i_s):
                    increaseNumberAndDuration(i_s + i, column,
                                              (windows[i_s + 1][1] - windows[i_s + 1][0]).microseconds,
                                              totalLabel)

    for operation in operations:
        for trace in traces[operation]['graph']:
            spans = trace.nodes(data=True)
            for span in spans:
                ts = span[1]['startTimestamp'].replace(microsecond=0)
                i_s, i_e = findIndex(ts), -1
                if span[1]['endTimestamp'] != 'Null':
                    te = span[1]['endTimestamp'].replace(microsecond=0)
                    i_e = findIndex(te)

                # graph with 5 features
                isCommunication, communication = getCommunication(span, trace, getLabel5)
                if isCommunication:
                    fillWindow(i_s, i_e, span[1], communication)

                # graph with 2 features (host_name)
                isCommunication, communication = getCommunication(span, trace, getLabel2)
                if isCommunication:
                    fillWindow(i_s, i_e, span[1], communication)

                # counting and duration per host_name
                fillWindow(i_s, i_e, span[1], getLabel2(span[1]))

                # communications between hosts
                isCommunication, communication = getCommunicationHosts(span, trace)
                if isCommunication:
                    fillWindow(i_s, i_e, span[1], communication, True)

                # total counting and duration per host
                fillWindow(i_s, i_e, span[1], span[1]['host'], True)

    return ds


def saveData(overlapping, pathToTraces, ds):
    title = 'non'
    if overlapping != 0:
        title = str(int(overlapping * 100)) + '%'

    ds.to_csv(pathToTraces / ('all_features_from_spans_' + title + '_overlapping.csv'), index=False)


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

    windows, ds = createWindowing(windowSize, overlapping)
    ds = createColumns(operations, nodeNames, metrics, traces, ds)
    ds = computeMI(windowSize, overlapping, nodeNames, metrics, windows, nodes, ds)
    ds = collectData(operations, nodeNames, windows, traces, ds)

    saveData(overlapping, pathToTraces, ds)


main()
