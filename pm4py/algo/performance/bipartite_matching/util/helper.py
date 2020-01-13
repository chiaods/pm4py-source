import re
import os
import numpy as np
from datetime import timedelta
from pm4py.objects.log.exporter.parquet import factory as parquet_exporter
from pm4py.objects.log.importer.parquet import factory as parquet_importer
from pm4py.objects.conversion.log import factory as log_conv_factory
from pm4py.objects.log.util import index_attribute
import copy
from pm4py.objects.log.util import xes as xes_util
from pm4py.objects.log import log as log_obj
from . import filtering
from . import shared_variables as sv
from pm4py.algo.filtering.log.variants import variants_filter
import sys


def select_attributes(log, classifier="concept:name"):
    """
    Select the case or event attributes of interests

    Parameters
    ------------
    log
        Event log
    classifier
        Event classifier (activity name by default)

    Returns
    ------------
    log
        Preprocessed event log
    """
    parquet_exporter.apply(log, "tmp.parquet")
    df = parquet_importer.apply("tmp.parquet",
                                parameters={"columns": ["case:concept:name", classifier, "time:timestamp"]})

    os.remove("tmp.parquet")

    return log_conv_factory.apply(df)


def index_event(log, attri="@@eventindex"):
    """
    Index event per trace in a log

    Parameters
    ------------
    log
        Event log

    Returns
    ------------
    log
        Indexed event log
    """
    for trace in log:
        for event_index, event in enumerate(trace, start=1):
            event[attri] = event_index

    return log


def compute_weight(node_start, node_end, function=None):
    """
    Compute the weight of an edge

    Parameters
    ------------
    node_start
        Event index of starting node
    node_end
        Event index of ending node
    function
        Customized function to compute the weight
    Returns
    ------------
    weight
        Number as weight
    """
    if function is None:
        return 1 / (node_end - node_start)
    else:
        return function(node_start, node_end)


def get_edge(variable):
    """
    Get the edge with the node of numerical index

    Parameters
    ------------
    variable
        Variables of LP represented in string
    Returns
    ------------
    tuple of nodes
        Edge represented in tuple of numbers
    """
    return (int(re.split('\(|\)|,', variable.name().replace(" ", ""))[1]),
            int(re.split('\(|\)|,', variable.name().replace(" ", ""))[-2]))


def compute_statistics(measures):
    """
    Compute the statistics

    Parameters
    ------------
    measures
        Measurements
    Returns
    ------------
    stats_timedelta
        Dictionary of statistical result
    """

    if_timedelta = False
    if isinstance(measures[0], timedelta):
        measures = [ts.total_seconds() for ts in measures if isinstance(ts, timedelta)]
        if_timedelta = True

    stats = dict()
    stats["minimum"] = round(np.min(np.array(measures)))
    stats["maximum"] = round(np.max(np.array(measures)))
    stats["median"] = round(np.median(np.array(measures)))
    stats["mean"] = round(np.mean(np.array(measures)))
    stats["std"] = round(np.std(np.array(measures)))

    if if_timedelta:
        stats = {k: timedelta(seconds=v) for (k, v) in stats.items()}  # if k != "freq"}

    stats["freq"] = len(np.array(measures))

    return stats


def get_variants_for_plot(log, edges):
    for case in log:
        case.attributes["last_index"] = len(case)

    log_filtered = filtering.filter_event(log, sv.start + sv.end)
    log_added = add_artificial_events(log_filtered)
    log_filtered_indexed = index_event(log_added, "@@eventindex_filtered")
    log_transform_attr = transform_attr(log_filtered_indexed, log)
    edges_plot = map_edge_to_filtered_log(log_transform_attr, edges)

    variants_by_trace = variants_filter.get_variants(log_transform_attr)
    variants_by_pattern = get_pattern(variants_by_trace, edges_plot)

    return get_pattern_stats_for_plot(variants_by_pattern)


def add_artificial_events(log):
    events_to_add = dict()

    for case in log:
        case_id = case.attributes["concept:name"]
        events_to_add[case_id] = []

        count = 0
        for event_count, event in enumerate(case, 1):
            if event == case[0] and case[0]["@@eventindex"] != 1:
                events_to_add[case_id].append(0)
                count += 1
            if event != case[-1]:
                if event["@@eventindex"] + 1 != case[event_count]["@@eventindex"]:
                    events_to_add[case_id].append(event_count + count)
                    count += 1
            if event == case[-1] and case[-1]["@@eventindex"] != case.attributes["last_index"]:
                events_to_add[case_id].append(event_count + count)
    for case in log:
        if case.attributes["concept:name"] in events_to_add.keys():
            for idx in sorted(events_to_add[case.attributes["concept:name"]]):
                case.insert(idx, log_obj.Event({xes_util.DEFAULT_NAME_KEY: "@@R", "@@eventindex": idx}))

    return log


def transform_attr(log, original_log):
    for case in log:
        case_org = [case_org for case_org in original_log if
                    case_org.attributes["concept:name"] == case.attributes["concept:name"]][0]
        for event_index, event in enumerate(case):
            event["original_act"] = [event["concept:name"]]
            if event["concept:name"] == "@@R":
                if event_index != 0:
                    last_index = case[event_index - 1]["@@eventindex"]
                else:
                    last_index = 0
                try:
                    next_index = case[event_index + 1]["@@eventindex"] - 1
                except IndexError:
                    next_index = case.attributes["last_index"]
                event["original_act"] = [event["concept:name"] for event in case_org[last_index:next_index]]
            if event["concept:name"] in sv.start:
                event["concept:name"] = "@@S"
            elif event["concept:name"] in sv.end:
                event["concept:name"] = "@@E"

    return log


def map_edge_to_filtered_log(log, edges):
    edges_plot = dict()

    for case in log:
        edges_plot[case.attributes["concept:name"]] = []
        for edge in edges[case.attributes["concept:name"]]:
            start_node = [event["@@eventindex_filtered"] for event in case if event["@@eventindex"] == edge[0]][0]
            end_node = [event["@@eventindex_filtered"] for event in case if event["@@eventindex"] == edge[1]][0]
            edges_plot[case.attributes["concept:name"]].append((start_node, end_node))

    return edges_plot


def get_pattern(variants, edges):
    pattern = dict()

    for trace, cases in variants.items():
        cases_id = [case.attributes["concept:name"] for case in cases]
        edges_selected = {case_id: edges_selected for case_id, edges_selected in edges.items() if case_id in cases_id}
        for case_id, edges_per_case in edges_selected.items():
            edges_str = str(edges_per_case)
            if (trace, edges_str) not in pattern.keys():
                pattern[(trace, edges_str)] = []
            pattern[(trace, edges_str)].append(
                [case for case in cases if case.attributes["concept:name"] == case_id][0])

    return pattern


def get_pattern_stats_for_plot(variants_by_pattern):
    pattern_summary = dict()

    pattern_id = 1
    for pattern, cases in variants_by_pattern.items():

        trace = [i for i in pattern[0].split(",")]
        edges = transform_str_to_list(pattern[1])
        edge_stat, pattern_stat = get_stat(edges, cases)
        cases_id = [case.attributes["concept:name"] for case in cases]
        traces_origin_act = []
        for case in cases:
            if case.attributes["concept:name"] in cases_id:
                traces_origin_act.append([event["original_act"] for event in case])
        actCount_per_dot = []
        for eid in range(len(trace)):
            act_dicto = dict()
            dot_traces = [trace_origin[eid] for trace_origin in traces_origin_act]
            dot = [event for events in dot_traces for event in events]
            for act in dot:
                if act not in act_dicto.keys():
                    act_dicto[act] = 0
                act_dicto[act] += 1
            actCount_per_dot.append(act_dicto)


        pattern_summary[pattern_id] = {"trace": trace, "edge_stat": edge_stat, "pattern_stat": pattern_stat,
                                       "cases": cases_id, "activities_per_dot": actCount_per_dot}
        pattern_id += 1

    return pattern_summary


def get_stat(edges, cases):
    edge_stat = dict()
    measure_pattern = []

    for edge in edges:
        measure = []
        for case in cases:
            start_time = case[edge[0] - 1]["time:timestamp"]
            end_time = case[edge[1] - 1]["time:timestamp"]
            measure.append(end_time - start_time)
        edge_stat[str(edge)] = compute_statistics(measure)
        measure_pattern = measure_pattern + measure

    pattern_stat = compute_statistics(measure_pattern)

    return edge_stat, pattern_stat


def transform_str_to_list(edges_str):
    edges = []
    for edge_str in edges_str.split("),"):
        for char in ["[", "(", ")", "]"]:
            edge_str = edge_str.replace(char, "")
        start_node = int(edge_str.split(",")[0])
        end_node = int(edge_str.split(",")[1])
        edges.append((start_node, end_node))
    return edges


def get_summary(measures, attr):
    """
    Summary of performance result

    Parameters
    ------------
    measures
        {Case ID: [measurements]}
    Returns
    ------------
    throughput_summary
        {Type of Statistics: throughput statistics}
    throughput_summary
        {Type of Statistics: amount of edges statistics}
    """
    throughput = []
    for measures_per_case in measures.values():
        throughput.append(compute_statistics(measures_per_case)[attr])
    throughput_summary = compute_statistics(throughput)

    amount = []
    for measures_per_case in measures.values():
        amount.append(len(measures_per_case))
    amount_summary = compute_statistics(amount)

    return throughput_summary, amount_summary, throughput, amount
