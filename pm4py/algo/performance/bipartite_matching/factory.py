from .versions import classic
from pm4py.objects.log.log import EventLog
from .util import shared_variables as sv
from .util import helper
import sys


def apply(log, start, end, classifier="concept:name", parameters=None):
    """
    Measure the duration of a log

    Parameters
    ------------
    log
        Event log
    start
        Start activities
    end
        End activities
    classifier
        Event classifier (activity name by default)
    parameters
        Parameters of the algorithm
    Returns
    ------------
    result
        {Case ID: (edges selected, measurements)}
    variants
        {trace: cases}
    """

    throughput_stat = {}
    measurement_amount_stat = {}
    throughput_data = []
    amount_data = []
    pattern_summary = []


    if isinstance(log, EventLog):
        edges_dictio, measures_dictio = classic.apply(log, start, end, classifier)
        throughput_stat, measurement_amount_stat, throughput_data, amount_data = helper.get_summary(measures_dictio, parameters['case_performance_attribute'])
        if len(amount_data) != 0:
            pattern_summary = helper.get_variants_for_plot(sv.filtered_log, edges_dictio)

    return throughput_stat, measurement_amount_stat, throughput_data, amount_data, pattern_summary

