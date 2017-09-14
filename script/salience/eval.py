"""
Evaluate salience results, assuming the gold standard file and system file use exact ordering of entities.
"""

from script.salience import utils
import sys


class Evaluator:
    def __init__(self):
        self.tp = 0
        self.gold_size = 0
        self.sys_size = 0

    def add_prediction(self, gold_set, sys_set):
        for s in sys_set:
            if s in gold_set:
                self.tp += 1

        self.gold_size += len(gold_set)
        self.sys_size += len(sys_set)

    def get_f_measures(self):
        precision = 1.0 * self.tp / self.sys_size
        recall = 1.0 * self.tp / self.gold_size
        f1 = 2 * precision * recall / (precision + recall)
        return f1, precision, recall, self.tp, self.gold_size, self.sys_size


def get_salience_entities(mentions):
    salient_entities = set()
    for mention in mentions:
        if mention[1] == 1:
            salient_entities.add(mention[6])
    return salient_entities


def get_salience_mentions(mentions):
    salient_mentions = set()
    for mention in mentions:
        if mention[1] == 1:
            # A mention is defined by its mention index and its predicted KB.
            salient_mentions.add((mention[0], mention[6]))
    return salient_mentions


def evaluate_entity_salience(gold_path, sys_path):
    mention_evaluator = Evaluator()
    entity_evaluator = Evaluator()

    num_docs = 0
    for gold_data, sys_data in zip(utils.read_salience_file(gold_path), utils.read_salience_file(sys_path)):
        gold_entities = get_salience_entities(gold_data['entities'])
        sys_entities = get_salience_entities(sys_data['entities'])
        entity_evaluator.add_prediction(gold_entities, sys_entities)

        gold_mentions = get_salience_mentions(gold_data['entities'])
        sys_mentions = get_salience_mentions(sys_data['entities'])
        mention_evaluator.add_prediction(gold_mentions, sys_mentions)

        sys.stdout.write("\rProcessed %d files." % num_docs)
        num_docs += 1

    f_e, p_e, r_e, tp_e, gold_size_e, sys_size_e = entity_evaluator.get_f_measures()
    f_m, p_m, r_m, tp_m, gold_size_m, sys_size_m = mention_evaluator.get_f_measures()

    print("Mention based F measures: Precision: %.4f, Recall: %.4f, F1: %.4f. "
          "True Positive: %d, #Gold: %d, #System:%d." % (p_m, r_m, f_m, tp_m, gold_size_m, sys_size_m))
    print("Entity based F measures: Precision: %.4f, Recall: %.4f, F1: %.4f. "
          "True Positive: %d, #Gold: %d, #System:%d." % (p_e, r_e, f_e, tp_e, gold_size_e, sys_size_e))
