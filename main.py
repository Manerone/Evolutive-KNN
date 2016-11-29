from evolutive_knn import EvolutiveKNN
from dataset_loader import DatasetLoader
import json
import os
import sys

NUMBER_OF_EXECUTIONS = 30


def datasets():
    return os.listdir('./datasets/')


def print_percent(dataset, current, total):
    sys.stdout.write("\033[K")
    percent = int((i+1.0)*100/total)
    sys.stdout.write("\r" + dataset + ': ' + str(percent) + '%')
    sys.stdout.flush()


def best_accuracies(dataset, accuracies):
    data = {
        'accuracies': accuracies,
        'average': np.mean(accuracies),
        'standard deviation': np.std(accuracies)
    }
    with open('./results/' + dataset + '.json', 'w') as file:
        json.dump(data, file)

for dataset in datasets():
    loader = DatasetLoader('./datasets/' + dataset)
    best_accuracies = []
    for i in xrange(NUMBER_OF_EXECUTIONS):
        print_percent(dataset, i, NUMBER_OF_EXECUTIONS)
        classifier = EvolutiveKNN(loader.examples, loader.labels)
        classifier.train()
        best_accuracies.append(classifier.global_best.fitness)
    print ''
    save_results(dataset, best_accuracies)
