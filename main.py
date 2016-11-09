from evolutive_knn import EvolutiveKNN
from banknote_loader import BanknoteLoader


banknote = BanknoteLoader('./datasets/banknote.data')
classifier = EvolutiveKNN(banknote.examples, banknote.labels)
classifier.train()
print classifier.global_best.k
print classifier.global_best.weights
print classifier.global_best.fitness