from evolutive_knn import EvolutiveKNN

classifier = EvolutiveKNN([[1, 2, 3, 1], [1, 4, 2, 8], [1, 1, 2, 1]], [0, 1, 0])
classifier.train()