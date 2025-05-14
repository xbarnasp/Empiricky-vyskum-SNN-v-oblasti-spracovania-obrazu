# Empirický výskum potenciálu samonormalizačných neurónových sietí v oblasti spracovania obrazu

Tento repozitár slúži na replikáciu experimentov z diplomovej práce zameranej na testovanie SNN v oblasti spracovania obrazových signálov. V jednotlivých zložkách sa nachádzajú kódy na replikáciu testov. Použité architektúry sú VGG, AlexNet a nami vytvorený DPModel. Každý kód je rozdelený podľa architektúry a následne podľa datasetu.

Na spustenie jednotlivých testov je potrebné stiahnuť a upraviť datasety z uvedených odkazov (alebo využiť knižnicu Torch):

CIFAR-10 a CIFAR-100: https://www.cs.toronto.edu/~kriz/cifar.html 

MNIST: https://www.kaggle.com/datasets/hojjatk/mnist-dataset 

Mini-ImageNet: https://www.kaggle.com/datasets/arjunashok33/miniimagenet

Food 101: https://www.kaggle.com/datasets/dansbecker/food-101

Flowers 102: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/

Následne je potrebné datasety upraviť podľa názvov súborov v kóde a usporiadať ich podľa štruktúry definovanej v diplomovej práci.
