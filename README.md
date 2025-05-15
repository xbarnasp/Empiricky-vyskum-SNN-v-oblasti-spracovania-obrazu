# Empirický výskum potenciálu samonormalizačných neurónových sietí v oblasti spracovania obrazu

Tento repozitár slúži na replikáciu experimentov z diplomovej práce zameranej na testovanie SNN v oblasti spracovania obrazových signálov. V jednotlivých zložkách sa nachádzajú kódy na replikáciu testov. Použité architektúry sú VGG, AlexNet a nami vytvorený DPModel. Každý kód je rozdelený podľa architektúry a následne podľa datasetu.

Na spustenie jednotlivých testov je potrebné stiahnuť a upraviť datasety z uvedených odkazov (alebo využiť knižnicu Torch):

CIFAR-10 a CIFAR-100: https://www.cs.toronto.edu/~kriz/cifar.html 

MNIST: https://www.kaggle.com/datasets/hojjatk/mnist-dataset 

Mini-ImageNet: https://www.kaggle.com/datasets/arjunashok33/miniimagenet

Food 101: https://www.kaggle.com/datasets/dansbecker/food-101

Flowers 102: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/

Následne je potrebné datasety upraviť podľa názvov súborov. Hlavná zložka je definovaná na začiatku každého kódu, v nej sa nachádzajú dáta rozdelené v zložkách test, train a validation, v ktorých sú ďalšie súbory jednotlivých tried a v nich obrázky. Nakoniec ich treba usporiadať podľa štruktúry definovanej v diplomovej práci.

Príklad datasetu:

📁 hlavna_zlozka/
├── 📁 train/
│   ├── 📁 trieda_1/
│   │   ├── obrazok1.jpg
│   │   └── obrazok2.jpg
│   ├── 📁 trieda_2/
│   │   └── ...
│   └── ...
├── 📁 validation/
│   ├── 📁 trieda_1/
│   │   └── ...
│   └── ...
├── 📁 test/
│   ├── 📁 trieda_1/
│   │   └── ...
│   └── ...
└── ...
