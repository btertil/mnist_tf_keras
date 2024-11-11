# Moje MNIST z TF Keras, MLFlow i Callbacks (TensorBoard + LearnigRateScheduler + EarlyStopping)

<br>
Moje eksperymenty, podrasowana przeze mnie sieć konwolucyjna dla MNIST. 
Chcę dostać na szybko dobry model (nie do końca dobra praktyka wg Andrew Ng) ale to też eksploracje szczegołnie z wykopywaniem .


### Callbacks used:
* LearnigRateScheduler + custom function dla schedulera
* Early stopping (zwraca najlepszy model)
* TensorBoard z monitorowaniem co daje "wykopanie LR"

Dodatkowo TensorFlow v2 + pre-processing i wykresy.
Oryginalnie stworzone w VSCode na ubuntulaptop. Potem znacznie poprawiony (win + Pycharm)

### Komendy do MLFlow:
- serwer musi być uruchomiony w tym samym katalogu co skrypt
- $ mlflow ui --backend-store-uri sqlite:///mlflow.db

### Komendy do TensorBoard:
- serwer musi być uruchomiony w tym samym katalogu co skrypt
- musi być pusty / wyczyszczony katalog ./logs
- $ rm -Rf logs (jeśli istnieje)
- $ mkdir logs
- $ tensorboard --logdir ./logs/runs
- $ tensorboard --logdir ./logs/runs
