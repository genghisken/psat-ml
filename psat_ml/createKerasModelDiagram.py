from keras.utils import plot_model
from kerasTensorflowClassifier import create_model, load_data
num_classes = 2
image_dim = 20

model = create_model(num_classes, image_dim)
model.load_weights('/tmp/hko_6nights_data_classifier.h5')

plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.eps')
