import os

from oli.math.math_utility import pretty_print_matrix
from oli.ml.Activation_functions import relu
from oli.ml.Activation_functions import relu_derivative
from oli.ml.Functions import softmax
from oli.ml.Loss import derivative_mean_squared_error_loss_categorical
from oli.ml.Loss import mean_squared_error_loss_categorical
from oli.ml.deep_learning.ANN import LinearLayer, NeuralNetwork
from oli.ml.deep_learning.mnist.Data import read_image_file, read_label_file, MNISTDataset

# train_img_path = "train-images-idx3-ubyte.gz"
# train_labels_path = "train-labels-idx1-ubyte.gz"
# test_img_path = "t10k-images-idx3-ubyte.gz"
# test_labels_path = "t10k-labels-idx1-ubyte.gz"
# def download_file(url, file_name):
#     if os.path.exists(file_name):
#         print(f"File {file_name} already exists.")
#         return True
#     with open(file_name, "wb") as file:
#             response = get(url)
#             file.write(response.content)
# download_file("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", train_img_path)
# download_file("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", train_labels_path)
# download_file("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", test_img_path)
# download_file("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", test_labels_path)

dir_path = os.getcwd()
print(dir_path)
print(os.listdir("../../../../"))
print(os.listdir("../../../../data/mnist/"))

train_images = read_image_file("../../../../data/mnist/train-images-idx3-ubyte")
train_labels = read_label_file("../../../../data/mnist/train-labels-idx1-ubyte")
train_set: MNISTDataset = MNISTDataset(train_images, train_labels)

test_images = read_image_file("../../../../data/mnist/t10k-images-idx3-ubyte")
test_labels = read_label_file("../../../../data/mnist/t10k-labels-idx1-ubyte")
test_set: MNISTDataset = MNISTDataset(test_images, test_labels)

layer = LinearLayer(
    neurons=4,
    inputs=3,
    activation_function=relu,
    derivative_activation_function=relu_derivative,
    derivative_cost_function=derivative_mean_squared_error_loss_categorical,
    test_mode=True
)
result = layer.forward([1, 2, 3])
assert result == [3.5, 3.5, 3.5, 3.5]
pretty_print_matrix(result)

loss = mean_squared_error_loss_categorical(result, 2)
assert loss == 10.7101
print(loss)

test_mode = False
nn = NeuralNetwork(
    LinearLayer(
        neurons=256,
        inputs=784,
        activation_function=relu,
        derivative_activation_function=relu_derivative,
        derivative_cost_function=derivative_mean_squared_error_loss_categorical,
        test_mode=test_mode
    ),
    LinearLayer(
        neurons=128,
        inputs=256,
        activation_function=relu,
        derivative_activation_function=relu_derivative,
        derivative_cost_function=derivative_mean_squared_error_loss_categorical,
        test_mode=test_mode
    ),
    LinearLayer(
        neurons=64,
        inputs=128,
        activation_function=relu,
        derivative_activation_function=relu_derivative,
        derivative_cost_function=derivative_mean_squared_error_loss_categorical,
        test_mode=test_mode
    ),
    LinearLayer(
        neurons=10,
        inputs=64,
        activation_function=relu,
        derivative_activation_function=relu_derivative,
        derivative_cost_function=derivative_mean_squared_error_loss_categorical,
        test_mode=test_mode
    )
)

prediction = softmax(
    nn.predict(test_set.images[0].get_linearized())
)
test_set.images[0].print()
pretty_print_matrix(softmax(prediction))

nn.train(
    X=train_set.get_linearized_images(),
    y=train_set.labels,
    epochs=20,
    batch_size=1000,
    learning_rate=0.001
)
