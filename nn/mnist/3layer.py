from mnist import mnist_loader as mnist, network

training, validation, test = mnist.load_data_wrapper()
network30 = network.Network([784, 30, 10])

network30.SGD(training, 20, 10, 3, validation)
