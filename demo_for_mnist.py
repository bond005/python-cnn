import numpy
from cnn.convolution_layer import ConvolutionLayer
from cnn.subsampling_layer import SubsamplingLayer
from cnn.output_layer import OutputLayer


def load_mnist_for_demo():
    """ Загрузить данные корпуса MNIST, чтобы продемонстрировать применение свёрточной нейронной сети для распознавания
    рукописных цифр от 0 до 9 (всего десять классов, 60 тыс. обучающих картинок и 10 тыс. тестовых картинок).
    :return Кортеж из двух элементов: обучающего и тестового множества. Каждое из множеств - как обучающее, так и
    тестовое - тоже задаётся в виде двухэлементого кортежа, первым элементом которого является множество векторов
    признаков входных объектов (двумерный numpy.ndarray-массив, число строк в котором равно числу входных объектов, а
    число столбцов равно количеству признаков объекта), а вторым элементом - множество желаемых выходных сигналов для
    каждого из соответствующих входных объектов (одномерный numpy.ndarray-массив, число элементов в котором равно числу
    входных объектов).
    """
    from sklearn.datasets import fetch_mldata  # импортируем специальный модуль из библиотеки ScikitLearn
    # загружаем MNIST из текущей директории или Интернета, если в текущей директории этих данных нет
    mnist = fetch_mldata('MNIST original', data_home='.')
    # получаем и нормируем вектора признаков для первых 60 тыс. картинок из MNIST, используемых для обучения
    # (матрица яркостей пикселей 28x28 -> одномерный вектор 784 признаков)
    X_train = mnist.data[0:60000].astype(numpy.float) / 255.0
    y_train = mnist.target[0:60000]  # получаем желаемые выходы (цифры от 0 до 9) для 60 тыс. обучающих картинок
    # получаем и нормируем вектора признаков для следующих 10 тыс. картинок из MNIST, используемых для тестирования
    # (матрица яркостей пикселей 28x28 -> одномерный вектор 784 признаков)
    X_test = mnist.data[60000:].astype(numpy.float) / 255.0
    y_test = mnist.target[60000:]  # получаем желаемые выходы (цифры от 0 до 9) для 10 тыс. тестовых картинок
    return ((X_train, y_train), (X_test, y_test))


if __name__ == '__main__':
    train_set, test_set = load_mnist_for_demo()