export DNNLROOT=/home/zinuo/Downloads/onednn/install

g++ -std=c++11 -I${DNNLROOT}/include -Imisc/include -Ilayers/include -L${DNNLROOT}/lib \
    layers/*/*.h layers/*/*.cpp misc/*/*.h misc/*/*.hpp misc/*/*.cpp \
    onednn_training_mnist.cpp -ldnnl -g

./a.out cpu config.json
