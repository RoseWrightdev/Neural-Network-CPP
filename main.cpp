#include "./NeuralNetwork.hpp"

typedef std::vector<RowVector*> data;
int main()
{
    NeuralNetwork n({2, 3, 1 }, 0.0004);
    data in_dat, out_dat;
    n.genData("test");
    n.ReadCSV("test-in", in_dat);
    n.ReadCSV("test-out", out_dat);
    n.train(in_dat, out_dat);
    return 0;
}