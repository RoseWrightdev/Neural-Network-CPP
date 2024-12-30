#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <vector>
#include <fstream>

typedef float Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;

class NeuralNetwork
{
public:
    /*
        The constructor initializes the neural network's topology, learning rate, neuron layers, cache layers,
        deltas, and weights. It handles the special cases for the last layer and bias neurons appropriately.

        note: The topology vector describes how many neurons we have in each layer, and the size of this vector
        is equal to a number of layers in the neural network.
    */

    // todo take instance of activation function class as a dependency, refactor
    NeuralNetwork(std::vector<uint> topology, Scalar learningRate = Scalar(0.005))
    {
        this->topology = topology;
        this->learningRate = learningRate;
        // init neuron layers
        for (uint i = 0; i < topology.size(); i++)
        {
            if (i == topology.size() - 1)
            {
                neuronLayers.push_back(new RowVector(topology[i]));
            }
            else
            {
                neuronLayers.push_back(new RowVector(topology[i] + 1));
                };
            

            // init deltas and cacheLayers to hold the
            cacheLayers.push_back(new RowVector(neuronLayers.size()));
            deltas.push_back(new RowVector(neuronLayers.size()));
            if (i != topology.size() - 1)
            {
                neuronLayers.back()->coeffRef(topology[i]) = 1.0;
                cacheLayers.back()->coeffRef(topology[i]) = 1.0;
            };

            // init weights
            if (i > 0)
            {
                if (i != topology.size() - 1)
                {
                    weights.push_back(new Matrix(topology[i - 1] + 1, topology[i] + 1));
                    weights.back()->setRandom();
                    weights.back()->col(topology[i]).setZero();
                    weights.back()->coeffRef(topology[i - 1], topology[i]) = 1.0;
                }
                else
                {
                    weights.push_back(new Matrix(topology[i - 1] + 1, topology[i]));
                    weights.back()->setRandom();
                };
            };
        };
    };

    void propagateForward(RowVector &input)
    {
        // block returns a part of the given vector or matrix
        // block takes 4 arguments : startRow, startCol, blockRows, blockCols
        neuronLayers.front()->block(0, 0, 1, neuronLayers.front()->size() - 1) = input;

        // apply the activation function
        for (uint i = 1; i < topology.size(); i++)
        {
            (*neuronLayers[i]) = (*neuronLayers[i - 1]) * (*weights[i - 1]);
            neuronLayers[i]->block(0, 0, 1, topology[i]).unaryExpr([this](Scalar x)
                                                                   { return activationFunction(x); });
        }
    }
    void propagateBackward(RowVector &output)
    {
        calcErrors(output);
        updateWeights();
    }

    void calcErrors(RowVector &output)
    {
    }
    void updateWeights()
    {
    }
    void train(std::vector<RowVector *> input_data, std::vector<RowVector *> output_data)
    {
        for (uint i = 0; i < input_data.size(); i++)
        {
            std::cout << "Input to neural network is : " << *input_data[i] << std::endl;
            propagateForward(*input_data[i]);
            std::cout << "Expected output is : " << *output_data[i] << std::endl;
            std::cout << "Output produced is : " << *neuronLayers.back() << std::endl;
            propagateBackward(*output_data[i]);
            std::cout << "MSE : " << std::sqrt((*deltas.back()).dot((*deltas.back())) / deltas.back()->size()) << std::endl;
        }
    }

    Scalar activationFunction(Scalar x)
    {
        return tanhf(x);
    }

    Scalar activationFunctionDerivative(Scalar x)
    {
        return 1 - tanhf(x) * tanhf(x);
    }

    Scalar activationFunctionSigmoid(Scalar x)
    {
        return 1 / (1 + std::exp(-x));
    }

    void ReadCSV(std::string filename, std::vector<RowVector*>& data)
    {
        data.clear();
        std::ifstream file(filename);
        std::string line, word;
        // determine number of columns in file
        getline(file, line, '\n');
        std::stringstream ss(line);
        std::vector<Scalar> parsed_vec;
        while (getline(ss, word, ',')) {
            parsed_vec.push_back(Scalar(std::stof(&word[0])));
        }
        uint cols = parsed_vec.size();
        data.push_back(new RowVector(cols));
        for (uint i = 0; i < cols; i++) {
            data.back()->coeffRef(1, i) = parsed_vec[i];
        }
    
        // read the file
        if (file.is_open()) {
            while (getline(file, line, '\n')) {
                std::stringstream ss(line);
                data.push_back(new RowVector(1, cols));
                uint i = 0;
                while (getline(ss, word, ',')) {
                    data.back()->coeffRef(i) = Scalar(std::stof(&word[0]));
                    i++;
                }
            }
        }
    }

    void genData(std::string filename)
    {
        std::ofstream file1(filename + "-in");
        std::ofstream file2(filename + "-out");
        for (uint r = 0; r < 1000; r++) {
            Scalar x = rand() / Scalar(RAND_MAX);
            Scalar y = rand() / Scalar(RAND_MAX);
            file1 << x << ", " << y << std::endl;
            file2 << 2 * x + 10 + y << std::endl;
        }
        file1.close();
        file2.close();
    }
    
    // storage
    std::vector<RowVector *> neuronLayers;
    std::vector<RowVector *> cacheLayers; // unactivated values of layers
    std::vector<RowVector *> deltas;      // neuron error contribution
    std::vector<Matrix *> weights;
    Scalar learningRate;
    std::vector<uint> topology;
};
