#include <torch/torch.h>
#include <iostream>

struct Net : torch::nn::Module {
    Net() {
        // Construct and register two Linear submodules.
        fc1 = register_module("fc1", torch::nn::Linear(784, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 32));
        fc3 = register_module("fc3", torch::nn::Linear(32, 10));
    }

    // Implement the Net's algorithm.
    torch::Tensor forward(torch::Tensor x) {
        // Use one of many tensor manipulation functions.
        x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
        x = torch::relu(fc2->forward(x));
        x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
        return x;
    }

    // Use one of many "standard library" modules.
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

std::string serializeModel(Net& n){
    std::ostringstream oss;
    torch::serialize::OutputArchive o;
    n.save(o); o.save_to(oss);
    return oss.str();
}

Net deserializeModel(char* buffer, size_t size){
    std::istringstream iss(std::string(buffer, size));
    torch::serialize::InputArchive i;
    i.load_from(iss);
    Net output;
    output.load(i);
    return std::move(output);
}