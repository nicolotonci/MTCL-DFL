
#include <torch/torch.h>
#include "mtcl.hpp"
#include "common.hpp"
#include "fl/fedavg.hpp"
#include "fl/traintest.hpp"

int main(int argc, char *argv[]) {
    int num_workers = 2;                // Number of workers
    int train_batchsize = 64;           // Train batch size
    int test_batchsize = 1000;          // Test batch size
    int train_epochs = 2;               // Number of training epochs at workers in each round
    int rounds = 10;                     // Number of training rounds
    std::string data_path = "../../data";  // Patch to the MNISt data files (absolute or with respect to build directory)
    int forcecpu = 0;
    int nt = 4; // Number of threads per process

    if (argc >= 2) {
        if (strcmp(argv[1], "-h") == 0) {
            std::cout << "Usage: master [forcecpu=0/1] [rounds=10] [epochs/round=2] [data_path]\n";
            exit(0);
        } else
            forcecpu = atoi(argv[1]);
    }
    if (argc >= 3)
        rounds = atoi(argv[2]);
    if (argc >= 4)
        train_epochs = atoi(argv[3]);
    if (argc >= 5)
        data_path = argv[4];
    if (argc >= 6)
        num_workers = atoi(argv[5]);
    std::cout << "Training on " << num_workers << " workers." << std::endl;

    // Use GPU, if available
    torch::DeviceType device_type;
    if (torch::cuda::is_available() && !forcecpu) {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    } else {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    torch::cuda::manual_seed_all(42);

    // Get train and test data
    auto train_dataset = torch::data::datasets::MNIST(data_path)
            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
            .map(torch::data::transforms::Stack<>());

    auto test_dataset = torch::data::datasets::MNIST(data_path, torch::data::datasets::MNIST::Mode::kTest)
            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
            .map(torch::data::transforms::Stack<>());

    auto test_data_loader = torch::data::make_data_loader(test_dataset, test_batchsize);

    Manager::init("Master", "config.json");

    auto bcast = Manager::createTeam("Master:Worker1:Worker2", "Master", BROADCAST);
    auto gather = Manager::createTeam("Master:Worker1:Worker2", "Master", GATHER);

    auto model = new Net;
    model->to(device);
    FedAvg<Net*> aggregator(model);
    size_t model_size = serializeModel(*model).size();
    bcast.sendrecv(&model_size, sizeof(size_t), nullptr, 0);
    char* buff = new char[model_size * gather.size()];
    for (int round = 0;  round < rounds; ++round){
        aggregator.new_round();
        auto serialized_model = serializeModel(*model);
        assert(serialized_model.size() == model_size);
        bcast.sendrecv(serialized_model.c_str(), model_size, nullptr, 0);

        // perfrom the gather collective communication
        gather.sendrecv(serialized_model.c_str(), model_size, buff, model_size);

        // skip my portion
        for(int i = 1; i < gather.size(); i++) {
            auto received_model = deserializeModel(buff+(i*model_size), model_size);
            aggregator.update_from(&received_model);
        }

        test(model, device, *test_data_loader);
    }
    delete [] buff;
    std::string eos("EOS", model_size);
    bcast.sendrecv(eos.c_str(), model_size, nullptr, 0);
    bcast.close();
    Manager::finalize(true);
    return 0;
}