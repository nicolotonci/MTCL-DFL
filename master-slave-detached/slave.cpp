#include "mtcl.hpp"
#include "common.hpp"
#include <torch/torch.h>
#include "fl/traintest.hpp"

int main(int argc, char *argv[]) {

    int num_workers = 2;                // Number of workers
    int train_batchsize = 64;           // Train batch size
    int test_batchsize = 1000;          // Test batch size
    int train_epochs = 2;               // Number of training epochs at workers in each round
    int rounds = 10;                     // Number of training rounds
    std::string data_path = "../data";  // Patch to the MNISt data files (absolute or with respect to build directory)
    int forcecpu = 0;
    int worker_id;

    if (argc >= 2) {
        if (strcmp(argv[1], "-h") == 0) {
            std::cout << "Usage: slave [workerid] [forcecpu=0/1] [rounds=10] [epochs/round=2] [data_path]\n";
            exit(0);
        } else
            worker_id = atoi(argv[1]);
    }

    if (argc >= 3)
        forcecpu = atoi(argv[2]);
    if (argc >= 4)
        rounds = atoi(argv[3]);
    if (argc >= 5)
        train_epochs = atoi(argv[4]);
    if (argc >= 6)
        data_path = argv[5];
    if (argc >= 7)
        num_workers = atoi(argv[6]);

    std::cout << "Training on " << num_workers << " workers." << std::endl;

    // Use GPU, if available
    torch::DeviceType device_type;
    if (torch::cuda::is_available() && !forcecpu) {
        //std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    } else {
        //std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);
    //torch::set_num_threads(nt);

    torch::cuda::manual_seed_all(42);

    // Get train
    auto train_dataset = torch::data::datasets::MNIST(data_path)
            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
            .map(torch::data::transforms::Stack<>());

    Manager::init("Worker"+std::to_string(worker_id), "config.json");

    auto bcast = Manager::createTeam("Broker:Worker1:Worker2", "Broker", BROADCAST);
    auto gather = Manager::createTeam("Broker:Worker1:Worker2", "Broker", MTCL_GATHER);

    auto train_data_loader = torch::data::make_data_loader(train_dataset, torch::data::samplers::DistributedRandomSampler(
                                                                                                               train_dataset.size().value(),
                                                                                                               8,
                                                                                                               worker_id%8,
                                                                                                               false),
                                                                                                       train_batchsize);
    

    size_t modelSize;
    bcast.sendrecv(nullptr, 0, &modelSize, sizeof(size_t));
    char* buff = new char[modelSize];
    while (true){
        bcast.sendrecv(nullptr, 0, buff, modelSize);
        if (buff[0] == 'E' && buff[1] == 'O' && buff[2] == 'S') {
            std::cerr << "Received EOS!\n";
            gather.close();
            break;
        }
        Net inputModel = deserializeModel(buff, modelSize);
        inputModel.to(device);
        torch::optim::SGD optimizer(inputModel.parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));
        
        for (int i = 0; i < train_epochs; i++) {
            train(i, &inputModel, device, *train_data_loader, optimizer, std::to_string(worker_id));
        }

        auto serialized_model = serializeModel(inputModel);
        assert(modelSize == serialized_model.size()); // check
        gather.sendrecv(serialized_model.c_str(), modelSize, nullptr, modelSize);
    }

    delete [] buff;

    Manager::finalize();
    return 0;
}
