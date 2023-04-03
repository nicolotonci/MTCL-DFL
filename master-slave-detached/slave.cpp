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
    std::string data_path = "../../data";  // Patch to the MNISt data files (absolute or with respect to build directory)
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

    std::cout << "Training on " << num_workers << " wokers." << std::endl;

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

    // Get train and test data
    auto train_dataset = torch::data::datasets::MNIST(data_path)
            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
            .map(torch::data::transforms::Stack<>());

    auto test_dataset = torch::data::datasets::MNIST(data_path, torch::data::datasets::MNIST::Mode::kTest)
            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
            .map(torch::data::transforms::Stack<>());

    Manager::init("Worker"+std::to_string(worker_id), "config.json");

    auto bcast = Manager::createTeam("Master:Worker1:Worker2", "Master", BROADCAST);
    auto gather = Manager::createTeam("Master:Worker1:Worker2", "Master", GATHER);

    auto train_data_loader = torch::data::make_data_loader(train_dataset, torch::data::samplers::DistributedRandomSampler(
                                                                                                               train_dataset.size().value(),
                                                                                                               1/*num_workers*/,
                                                                                                               worker_id%8,
                                                                                                               false),
                                                                                                       train_batchsize);
    auto optimizer = torch::optim::SGD(Net().parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));

    int epoch = 0;
    while (true){
        size_t received_size = -1;
        bcast.probe(received_size, true);
        if (!received_size){
            gather.close();
            break;
        }

        char* buff = new char[received_size];
        bcast.receive(buff, received_size);
        Net inputModel = deserializeModel(buff, received_size);
        delete [] buff;

        for (int i = 0; i < train_epochs; i++) {
            train(++epoch, &inputModel, device, *train_data_loader, optimizer, std::to_string(worker_id));
        }

        auto serialized_model = serializeModel(inputModel);
        gather.sendrecv(serialized_model.c_str(), serialized_model.size(), nullptr, 0);
    }

    Manager::finalize();
    return 0;
}