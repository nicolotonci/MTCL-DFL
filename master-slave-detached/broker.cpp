
#include "mtcl.hpp"

int main(int argc, char *argv[]) {
    Manager::init("Broker", "config.json");

    auto master = Manager::connect("TCP:Master");
    auto bcast = Manager::createTeam("Broker:Worker1:Worker2", "Broker", BROADCAST);
    auto gather = Manager::createTeam("Broker:Worker1:Worker2", "Broker", MTCL_GATHER);


    size_t modelSize;
    master.receive(&modelSize, sizeof(size_t));
    bcast.sendrecv(&modelSize, sizeof(size_t), nullptr, 0);
    char* buffer = new char[modelSize];
    char* backBuff = new char[modelSize * gather.size()];
    while(true){
        if (master.receive(buffer, modelSize) == 0)
            break;

        bcast.sendrecv(buffer, modelSize, nullptr, 0);

        gather.sendrecv(buffer, modelSize, backBuff, modelSize); // local send not meaningful, just skipped

        for(int i = 1; i < gather.size(); i++)
            master.send(backBuff+(i*modelSize), modelSize);
    }
    delete [] backBuff;
    delete [] buffer;
    master.close();
    std::string eos(modelSize, 'E'); eos[1] = 'O'; eos[2] = 'S';
    bcast.sendrecv(eos.c_str(), modelSize, nullptr, 0);
    bcast.close();
    Manager::finalize(true);
    return 0;
}
