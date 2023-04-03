
#include "mtcl.hpp"

int main(int argc, char *argv[]) {
    Manager::init("Broker", "config.json");

    auto master = Manager::connect("Master");
    auto bcast = Manager::createTeam("Broker:Worker1:Worker2", "Broker", BROADCAST);
    auto gather = Manager::createTeam("Broker:Worker1:Worker2", "Broker", GATHER);


    size_t bufferSize;

    while(master.probe(bufferSize)){
        char* buffer = new char[bufferSize];
        master.receive(buffer, bufferSize);
        bcast.send(buffer, bufferSize);
        delete [] buffer;

        char* backBuff = new char[bufferSize * gather.size()];

        gather.sendrecv(backBuff, bufferSize, backBuff, bufferSize); // local send not meaningful, just skipped

        for(int i = 1; i < gather.size(); i++)
            master.send(backBuff+(i*bufferSize), bufferSize);

        delete [] backBuff;
    }

    master.close();
    bcast.close();
    Manager::finalize(true);
    return 0;
}