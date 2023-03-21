# MTCL based distributed Federated Learning!

To compile make sure you have defined the following environment variables:

 - `MTCL_HOME` and `RAPIDJSON_HOME` in order to use the MTCL library
 - `TORCH_HOME` to include and link pytorch
 
 You need also MPI and UCX (with UCC) to exploits those transport protocols in MTCL library.
 
 After that use cmake to compile the examples:

```bash
mkdir build
cd build
cmake ../
make
```

Use convenience scripts in `./data` to download the MNIST dataset.

## Master-slave example

In order to run this example, make sure the configuration file config.json is well formatted and structured. Then execute the application using the following structure:

    mpirun -np 1 ./master : -np 1 ./slave 1 : -np 1 ./slave 2

