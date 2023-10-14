#include <iostream>


__global__ void add(int a, int b, int *c){
    *c = a + b;
}   
    


int main(){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    std::cout << "Hello World! You have " << deviceCount << " available device" << std::endl;

    std::cout << "Info for each device: " << std::endl;

    cudaDeviceProp deviceProp;

    for (int i = 0; i < deviceCount; i++) {
        cudaGetDeviceProperties(&deviceProp, i);
        std::cout << "Device " << i << ": " << deviceProp.name << std::endl;
        std::cout << "TotalGlobalMem" << ": " << deviceProp.totalGlobalMem << "Bytes"<< std::endl;
        std::cout << "SharedMemPerBlock"<< ": " << deviceProp.sharedMemPerBlock << "Bytes" << std::endl;
    }
    memset(&deviceProp, 0, sizeof(cudaDeviceProp));
    deviceProp.major = 2;
    deviceProp.minor = 0;
    
    std::cout << "Selecting all gpu with compute capability >= 2.0" << std::endl;
    cudaChooseDevice(&deviceCount, &deviceProp);
    cudaSetDevice(deviceCount);
    
    

    int c;
    int *dev_c;
    cudaMalloc((void**)&dev_c, sizeof(int));
    add<<<1,1>>>(2,7,dev_c);
    // host memory can only be access by host code, and vice versa!
    cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);

    printf("2 + 7 = %d\n", c);
    cudaFree(dev_c);




    return 0;
}