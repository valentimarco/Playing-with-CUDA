#include <iostream>

#define N 66000

__global__ void add(int* a, int* b, int* c){
    int tid = blockIdx.x; // block index 
    if (tid < N)
        c[tid] = a[tid] + b[tid];
}


int main(){
    int a[N], b[N], c[N] = {0};
    int *dev_a, *dev_b, *dev_c;
    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * sizeof(int));

    for(int i=0; i < N; i++){
        a[i] = -i;
        b[i] = i * i;
    }

    cudaMemcpy(dev_a,a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b,b, N * sizeof(int), cudaMemcpyHostToDevice);

    add<<<N,1>>>(dev_a,dev_b,dev_c);

    cudaMemcpy(c,dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i=0; i < N; i++){
        std::cout << a[i] << " + " << b[i] << "= "<< c[i] << std::endl;
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}