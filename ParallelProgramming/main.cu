#include <iostream>
#include "../common/cpu_bitmap.h"

#define N 66000


__global__ void add(int* a, int* b, int* c){
    int tid = blockIdx.x; // block index 
    if (tid < N)
        c[tid] = a[tid] + b[tid];
}


void sumVectors(){
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
}

//Julia Set

#define DIM 1080

struct cuComplex{
    float r;
    float i;
    __device__ cuComplex(float a, float b): r(a), i(b) {}
    __device__ float magnitude2(){
        return r * r + i * i; 
    }
    __device__ cuComplex operator*(const cuComplex& a){
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    __device__ cuComplex operator+(const cuComplex& a){
        return cuComplex(r+a.r, i+a.i);
    }
};

__device__ int julia(int x, int y){
    const float scale = 1.5;
    float jx = scale * (float) (DIM/2 - x) / (DIM/2);
    float jy = scale * (float) (DIM/2 - y) / (DIM/2);
    cuComplex c(0.285, 0.013);
    cuComplex a(jx,jy);
    
    for (size_t i = 0; i < 200; i++){
        a = a*a + c;
        if(a.magnitude2() > 1000) return 0;
    }
    
    return 1;
    
}

__global__ void kernel(unsigned char *ptr){
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;

    int juliaValue = julia(x,y);
    ptr[offset*4 + 0] = 255 * juliaValue;
    ptr[offset*4 + 1] = 0;
    ptr[offset*4 + 2] = 0;
    ptr[offset*4 + 3] = 255;
}


void dispalyJuliaSet(){
    CPUBitmap bitmap(DIM, DIM);
    unsigned char *dev_bitmap;

    cudaMalloc((void**) &dev_bitmap, bitmap.image_size());

    dim3 grid(DIM,DIM);
    kernel<<<grid,1>>>(dev_bitmap);

    cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);
    bitmap.display_and_exit();

    cudaFree(dev_bitmap);
}

int main(){
    sumVectors();
    dispalyJuliaSet();
    return 0;
}