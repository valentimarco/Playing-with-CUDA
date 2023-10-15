#include <iostream>
#include "../common/cpu_anim.h"

#define N (100*1024)


__global__ void add(int* a, int* b, int* c){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N){
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }
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

    int blocks = (N + 127) / 128;
    add<<<128,128>>>(dev_a,dev_b,dev_c);

    cudaMemcpy(c,dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i=0; i < N; i++){
        std::cout << a[i] << " + " << b[i] << "= "<< c[i] << std::endl;
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}


//GPU ripple
#define DIM_x 1920
#define DIM_y 1080
struct DataBlock{
    unsigned char *dev_bitmap;
    CPUAnimBitmap *bitmap;
};

void cleanUp(DataBlock *d){
    cudaFree(d->dev_bitmap);
}

__global__ void kernel(unsigned char *ptr, int ticks){
    //map from threadIdx/BlockIdx to pixel position 
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    //now calculate the value at that position
    float fx = x - DIM_x/2;
    float fy = y - DIM_y/2;
    float d = sqrtf(fx * fx + fy * fy );
    unsigned char gray = (unsigned char) (128.0f + 127.0f * cos(d / 10.0f - ticks / 7.0f) / (d / 10.0f + 1.0f));
    ptr[offset * 4 + 0] = gray;
    ptr[offset * 4 + 1] = gray;
    ptr[offset * 4 + 2] = gray;
    ptr[offset * 4 + 3] = 255;
}

//generate the frame by indicating the datablock and the tick
void generate_frame(DataBlock *d, int ticks){
    dim3 blocks(DIM_x/16,DIM_y/16); //dividing the image DIMxDIM with DIM/16 x DIM/16
    dim3 threads(16,16); //each block have 16 x 16
    kernel<<<blocks,threads>>>(d->dev_bitmap, ticks);
    cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,d->bitmap->image_size(), cudaMemcpyDeviceToHost);
}

void gpuRipple(){
    DataBlock data;
    CPUAnimBitmap bitmap(DIM_x,DIM_y, &data);
    data.bitmap = &bitmap;
    cudaMalloc((void**)&data.dev_bitmap, bitmap.image_size());
    bitmap.anim_and_exit((void (*)(void*,int))generate_frame, (void (*) (void*)) cleanUp);
}



int main(){
    //sumVectors();
    gpuRipple();
    return 0;
}