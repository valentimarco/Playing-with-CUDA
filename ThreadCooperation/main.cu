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


// Dot product
#define imin(a,b) (a < b ? a : b)
const int N_v = 33 * 1024;
const int tpb = 256; //thread per blocks

__global__ void dot(float *a, float *b, float *c){
    __shared__ float cache[tpb];
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //index for vectors
    int cacheindex = threadIdx.x;
    float temp = 0;
    
    while(tid < N){
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheindex] = temp;

    __syncthreads();

    //time to reduce, tpb must be a power of 2
    int i = blockDim.x/2;
    while(i != 0){
        if(cacheindex < i) cache[cacheindex] += cache[cacheindex + i];
        __syncthreads();
        
        i /= 2;
    }

    //after the reduction, the value in the first position is the sum of each pairwise
    if(cacheindex == 0) c[blockIdx.x] = cache[0];
}


void dotProduct(){
    const int bpg = imin(32, (N_v+tpb-1) / tpb); //blocks per grid
    float *a, *b, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;

    a = (float*)malloc(N_v*sizeof(float));
    b = (float*)malloc(N_v*sizeof(float));
    
    partial_c = (float*)malloc(bpg*sizeof(float));

    cudaMalloc((void**) &dev_a, N_v*sizeof(float));
    cudaMalloc((void**) &dev_b, N_v*sizeof(float));
    cudaMalloc((void**) &dev_partial_c, N_v*sizeof(float));

    for(int i = 0; i < N_v; i++){
        a[i] = i;
        b[i] = i*2;
    }

    cudaMemcpy(dev_a, a, N_v*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N_v*sizeof(float), cudaMemcpyHostToDevice);

    dot<<<bpg,tpb>>>(dev_a,dev_b,dev_partial_c);

    cudaMemcpy(partial_c, dev_partial_c, bpg*sizeof(float), cudaMemcpyDeviceToHost);

    float c_v = 0;
    for(int i = 0; i < bpg; i++) 
        c_v += partial_c[i];
    
    //the dot product should be the sum of squares of the integers......
    #define sum_squares(x) (x*(x+1)*(2*x-1)/6)

    std::cout << "Does GPU value " << c_v << " = " << sum_squares((float) (N_v-1)) << " ?" << std::endl;

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_c);

    free(a);
    free(b);
    free(partial_c);
}


int main(){
    //sumVectors();
    gpuRipple();
    //dotProduct();
    return 0;
}