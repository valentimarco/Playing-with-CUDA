## Thread Cooperation

Now that we understand the basics, we can now talk about splitting Parallel Blocks!

All code we launch in device before,have N blocks with 1 thread each, but we can half the number of blocks by apply 2 thread for each block and so on!

By taking the summing of 2 vectors, we can modify the code as following:

1. In the triple angle brackets, we change the second parameter with a value that not exceed the **maxThreadsPerBlock** defined on the device properties (finded in the **cudaDeviceProp** struct)
2. change the tid variable by using the row-major ordering in the kernel function **add**, but why? Imagine have a matrix where the columns are Threads and the rows are blocks, we can access the entry of the matrix by taking the product of the block index with the number of threads and adding the thread index within the block.
3. To not exceed the 512 thread limitation (??) we use a fixed amount of threads like 128 and calculate the number of blocks by dividing N (array size) with 128. To do the division without using celling, we can use the trick of adding 127 to N and than divide by 128. (This method not work for value grater than 65 355, so change it with a fixed value)
4. There is also a limitation on the number of blocks... The solution to this issue is very simple: we change the **if statement** with a **while statement** and increment the tid value by itself with the product of the blockDim and the gridDim


Now let's try with a image: We wanna create the GPU ripple!

1. We create a Datablock struct where we store the dev pointer that we pass later to the device and the bitmap. We than pass the reference to the CPUAnimBitmap constructor and assign the bitmap on the Datablock.
2. We allocate memory in the device of the image size.
3. Finally we use the anim_and_exit method to render the image, where we 2 functions' pointers.

The function call generate_frame is the core of all code: We create 2 dim3 struct call blocks and threads, where the first one is DIM_x/16 x DIM_x/16 and the second one is a 16 x 16.
Than we pass this vectors, the dev_pointer and the value ticks to the kernel function, and the kernel function calculate some shit!
