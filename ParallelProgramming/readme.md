## Parallel Programming

In this example we will discover the parallel programming by summing 2 vectors and display a Julia set!

In the code for summing 2 vectors, we create the following:

1. &nbsp;3 vectors of N dimension in the Host
2. &nbsp;3 poiters for allocate the 3 vectors in the Device
3. &nbsp;Fill 2 of this vectors in the Host
4. &nbsp;Calculate the Sum
5. &nbsp;Retrive the result from the device and Print the result in the Host

In the code we have 2 major difference:

1. &nbsp;When we define the Kernel, the first value of the triple angle brackets rapresent the number of "copies" we wanna create. We call each of this copy a **block**

2. &nbsp;In the kernel, we define a new variable called "tid" and assign **blockIdx** value in it. It contains the block index in which the kernel is execute and we <span style="color:orange">MUST</span> check if exceeds the lenght of the vectors!
The blockIdx is a multi-dimensional value, useful for problem in multi-dimensional domains, such as matrix

In the code for dipaly a julia set, we create the following:

1. &nbsp;Define a struct for Complex number, where all methods are usable only from the device
2. &nbsp;Define a Julia function usable only from the device that calulate the set and return 1 if the point is in the set
3. &nbsp;Define the kernel where iterate through all points we care to render, in particular all points that are in the julia set are red!
4. &nbsp;On main, define a Bitmap and allocate the necessary memory in the device
5. &nbsp;Call the kernel with a two-dimensional grid (Needs a dim3)
6. &nbsp;Retrive the result from the device and display the result in the host using openGL







