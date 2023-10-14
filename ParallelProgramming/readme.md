## Parallel Programming

In this example we will discover the parallel programming by summing 2 vectors!

In the code we create the following:

1. &nbsp;3 vectors of N dimension in the Host
2. &nbsp;3 poiters for allocate the 3 vectors in the Device
3. &nbsp;Fill 2 of this vectors in the Host
4. &nbsp;Calculate the Sum
5. &nbsp;Retrive the result from the device and Print the result in the Host

In the code we have 2 major difference:

1. &nbsp;When we define the Kernel, the first value of the triple angle brackets rapresent the number of "copies" we wanna create. We call each of this copy a **block**

2. &nbsp;In the kernel, we define a new variable called "tid" and assign **blockIdx** value in it. It contains the block index in which the kernel is execute and we <span style="color:orange">MUST</span> check if exceeds the lenght of the vectors!
The blockIdx is a multi-dimensional value, useful for problem in multi-dimensional domains, such as matrix








