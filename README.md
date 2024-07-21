# Welcome
Hi there,

Over the next few months, I'll be reading **Programming Massively Parallel Processors** by Hwu, Kirk, and Hajj.

I'll be posting my notes here as I go along. I hope it helps anyone who shares my interest in parallel computation!

Happy learning!

\- Logan

# Running CUDA Code
I'll be using Google Colab to start out.

To run a CUDA program in Colab:
1. Change the runtime to T4 GPU (as of May 2024)
2. Run the following in your notebook to check that NVCC & CUDA are installed. By default, the T4 GPU runtime should come with CUDA 12.
    ```
    !nvcc --version
    ```
3. Copy your .cu files to your Colab environment's file system
4. Add a second code block where you compile your .cu files. For a simple program contained in a single file `main.cu`, linking a single library (jpeglib, for example), we can use the following:
    ```
    !nvcc -o main main.cu -ljpeg
    ```
5. Add a third code block and run your compiled executable:
    ```
    !./main
    ```

# Table of Contents

0. [Preface](./notes/chapter00.md)
1. [Intro](./notes/chapter01.md)
### Part I: Fundamental Concepts
2. [Heterogeneous data parallel computing](./notes/chapter02.md)
3. [Multidimensional grids and data](./notes/chapter03.md)
4. [Compute architecture and scheduling](./notes/chapter04.md)
5. [Memory architecture and data locality](./notes/chapter05.md)
6. [Performance considerations](./notes/chapter06.md)
### Part II: Parallel Patterns
7. [Convolution](./notes/chapter07.md)
8. Stencil
9. Parallel histogram
10. Reduction
11. Prefix sum (scan)
12. Merge
### Part III: Advanced Patterns and Applications
13. Sorting
14. Sparse matrix computation
15. Graph traversal
16. Deep learning
17. Iterative MRI reconstruction
18. Electrostatic potential map
19. Parallel programming and computational thinking
### Part IV: Advanced Practices
20. Programming a heterogeneous computing cluster
21. CUDA dynamic parallelism
22. Advanced practices and future evolution
23. Conclusion

<br>

# Sample Programs:
- [Vector Addition](./programs/c02s06_vectorAddition.cu)
- [RGB to Grayscale](./programs/c03s02_rgbToGrayscale.cu)
- [Thread-Coarsened Tiled Matrix Multiplication](./programs/c06s03_thread_coarsened_tiled_matmul.cu)