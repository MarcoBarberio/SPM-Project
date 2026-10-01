# SPM Project

This repository contains the modular projects developed for the **Parallel and Distributed Systems: Paradigms and Models** course, academic year **2025/2026**.

The purpose of the project is to explore different paradigms for **parallel and distributed programming**, progressively moving from low-level data parallelism to shared-memory and distributed-memory approaches.

## Project Structure

The repository is organized into three main directories:

```text
SPM-Project/
├── assignments/     # Course assignments
├── playground/      # Tests, experiments and preliminary implementations
└── projects/        # Modular projects
```

## Modular Projects

The project is divided into four main parts, each focusing on a different parallel programming paradigm.

### 1. SIMD / AVX

The first part explores **data-level parallelism** using SIMD instructions.

The implementation uses **AVX vectorization** to execute the same operation on multiple data elements simultaneously, with the goal of reducing execution time compared to the corresponding scalar implementation.

Main topics:

- SIMD programming
- AVX intrinsics
- Vectorization
- Memory access patterns
- Performance comparison with scalar code

### 2. C++ Multithreading

The second part introduces **shared-memory parallelism** using native C++ threads.

The implementation explicitly manages threads and distributes the workload among them, providing direct control over synchronization and work scheduling.

Main topics:

- `std::thread`
- Thread synchronization
- Static and dynamic work distribution
- Thread pools
- Shared-memory parallelism
- Scalability and speedup

### 3. OpenMP

The third part implements shared-memory parallelism using **OpenMP**.

OpenMP provides a higher-level abstraction compared to explicitly managing C++ threads, allowing parallelism to be expressed through compiler directives.

Main topics:

- Parallel regions
- Parallel loops
- Work sharing
- Scheduling strategies
- Tasks
- Synchronization
- OpenMP scalability

### 4. MPI

The final part moves from shared-memory parallelism to **distributed-memory parallelism** using the **Message Passing Interface (MPI)**.

The workload is distributed across multiple independent processes, which communicate explicitly through message passing.

Main topics:

- MPI processes and ranks
- Point-to-point communication
- Collective communication
- Data distribution
- Distributed computation
- Scalability across multiple processes

## Performance Evaluation

The different implementations are evaluated experimentally by measuring execution time and scalability while varying the available parallel resources.

The analysis focuses on aspects such as:

- Execution time
- Speedup
- Scalability
- Work distribution
- Synchronization overhead
- Communication overhead
- Effects of increasing the number of threads or MPI processes

These experiments highlight the trade-offs between the different parallel programming models and the limits imposed by synchronization, memory bandwidth, communication and hardware resources.

## Technologies

The project mainly uses:

- **C / C++**
- **AVX / SIMD intrinsics**
- **C++ Threads**
- **OpenMP**
- **MPI**
- **Linux**

## Course

**Parallel and Distributed Systems: Paradigms and Models**  
Academic Year **2025/2026**
