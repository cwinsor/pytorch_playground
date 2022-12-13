import parallelTestModule

# extractor = parallelTestModule.ParallelExtractor()
# extractor.runInParallel(numProcesses=2, numThreads=4)

if __name__ == '__main__':    
    extractor = parallelTestModule.ParallelExtractor()
    extractor.runInParallel(numProcesses=2, numThreads=4)
