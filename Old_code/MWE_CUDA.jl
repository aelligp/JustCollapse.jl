using ParallelStencil, CUDA
using ParallelStencil.FiniteDifferences2D
@init_parallel_stencil(CUDA, Float64, 2);

@parallel function foo(A, B)
    @all(A) = @av(B)
    return nothing
end

ni = nx,ny = 126,126

center = CuArray(zeros(ni...))
vertex = CuArray(ones(ni.+1...))

@parallel foo(center, vertex)

center = Int64.(center)
vertex = Int64.(vertex)