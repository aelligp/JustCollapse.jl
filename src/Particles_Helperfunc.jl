## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------
using JustRelax 
import JustRelax.@cell

@inline init_particle_fields(particles) = @zeros(size(particles.coords[1])...) 
@inline init_particle_fields(particles, nfields) = tuple([zeros(particles.coords[1]) for i in 1:nfields]...)
@inline init_particle_fields(particles, ::Val{N}) where N = ntuple(_ -> @zeros(size(particles.coords[1])...) , Val(N))
@inline init_particle_fields_cellarrays(particles, ::Val{N}) where N = ntuple(_ -> @fill(0.0, size(particles.coords[1])..., celldims=(cellsize(particles.index))), Val(N))

function init_particles_cellarrays(nxcell, max_xcell, min_xcell, x, y, dx, dy, nx, ny)
    ni     = nx, ny
    ncells = nx * ny
    np     = max_xcell * ncells
    px, py = ntuple(_ -> @fill(NaN, ni..., celldims=(max_xcell,)) , Val(2))
    inject = @fill(false, nx, ny, eltype=Bool)
    index  = @fill(false, ni..., celldims=(max_xcell,), eltype=Bool) 
    
    @parallel_indices (i, j) function fill_coords_index(px, py, index)    
        # lower-left corner of the cell
        x0, y0 = x[i], y[j]
        # fill index array
        for l in 1:nxcell
            JustRelax.@cell px[l, i, j]    = x0 + dx * rand(0.05:1e-5:0.95)
            JustRelax.@cell py[l, i, j]    = y0 + dy * rand(0.05:1e-5:0.95)
            JustRelax.@cell index[l, i, j] = true
        end
        return nothing
    end

    @parallel (1:nx, 1:ny) fill_coords_index(px, py, index)    

    return Particles(
        (px, py), index, inject, nxcell, max_xcell, min_xcell, np, (nx, ny)
    )
end

# Velocity helper grids for the particle advection
function velocity_grids(xci, xvi, di)
    dx, dy  = di
    yVx     = LinRange(xci[2][1] - dy, xci[2][end] + dy, length(xci[2])+2)
    xVy     = LinRange(xci[1][1] - dx, xci[1][end] + dx, length(xci[1])+2)
    grid_vx = xvi[1], yVx
    grid_vy = xVy, xvi[2]

    return grid_vx, grid_vy
end

function copyinn_x!(A, B)

    @parallel function f_x(A, B)
        @all(A) = @inn_x(B)
        return nothing
    end

    @parallel f_x(A, B)
end