# const isCUDA = false
const isCUDA = true

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO

const backend = @static if isCUDA
    CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustRelax.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

using ParallelStencil, ParallelStencil.FiniteDifferences2D

@static if isCUDA
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

using JustPIC, JustPIC._2D
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
const backend_JP = @static if isCUDA
    CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

# Load script dependencies
using GeoParams, CairoMakie, CellArrays, Statistics, Dates, JLD2

# Load file with all the rheology configurations
include("Caldera_setup_sinkhole.jl")
include("Caldera_rheology_sinkhole.jl")

## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------

import ParallelStencil.INDICES
const idx_k = INDICES[2]
macro all_k(A)
    return esc(:($A[$idx_k]))
end

const idx_i = INDICES[1]
macro all_i(A)
    return esc(:($A[$idx_i]))
end

function copyinn_x!(A, B)
    @parallel function f_x(A, B)
        @all(A) = @inn_x(B)
        return nothing
    end

    return @parallel f_x(A, B)
end

# Initial pressure profile - not accurate
@parallel function init_P!(P, ρg, z)
    @all(P) = abs(@all(ρg) * @all_k(z)) #* <(@all_k(z), 0.0)
    return nothing
end

@parallel function init_P!(P, ρg, z, topo_y)
    @all(P) = @all(ρg) * (@all_i(topo_y) - @all_k(z))
    return nothing
end


function apply_pure_shear(Vx, Vy, εbg, xvi, lx, ly)
    xv, yv = xvi

    @parallel_indices (i, j) function pure_shear_x!(Vx, εbg, lx)
        xi = xv[i]
        Vx[i, j + 1] = εbg * (xi - lx * 0.5)
        return nothing
    end

    @parallel_indices (i, j) function pure_shear_y!(Vy, εbg, ly)
        yi = yv[j]
        Vy[i + 1, j] = abs(yi) * εbg
        return nothing
    end

    nx, ny = size(Vx)
    @parallel (1:nx, 1:(ny - 2)) pure_shear_x!(Vx, εbg, lx)
    nx, ny = size(Vy)
    @parallel (1:(nx - 2), 1:ny) pure_shear_y!(Vy, εbg, ly)

    return nothing
end

function apply_pure_shear(Ux, Uy, εbg, xvi, lx, ly, dt)
    xv, yv = xvi

    @parallel_indices (i, j) function pure_shear_x!(Ux, εbg, lx, dt)
        xi = xv[i]
        Ux[i, j + 1] = εbg * (xi - lx * 0.5) * dt
        return nothing
    end

    @parallel_indices (i, j) function pure_shear_y!(Uy, εbg, ly, dt)
        yi = yv[j]
        Uy[i + 1, j] = abs(yi) * εbg *dt
        return nothing
    end

    nx, ny = size(Ux)
    @parallel (1:nx, 1:(ny - 2)) pure_shear_x!(Ux, εbg, lx, dt)
    nx, ny = size(Uy)
    @parallel (1:(nx - 2), 1:ny) pure_shear_y!(Uy, εbg, ly, dt)

    return nothing
end

function extract_topo_from_GMG_phases(phases_GMG, xvi, air_phase)
    topo_idx = [findfirst(x -> x == air_phase, row) - 1 for row in eachrow(phases_GMG)]
    yv = xvi[2]
    topo_y = yv[topo_idx]
    return topo_y
end

function thermal_anomaly!(Temp, Ω_T, phase_ratios, T_chamber, T_air, conduit_phase, magma_phase, anomaly_phase, air_phase)

    @parallel_indices (i, j) function _thermal_anomaly!(Temp, Ω_T, T_chamber, T_air, vertex_ratio, conduit_phase, magma_phase, anomaly_phase, air_phase)
        # quick escape
        # conduit_ratio_ij = @index vertex_ratio[conduit_phase, i, j]
        magma_ratio_ij = @index vertex_ratio[magma_phase, i, j]
        anomaly_ratio_ij = @index vertex_ratio[anomaly_phase, i, j]
        air_ratio_ij = @index vertex_ratio[air_phase, i, j]

        # if conduit_ratio_ij > 0.5 || magma_ratio_ij > 0.5
        if anomaly_ratio_ij > 0.5
            Ω_T[i + 1, j] = Temp[i + 1, j] = T_chamber
        elseif magma_ratio_ij > 0.5
            Ω_T[i + 1, j] = Temp[i + 1, j] = T_chamber - 100.0e0
        elseif air_ratio_ij > 0.5
            Ω_T[i + 1, j] = Temp[i + 1, j] = T_air
        end

        return nothing
    end

    ni = size(phase_ratios.vertex)

    @parallel (@idx ni) _thermal_anomaly!(Temp, Ω_T, T_chamber, T_air, phase_ratios.vertex, conduit_phase, magma_phase, anomaly_phase, air_phase)

    @views Ω_T[1, :] .= Ω_T[2, :]
    @views Ω_T[end, :] .= Ω_T[end - 1, :]
    @views Temp[1, :] .= Temp[2, :]
    @views Temp[end, :] .= Temp[end - 1, :]

    return nothing
end

function plot_particles(particles, pPhases, chain; clrmap = :roma)
    p = particles.coords
    # pp = [argmax(p) for p in phase_ratios.center] #if you want to plot it in a heatmap rather than scatter
    ppx, ppy = p
    pxv = ppx.data[:] ./ 1.0e3
    pyv = ppy.data[:] ./ 1.0e3
    clr = pPhases.data[:]
    chain_x = chain.coords[1].data[:] ./ 1.0e3
    chain_y = chain.coords[2].data[:] ./ 1.0e3
    idxv = particles.index.data[:]
    f = Figure(size = (800, 600))
    ax = Axis(f[1, 1], title = "Particle positions", xlabel = "x [km]", ylabel = "y [km]", aspect = DataAspect())
    h = scatter!(ax, Array(pxv[idxv]), Array(pyv[idxv]), color = Array(clr[idxv]), colormap = clrmap, markersize = 1)
    scatter!(ax, Array(chain_x), Array(chain_y), color = :red, markersize = 1)
    Colorbar(f[1, 2], h)
    return f
end

function make_it_go_boom!(Q, threshold, cells, ϕ, V_erupt, V_tot, di, phase_ratios, magma_phase, anomaly_phase)

    @parallel_indices (i, j) function _make_it_go_boom!(Q, threshold, cells, ϕ, V_erupt, V_tot, dx, dy, center_ratio, magma_phase, anomaly_phase)

        magma_ratio_ij = @index center_ratio[magma_phase, i, j]
        anomaly_ratio_ij = @index center_ratio[anomaly_phase, i, j]
        total_fraction = magma_ratio_ij + anomaly_ratio_ij
        ϕ_ij = ϕ[i, j]
        cells_ij = cells[i, j]

        if (anomaly_ratio_ij > 0.5 || magma_ratio_ij > 0.5) && ϕ_ij ≥ threshold
            Q[i, j] = (V_erupt * inv(V_tot)) * ((total_fraction * cells_ij) * inv(numcells(cells)))
        end
        return nothing
    end

    ni = size(phase_ratios.center)

    @parallel (@idx ni) _make_it_go_boom!(Q, threshold, cells, ϕ, V_erupt, V_tot, di..., phase_ratios.center, magma_phase, anomaly_phase)
    V_tot += V_erupt

    return V_tot, V_erupt
end


function make_it_go_boom_smooth!(
    Q,
    cells,
    ϕ,
    V_erupt,
    V_tot,
    weights,
    phase_ratios,
    magma_phase,
    anomaly_phase,
)

    @parallel_indices (i, j) function _make_it_go_boom_smooth!(
        Q,
        cells,
        ϕ,
        V_erupt,
        V_tot,
        weights,
        center_ratio,
        magma_phase,
        anomaly_phase
    )
        magma_ratio_ij = @index center_ratio[magma_phase, i, j]
        anomaly_ratio_ij = @index center_ratio[anomaly_phase, i, j]
        total_fraction = magma_ratio_ij + anomaly_ratio_ij
        weight = weights[i,j]
        cells_ij = cells[i,j]

        if cells_ij > 0 && weight > 0
            Q[i, j] = (V_erupt * inv(V_tot)) * weight
        else
            Q[i, j] = 0.0
        end

        return nothing
    end

    ni = size(phase_ratios.center)

    @parallel (@idx ni) _make_it_go_boom_smooth!(
        Q,
        cells,
        ϕ,
        V_erupt,
        V_tot,
        weights,
        phase_ratios.center,
        magma_phase,
        anomaly_phase,
    )

    V_tot += V_erupt
    return V_tot, V_erupt
end

function compute_total_eruptible_volume(cells, dx::Float64, dy::Float64)
    V_total = 0.0
    nx, ny = size(cells)
    @inbounds for j in 1:ny
        # Compute semi-axes for the ellipse at this y-row
        a = sum(@views cells[:, j] .> 0) * dx / 2   # semi-major axis (x-direction)
        b = dy                                      # semi-minor axis (y-direction, per row)
        V_total += (4/3) * π * a * b * a
    end
    return V_total
end


function compute_cells_for_Q!(cells, threshold, phase_ratios, magma_phase, anomaly_phase, melt_fraction)
    @parallel_indices (I...) function _compute_cells_for_Q!(cells, threshold, center_ratio, magma_phase, anomaly_phase, melt_fraction)
        magma_ratio_ij = @index center_ratio[magma_phase, I...]
        anomaly_ratio_ij = @index center_ratio[anomaly_phase, I...]
        melt_fraction_ij = melt_fraction[I...]

        if (anomaly_ratio_ij > 0.5 || magma_ratio_ij > 0.5) && melt_fraction_ij ≥ threshold
            cells[I...] = true
        else
            cells[I...] = false
        end

        return nothing
    end

    ni = size(phase_ratios.center)
    return @parallel (@idx ni) _compute_cells_for_Q!(cells, threshold, phase_ratios.center, magma_phase, anomaly_phase, melt_fraction)

end

function compute_vertical_weights(cells, z; smoothing="cosine")
    weights = @zeros(size(z))

    # Convert to CPU arrays for computation
    z_mask = (cells .> 0)
    @views z_vals = z[z_mask]
    if isempty(z_vals)
        return weights
    end

    z_max = maximum(z_vals)  # top (shallower)
    z_min = minimum(z_vals)  # bottom (deeper)

    @parallel_indices (i, j) function _compute_weights!(weights, cells, z, z_max, z_min, smoothing_type)
        if cells[i,j] > 0
            # Now z_rel = 1 at top, 0 at bottom
            z_rel = (z[i,j] - z_min) / (z_max - z_min)

            if smoothing_type == 1  # linear
                weights[i,j] = z_rel
            elseif smoothing_type == 2  # cosine
                weights[i,j] = 0.5 * (1 + cos(pi * (1 - z_rel)))  # taper: 1 at top, 0 at bottom
            elseif smoothing_type == 3  # exp
                weights[i,j] = exp(-3 * (1 - z_rel))  # also 1 at top, decays downward
            else
                weights[i,j] = z_rel
            end
        end
        return nothing
    end

    smoothing_type = if smoothing == "linear"
        1
    elseif smoothing == "cosine"
        2
    elseif smoothing == "exp"
        3
    else
        1
    end

    ni = size(cells)
    @parallel (@idx ni) _compute_weights!(weights, cells, z, z_max, z_min, smoothing_type)

    # Normalize weights to sum to 1
    s = sum(weights)
    if s > 0
        weights ./= s
    end

    return weights
end

function compute_vertical_weights_bottom(cells, z; smoothing="cosine")
    weights = @zeros(size(z))

    # Convert to CPU arrays for computation
    z_mask = (cells.> 0)
    @views z_vals = z[z_mask]
    if isempty(z_vals)
        return weights
    end

    z_max = maximum(z_vals)  # top (shallower)
    z_min = minimum(z_vals)  # bottom (deeper)

    @parallel_indices (i, j) function _compute_weights_bottom!(weights, cells, z, z_max, z_min, smoothing_type)
        if cells[i,j] > 0
            # Now z_rel = 0 at top, 1 at bottom (inverted from original)
            z_rel = (z_max - z[i,j]) / (z_max - z_min)

            if smoothing_type == 1  # linear
                weights[i,j] = z_rel
            elseif smoothing_type == 2  # cosine
                weights[i,j] = 0.5 * (1 + cos(pi * (1 - z_rel)))  # taper: 1 at bottom, 0 at top
            elseif smoothing_type == 3  # exp
                weights[i,j] = exp(-3 * (1 - z_rel))  # also 1 at bottom, decays upward
            else
                weights[i,j] = z_rel
            end
        end
        return nothing
    end

    smoothing_type = if smoothing == "linear"
        1
    elseif smoothing == "cosine"
        2
    elseif smoothing == "exp"
        3
    else
        1
    end

    ni = size(cells)
    @parallel (@idx ni) _compute_weights_bottom!(weights, cells, z, z_max, z_min, smoothing_type)

    # Normalize weights to sum to 1
    s = sum(weights)
    if s > 0
        weights ./= s
    end

    return weights
end

numcells(A::AbstractArray) = count(x -> x == 1.0, A)

function compute_thermal_source!(
    H, T_erupt, threshold, V_erupt, cells, ϕ,
    phase_ratios, dt, args, di, magma_phase, anomaly_phase, rheology; α=1.0
)
    @parallel_indices (I...) function _compute_thermal_source!(
        H, T_erupt, threshold, V_erupt, cells, center_ratio, dt, args,
        dx, dy, magma_phase, anomaly_phase, rheology, α
    )
        magma_ratio_ij = @index center_ratio[magma_phase, I...]
        anomaly_ratio_ij = @index center_ratio[anomaly_phase, I...]
        total_fraction = magma_ratio_ij + anomaly_ratio_ij
        ϕ_ij = ϕ[I...]
        Tij_bg = args.Temp_bg[I...]
        Tij = args.T[I...]
        phase_ij = center_ratio[I...]
        args_ij = (; T = Tij, P = args.P[I...])
        V_eruptij = V_erupt * ((total_fraction * cells[I...]) * inv(numcells(cells)))
        ρCp = JustRelax.JustRelax2D.compute_ρCp(rheology, phase_ij, args_ij)

        if ((anomaly_ratio_ij > 0.5 || magma_ratio_ij > 0.5) && ϕ_ij ≥ threshold)
            # Ensure temperature does not go below Temp_bg
            ΔT = max(T_erupt - max(Tij, Tij_bg), 0.0)
            H[I...] = ((V_eruptij / dt) * ρCp[I...] * ΔT) / (dx * dy * dx)
            # H[I...] = ((V_eruptij / dt) * ρCp[I...] * ΔT / (dx * dy * dx)

        #   [W/m^3] = [[m3/[s]] * [[kg/m^3] * [J/kg/K]] * [K]] / [[m] * [m] * [m]]
        end
        return nothing
    end

    ni = size(phase_ratios.center)

    @parallel (@idx ni) _compute_thermal_source!(
        H, T_erupt, threshold, V_erupt, cells, phase_ratios.center, dt, args,
        di..., magma_phase, anomaly_phase, rheology, α
    )
end


function compute_thermal_source!(
    H, T_erupt, threshold, V_erupt, cells, ϕ,
    phase_ratios, dt, args, di, magma_phase, anomaly_phase, rheology, weights;
    α = 0.3
)
    @parallel_indices (I...) function _compute_thermal_source!(
        H, T_erupt, threshold, V_erupt, cells, weights,
        center_ratio, dt, args, dx, dy, magma_phase, anomaly_phase, rheology, α
    )
        magma_ratio_ij = @index center_ratio[magma_phase, I...]
        anomaly_ratio_ij = @index center_ratio[anomaly_phase, I...]
        # total_fraction = magma_ratio_ij + anomaly_ratio_ij
        ϕ_ij = ϕ[I...]
        Tij_bg = args.Temp_bg[I...]
        Tij = args.T[I...]
        phase_ij = center_ratio[I...]
        args_ij = (; T = Tij, P = args.P[I...])
        V_eruptij = V_erupt * weights[I...]  # ← smoother vertical control
        ρCp = JustRelax.JustRelax2D.compute_ρCp(rheology, phase_ij, args_ij)

        if ((anomaly_ratio_ij > 0.5 || magma_ratio_ij > 0.5) && ϕ_ij ≥ threshold)
            ΔT = max(T_erupt - max(Tij, Tij_bg), 0.0) * α       # ← partial ΔT removal
            H[I...] = ((V_eruptij / dt) * ρCp[I...] * ΔT) / (dx * dy * dx)
        end

        return nothing
    end

    ni = size(phase_ratios.center)

    @parallel (@idx ni) _compute_thermal_source!(
        H, T_erupt, threshold, V_erupt, cells, weights, phase_ratios.center,
        dt, args, di..., magma_phase, anomaly_phase, rheology, α
    )
end

function compute_VEI!(V_erupt)
    if V_erupt <= 1.0e4
        return 0
    elseif V_erupt <= 1.0e6
        return 1
    elseif V_erupt <= 1.0e7
        return 2
    elseif V_erupt <= 1.0e8
        return 3
    elseif V_erupt <= 1.0e9
        return 4
    elseif V_erupt <= 1.0e10
        return 5
    elseif V_erupt <= 1.0e11
        return 6
    elseif V_erupt <= 1.0e12
        return 7
    else
        return 8
    end
end


# ## END OF HELPER FUNCTION ------------------------------------------------------------

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main(li, origin, phases_GMG, T_GMG, T_bg, igg; nx = 16, ny = 16, figdir = "figs2D", do_vtk = false, fric_angle = 30, extension = 1.0e-15 * 0, cutoff_visc = (1.0e16, 1.0e23), V_total = 0.0, V_eruptible = 0.0, layers = 1, air_phase = 6, progressiv_extension = false, plotting = true, displacement=false)

    # Physical domain ------------------------------------
    ni = nx, ny           # number of cells
    di = @. li / ni       # grid steps
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid             # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    oxd_wt = (61.6, 0.9, 17.7, 3.65, 2.35, 5.38, 4.98, 1.27, 3.0)
    rheology = init_rheologies(layers, oxd_wt, fric_angle; incompressible = false, magma = true)
    rheology_incomp = init_rheologies(layers, oxd_wt, fric_angle; incompressible = true, magma = true)
    # dt_time = 100 * 3600 * 24 * 365
    dt_time = 1.0e3 * 3600 * 24 * 365
    κ = (4 / (rheology[1].HeatCapacity[1].Cp.val * rheology[1].Density[1].ρ.val)) # thermal diffusivity                                 # thermal diffusivity
    dt_diff = 0.5 * min(di...)^2 / κ / 2.01
    dt = min(dt_time, dt_diff)
    # ----------------------------------------------------
    # Weno model -----------------------------------------
    weno = WENO5(backend, Val(2), ni.+1) # ni.+1 for ∂18O
    # WENO arrays
    Vx_v = @zeros(ni.+1...)
    Vy_v = @zeros(ni.+1...)
    # Initialize particles -------------------------------
    nxcell = 100
    max_xcell = 150
    min_xcell = 75
    particles = init_particles(
        backend_JP, nxcell, max_xcell, min_xcell, xvi, di, ni
    )
    subgrid_arrays = SubgridDiffusionCellArrays(particles)
    # velocity grids
    grid_vxi = velocity_grids(xci, xvi, di)
    # material phase & temperature
    pPhases, pT, pδ18O = init_cell_arrays(particles, Val(3))

    # Assign particles phases anomaly
    phases_device = PTArray(backend)(phases_GMG)
    phase_ratios = phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    init_phases!(pPhases, phases_device, particles, xvi)

    # Initialize marker chain
    nxcell, max_xcell, min_xcell = 100, 150, 75
    initial_elevation = 0.0e0
    chain = init_markerchain(backend_JP, nxcell, min_xcell, max_xcell, xvi[1], initial_elevation)
    # air_phase                    = 6
    topo_y = extract_topo_from_GMG_phases(phases_GMG, xvi, air_phase)

    fill_chain_from_vertices!(chain, PTArray(backend)(topo_y))
    update_phases_given_markerchain!(pPhases, chain, particles, origin, di, air_phase)

    for _ in 1:3
        @views hn = 0.5 .* (topo_y[1:(end - 1)] .+ topo_y[2:end])
        @views topo_y[2:(end - 1)] .= 0.5 .* (hn[1:(end - 1)] .+ hn[2:end])
        fill_chain_from_vertices!(chain, PTArray(backend)(topo_y))
        update_phases_given_markerchain!(pPhases, chain, particles, origin, di, air_phase)
    end
    update_phase_ratios!(
        phase_ratios, particles, xci, xvi, pPhases
    )
    # rock ratios for variational stokes
    # RockRatios
    ϕ = RockRatio(backend, ni)
    # update_rock_ratio!(ϕ, phase_ratios, air_phase)
    compute_rock_fraction!(ϕ, chain, xvi, di)

    # particle fields for the stress rotation
    pτ = StressParticles(particles)
    particle_args = (pT, pδ18O, pPhases, unwrap(pτ)...)
    particle_args_reduced = (pT, unwrap(pτ)...)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ_rel = 1.0e-5, ϵ_abs = 1.0e-4, Re = 3.0, r = 0.7, CFL = 0.8 / √2.1) # Re=3π, r=0.7


    # randomize cohesion
    perturbation_C = @rand(ni...)
    # stokes.EII_pl .+= (1e-2.*perturbation_C)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal = ThermalArrays(backend, ni)
    @views thermal.T[2:(end - 1), :] .= PTArray(backend)(T_GMG)

    # Add thermal anomaly BC's
    T_chamber = 1223.0e0
    T_air = 273.0e0

    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (; left = true, right = true, top = false, bot = false),
    )
    thermal_bcs!(thermal, thermal_bc)
    temperature2center!(thermal)
    Ttop = thermal.T[2:(end - 1), end]
    Tbot = thermal.T[2:(end - 1), 1]
    Temp_bg =copy(thermal.Tc)
    @views Temp_bg .= PTArray(backend)(T_bg[1:end-1, 1:end-1])
    # ----------------------------------------------------

    # Buoyancy forces
    ρg = ntuple(_ -> @zeros(ni...), Val(2))
    for _ in 1:5
        compute_ρg!(ρg, phase_ratios, rheology, (T = thermal.Tc, P = stokes.P))
        @parallel init_P!(stokes.P, ρg[end], xvi[2])
        # @parallel init_P!(stokes.P, ρg[end], xvi[2], PTArray(backend)(topo_y))
    end
    # stokes.P        .= PTArray(backend)(reverse(cumsum(reverse((ρg[2]).* di[2], dims=2), dims=2), dims=2))

    # Melt fraction
    ϕ_m = @zeros(ni...)
    compute_melt_fraction!(
        ϕ_m, phase_ratios, rheology, (T = thermal.Tc, P = stokes.P)
    )
    # Rheology
    args0 = (; ϕ = ϕ_m, T = thermal.Tc, P = stokes.P, dt = Inf, perturbation_C = perturbation_C, Temp_bg = Temp_bg)
    viscosity_cutoff = cutoff_visc
    compute_viscosity!(stokes, phase_ratios, args0, rheology, viscosity_cutoff; air_phase = air_phase)

    # PT coefficients for thermal diffusion
    pt_thermal = PTThermalCoeffs(
        backend, rheology, phase_ratios, args0, dt, ni, di, li; ϵ = 1.0e-8, CFL = 0.95 / √2
    )

    # Boundary conditions
    displacement == true ?
        flow_bcs         = DisplacementBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        free_surface = false,
    ) : flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        free_surface = false,)
    εbg = extension
    isa(flow_bcs, VelocityBoundaryConditions) ? apply_pure_shear(@velocity(stokes)..., εbg, xvi, li...) : apply_pure_shear(@displacement(stokes)..., εbg, xvi, li..., dt)

    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    isa(flow_bcs, VelocityBoundaryConditions) ? nothing : displacement2velocity!(stokes, dt)
    update_halo!(@velocity(stokes)...)

    # IO -------------------------------------------------
    # if it does not exist, make folder where figures are stored
    if plotting
        if do_vtk
            vtk_dir = joinpath(figdir, "vtk")
            take(vtk_dir)
        end
        take(figdir)
        checkpoint = joinpath(figdir, "checkpoint")
    end
    # ----------------------------------------------------

    local Vx_v, Vy_v
    if do_vtk
        Vx_v = @zeros(ni .+ 1...)
        Vy_v = @zeros(ni .+ 1...)
    end

    T_buffer = @zeros(ni .+ 1)
    Told_buffer = similar(T_buffer)
    dt₀ = similar(stokes.P)
    for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
        copyinn_x!(dst, src)
    end
    grid2particle!(pT, xvi, T_buffer, particles)

    # ## Plot initial T and P profile
    τxx_v = @zeros(ni .+ 1...)
    τyy_v = @zeros(ni .+ 1...)
    cells = similar(stokes.Q)
    P_lith = @zeros(ni...)

    # Time loop
    t, it, er_it = 0.0, 0, 0
    interval = 1
    eruption_counter = 0
    iterMax = 150.0e3
    thermal.Told .= thermal.T

    eruption = false
    V_erupt_fast = -V_total / 3
    V_max_eruptable = V_total / 2
    ΔPc = 20.0e6 # 20MPa

    # Initialize the tracking arrays
    VEI_array = Int[]
    eruption_times = Float64[]
    eruption_counters = Int[]
    Volume = Float64[]
    erupted_volume = Float64[]
    volume_times = Float64[]
    overpressure = Float64[]
    overpressure_t = Float64[]
    local iters, er_it, eruption_counter, Vx_v, Vy_v, d18O

    depth = PTArray(backend)([y for _ in xci[1], y in xci[2]]);
    compute_cells_for_Q!(cells, 0.01, phase_ratios, 3, 4, ϕ_m);
    V_total = compute_total_eruptible_volume(cells, di...);

    while it < 500
        if it == 1
            P_lith .= stokes.P
        end

        if it >1 && iters.iter > iterMax && iters.err_evo1[end] > pt_stokes.ϵ_abs * 5
            iterMax += 10e3
            iterMax = min(iterMax, 200e3)
            println("Increasing maximum pseudo timesteps to $iterMax")
        else
            iterMax = 150e3
        end

        if progressiv_extension
            if it > 4 && round(t/(3600 * 24 *365.25); digits=2) >= 6e3*interval
                if interval ==1
                    εbg = 1e-15
                    extension = 1e-15
                    interval += 1
                else
                    εbg += extension
                    εbg = min(εbg, 5*extension)
                end
                println("Progressively increased extension to $εbg")
                apply_pure_shear(@velocity(stokes)..., εbg, xvi, li...)
                flow_bcs!(stokes, flow_bcs) # apply boundary conditions
                update_halo!(@velocity(stokes)...)
            end
        end

        # interpolate fields from particle to grid vertices
        particle2grid!(T_buffer, pT, xvi, particles)
        @views T_buffer[:, end] .= Ttop
        @views T_buffer[:, 1] .= Tbot
        @views thermal.T[2:(end - 1), :] .= T_buffer
        thermal_bcs!(thermal, thermal_bc)
        temperature2center!(thermal)

        args = (; ϕ = ϕ_m, T = thermal.Tc, P = stokes.P, dt = Inf, perturbation_C = perturbation_C, ΔTc=thermal.ΔTc,  Temp_bg = Temp_bg)
        if it == 4
            CUDA.@allowscalar pp = [p[3] > 0 || p[4] > 0 for p in phase_ratios.center]
            # pp = [p[3] > 0 || p[4] > 0 for p in phase_ratios.center]
            V_max_eruptable = V_total / 2
            V_erupt_fast = -V_total / 2
            compute_cells_for_Q!(cells, 0.01, phase_ratios, 3, 4, ϕ_m)
            V_total_cells = compute_total_eruptible_volume(cells, di...)
            V_total = min(V_total_cells, V_total)
            println("Critical overpressure reached - erupting with fast rate")
            @views stokes.Q .= 0.0
            @views thermal.H .= 0.0
            eruption_counter += 1
            er_it = 0
            eruption = true
            dt *= 0.1
            V_erupt = V_erupt_fast

            compute_cells_for_Q!(cells, 0.5, phase_ratios, 3, 4, ϕ_m)
            V_tot = V_total
            if (V_total - V_erupt) > 0.0
                weights = compute_vertical_weights(cells, PTArray(backend)(depth); smoothing = "cosine")  # or "linear", "exp"
                T_erupt = mean(thermal.Tc[cells .== true])
                V_total, V_erupt = make_it_go_boom_smooth!(stokes.Q, cells, ϕ_m, V_erupt, V_tot, weights, phase_ratios, 3, 4)
            end
            println("Volume total: $(round(ustrip.(uconvert(u"km^3", (V_total)u"m^3")); digits = 5)) km³")
            println("Erupted Volume: $(round(ustrip.(uconvert(u"km^3", (V_erupt)u"m^3")); digits = 2)) km³")

            # Compute VEI and update arrays
            VEI = compute_VEI!(abs(V_erupt))
            push!(VEI_array, VEI)
            push!(erupted_volume, abs(V_erupt))
            push!(eruption_times, (t / (3600 * 24 * 365.25) / 1.0e3))
            push!(eruption_counters, eruption_counter)
        end


        stress2grid!(stokes, pτ, xvi, xci, particles)

        t_stokes = @elapsed begin
            iters = solve_VariationalStokes!(
                stokes,
                pt_stokes,
                di,
                flow_bcs,
                ρg,
                phase_ratios,
                ϕ,
                it <= 3 ? rheology_incomp : rheology,
                args,
                dt,
                igg;
                kwargs = (;
                    iterMax = it < 5 || eruption == true ? 150.0e3 : iterMax,
                    strain_increment = displacement,
                    nout = 2.0e3,
                    viscosity_cutoff = viscosity_cutoff,
                )
            )
        end
        # rotate stresses
        rotate_stress!(pτ, stokes, particles, xci, xvi, dt)

        println("Stokes solver time             ")
        println("   Total time:      $t_stokes s")
        println("Extrema T[C]: $(extrema(thermal.T .- 273))")
        tensor_invariant!(stokes.ε)
        tensor_invariant!(stokes.ε_pl)

        if eruption == true
            er_it += 1
            dtmax = 100 * 3600 * 24 * 365.25
        else
            if it > 3 && !isempty(overpressure) && overpressure[end] < 0.0
                dtmax = 5e3 * 3600 * 24 * 365.25
            else
            dtmax = 1.0e3 * 3600 * 24 * 365.25
            end
        end
        dt = compute_dt(stokes, di, dtmax)

        println("dt = $(dt / (3600 * 24 * 365.25)) years")
        # ------------------------------

        # Thermal solver ---------------
        # if it ≥ 3
        # heatdiffusion_PT!(
        #     thermal,
        #     pt_thermal,
        #     thermal_bc,
        #     it <= 3 ? rheology_incomp : rheology,
        #     args,
        #     dt,
        #     di;
        #     kwargs = (
        #         igg = igg,
        #         phase = phase_ratios,
        #         iterMax = 100.0e3,
        #         nout = 1.0e2,
        #         verbose = true,
        #     )
        # )
        # @. thermal.T[2:end-2, 1:end-1] = max(thermal.T[2:end-2, 1:end-1], Temp_bg)
        # temperature2center!(thermal)
        # thermal.ΔT .= thermal.T .- thermal.Told
        # vertex2center!(thermal.ΔTc, thermal.ΔT[2:(end - 1), :])

        # subgrid_characteristic_time!(
        #     subgrid_arrays, particles, dt₀, phase_ratios, rheology, thermal, stokes, xci, di
        # )
        # centroid2particle!(subgrid_arrays.dt₀, xci, dt₀, particles)
        # subgrid_diffusion!(
        #     pT, thermal.T, thermal.ΔT, subgrid_arrays, particles, xvi, di, dt
        # )
        # end
        # ------------------------------

        # Advection --------------------
        copyinn_x!(T_buffer, thermal.T)
        # advect particles in space
        advection!(particles, RungeKutta4(), @velocity(stokes), grid_vxi, dt)
        # advect particles in memory
        move_particles!(particles, xvi, particle_args)
        # check if we need to inject particles
        center2vertex!(τxx_v, stokes.τ.xx)
        center2vertex!(τyy_v, stokes.τ.yy)
        inject_particles_phase!(
            particles,
            pPhases,
            particle_args_reduced,
            (T_buffer, τxx_v, τyy_v, stokes.τ.xy, stokes.ω.xy),
            xvi
        )

        # advect marker chain
        advect_markerchain!(chain, RungeKutta4(), @velocity(stokes), grid_vxi, dt)
        update_phases_given_markerchain!(pPhases, chain, particles, origin, di, air_phase)

        compute_melt_fraction!(
            ϕ_m, phase_ratios, rheology, (T = thermal.Tc, P = stokes.P)
        )

        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
        compute_rock_fraction!(ϕ, chain, xvi, di)

        tensor_invariant!(stokes.τ)

        @show it += 1
        t += dt
        if igg.me == 0
            push!(Volume, V_total)
            push!(volume_times, (t / (3600 * 24 * 365.25) / 1.0e3))
            CUDA.@allowscalar pp = [p[3] > 0 || p[4] > 0 for p in phase_ratios.center]
            # pp = [p[3] > 0 || p[4] > 0 for p in phase_ratios.center]
            if it > 1 && !isempty(Array(stokes.P)[pp][Array(ϕ_m)[pp] .≥ 0.3])
            push!(overpressure, maximum(Array(stokes.P)[pp][Array(ϕ_m)[pp]  .≥ 0.3] .- Array(P_lith)[pp][Array(ϕ_m)[pp]  .≥ 0.3]))
            push!(overpressure_t, t / (3600 * 24 * 365.25) / 1.0e3)
            end
        end

        if it == 1
            stokes.EII_pl .= 0.0
        end

        if plotting
            # Data I/O and plotting ---------------------
            if it == 1 || rem(it, 10) == 0
                if igg.me == 0 && it == 1
                    metadata(pwd(), checkpoint, joinpath(@__DIR__, "Caldera2D.jl"), joinpath(@__DIR__, "Caldera_setup.jl"), joinpath(@__DIR__, "Caldera_rheology.jl"))
                end
                checkpointing_jld2(checkpoint, stokes, thermal, t, dt, igg)
                checkpointing_particles(checkpoint, particles; phases = pPhases, phase_ratios = phase_ratios, chain = chain, particle_args = particle_args, t = t, dt = dt)
                η_eff = @. stokes.τ.II / (2 * stokes.ε.II)
                (; η_vep, η) = stokes.viscosity
                if do_vtk
                    velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
                    data_v = (;
                        T = Array(T_buffer),
                        stress_xy = Array(stokes.τ.xy),
                        strain_rate_xy = Array(stokes.ε.xy),
                        phase_vertices = [argmax(p) for p in Array(phase_ratios.vertex)],
                    )
                    data_c = (;
                        P = Array(stokes.P),
                        viscosity_vep = Array(η_vep),
                        viscosity_eff = Array(η_eff),
                        viscosity = Array(η),
                        phases = [argmax(p) for p in Array(phase_ratios.center)],
                        Melt_fraction = Array(ϕ_m),
                        EII_pl = Array(stokes.EII_pl),
                        stress_II = Array(stokes.τ.II),
                        strain_rate_II = Array(stokes.ε.II),
                        plastic_strain_rate_II = Array(stokes.ε_pl.II),
                        density = Array(ρg[2] ./ 9.81),
                    )
                    velocity_v = (
                        Array(Vx_v),
                        Array(Vy_v),
                    )
                    save_vtk(
                        joinpath(vtk_dir, "vtk_" * lpad("$it", 6, "0")),
                        xvi ./ 1.0e3,
                        xci ./ 1.0e3,
                        data_v,
                        data_c,
                        velocity_v;
                        t = round(t / (1.0e3 * 3600 * 24 * 365.25); digits = 3),
                        pvd = joinpath(vtk_dir, "Caldera2D")
                    )
                    save_particles(particles, pPhases; conversion = 1.0e3, fname = joinpath(vtk_dir, "particles_" * lpad("$it", 6, "0")),
                        pvd = joinpath(vtk_dir, "Caldera2D_Particles"), t = round(t / (1.0e3 * 3600 * 24 * 365.25); digits = 3))
                    save_marker_chain(joinpath(vtk_dir, "chain_" * lpad("$it", 6, "0")), xvi[1] ./ 1.0e3, Array(chain.h_vertices) ./ 1.0e3; pvd = joinpath(vtk_dir, "Caldera2D_Markerchain"), t = round(t / (1.0e3 * 3600 * 24 * 365.25); digits = 3))
                end

                chain_x = chain.coords[1].data[:] ./ 1.0e3
                chain_y = chain.coords[2].data[:] ./ 1.0e3

                ar = DataAspect()
                fig = Figure(size = (1200, 900), title = "t = $t")
                ax1 = Axis(fig[1, 1], aspect = ar, title = "T [C]  (t=$(round(t / (1.0e3 * 3600 * 24 * 365.25); digits = 3)) Kyrs)")
                ax2 = Axis(fig[2, 1], aspect = ar, title = "Vy [cm/yr]")
                ax3 = Axis(fig[1, 3], aspect = ar, title = "τII [MPa]")
                ax4 = Axis(fig[2, 3], aspect = ar, title = "log10(η)")
                ax5 = Axis(fig[3, 1], aspect = ar, title = "EII_pl")
                ax6 = Axis(fig[3, 3], aspect = ar, title = "ΔP from P_lith")
                # Plot temperature
                h1 = heatmap!(ax1, xvi[1] .* 1.0e-3, xvi[2] .* 1.0e-3, Array(thermal.T[2:(end - 1), :] .- 273), colormap = :batlow)
                # Plot velocity
                h2 = heatmap!(ax2, xvi[1] .* 1.0e-3, xvi[2] .* 1.0e-3, ustrip.(uconvert.(u"cm/yr", Array(stokes.V.Vy)u"m/s")), colormap = :vik) #, colorrange= (-(ustrip.(1u"cm/yr")), (ustrip(1u"cm/yr"))))
                scatter!(ax2, Array(chain_x), Array(chain_y), color = :red, markersize = 3)
                # Plot 2nd invariant of stress
                h3 = heatmap!(ax3, xci[1] .* 1.0e-3, xci[2] .* 1.0e-3, Array(stokes.τ.II) ./ 1.0e6, colormap = :batlow)
                # Plot effective viscosity
                h4 = heatmap!(ax4, xci[1] .* 1.0e-3, xci[2] .* 1.0e-3, Array(log10.(stokes.viscosity.η_vep)), colorrange = log10.(viscosity_cutoff), colormap = :batlow)
                h5 = heatmap!(ax5, xci[1] .* 1.0e-3, xci[2] .* 1.0e-3, Array(stokes.EII_pl), colormap = :batlow)
                # contour!(ax5, xci[1] .* 1.0e-3, xci[2] .* 1.0e-3, Array(ϕ_m), levels = [0.5, 0.75, 1.0], color = :white, linewidth = 1.5, labels=true)
                # h6  = heatmap!(ax6, xci[1].*1e-3, xci[2].*1e-3, Array(ϕ_m) , colormap=:lipari, colorrange=(0.0,1.0))
                h6 = heatmap!(ax6, xci[1] .* 1.0e-3, xci[2] .* 1.0e-3, (Array(stokes.P) .- Array(P_lith)) ./ 1.0e6, colormap = :roma)

                hidexdecorations!(ax1)
                hidexdecorations!(ax2)
                hidexdecorations!(ax3)
                hidexdecorations!(ax4)
                hideydecorations!(ax3)
                hideydecorations!(ax4)
                hideydecorations!(ax6)

                Colorbar(fig[1, 2], h1)
                Colorbar(fig[2, 2], h2)
                Colorbar(fig[1, 4], h3)
                Colorbar(fig[2, 4], h4)
                Colorbar(fig[3, 2], h5)
                Colorbar(fig[3, 4], h6)
                linkaxes!(ax1, ax2, ax3, ax4, ax5)
                fig
                save(joinpath(figdir, "$(it).png"), fig)

                # Plot Drucker Prager yield surface
                fig1 = let
                    depth = [y for _ in xci[1], y in xci[2]]
                    fig = Figure(; size = (1200, 900))
                    ax = Axis(fig[1, 1]; title = "Drucker Prager")
                    lines!(ax, [0.0e6, maximum(stokes.P)] ./ 1.0e6, [10.0e6 * cosd(rheology[1].CompositeRheology[1].elements[end].ϕ.val) ; (maximum(stokes.P) * sind(rheology[1].CompositeRheology[1].elements[end].ϕ.val) + rheology[1].CompositeRheology[1].elements[end].C.val * cosd(rheology[1].CompositeRheology[1].elements[end].ϕ.val))] ./ 1.0e6, color = :black, linewidth = 2)
                    s1 = scatter!(ax, Array((stokes.P) ./ 1.0e6)[:], Array(stokes.τ.II ./ 1.0e6)[:]; color = Array(stokes.R.RP)[:], colormap = :roma, markersize = 3)
                    Colorbar(fig[1, 2], s1)
                    fig
                    save(joinpath(figdir, "DruckerPrager_$it.png"), fig)
                end
                fig2 = let
                    fig = Figure(; size = (1200, 900))

                    ax2 = Axis(
                        fig[1, 1], ylabel = "Erupted Volume [km³]", yscale = log10, xlabel = "Time [Kyrs]",
                        yticklabelsize = 25,
                        xticklabelsize = 25,
                        xlabelsize = 25,
                        ylabelsize = 25,
                    )
                    vei_boundaries = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

                    cmap = Makie.to_colormap(:lipari10)
                    vei_colors = [cmap[i] for i in round.(Int, range(1, length(cmap), length=9))]  # 8 colors for VEI 0-8

                    vei_labels = ["VEI 0\n(non-explosive)", "VEI 1\n(small)", "VEI 2\n(moderate)", "VEI 3\n(moderate-large)",
                                    "VEI 4\n(large)", "VEI 5\n(very large)", "VEI 6\n(colossal)", "VEI 7\n(super-colossal)", "VEI 8 \n (Mega-colossal)"]
                    vei_volume_thresholds = [1e-6, 0.00001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 5000.0]
                    x_band = [0.0, (t / (3600 * 24 * 365.25) / 1.0e3) +1.0]

                    for i in 1:length(vei_boundaries)-1
                        y_bottom = vei_volume_thresholds[i]
                        y_top = vei_volume_thresholds[i+1]
                        band!(
                            ax2,
                            x_band,
                            [y_bottom, y_bottom],
                            [y_top, y_top],
                            color = (vei_colors[i], 0.5)
                        )
                        text!(
                            ax2,
                            0.0 + 0.02 * (0.0),
                            (y_bottom + y_top) / 2,
                            text = vei_labels[i],
                            align = (:left, :center),
                            fontsize = 16,
                            color = :black,
                            font = "bold"
                        )
                    end
                    scatterlines!(
                        ax2, eruption_times, (round.(ustrip.(uconvert.(u"km^3", (erupted_volume)u"m^3")); digits = 5)),
                        color = :blue,
                        markersize = 10,
                        linewidth = 2,
                    )
                    ylims!(ax2, 1e-6, 5000)
                    xlims!(ax2,0.0, (t / (3600 * 24 * 365.25) / 1.0e3) +1.0)
                    fig
                    save(joinpath(figdir, "eruption_data.png"), fig)
                    save(joinpath(figdir, "eruption_data.svg"), fig)
                end
                fig3 = let
                    fig1 = Figure(; size = (1200, 900))
                    ax1 = Axis(
                        fig1[1, 1]; title = L"Volume\ change\ over\ time", xlabel = L"Time\ [Kyrs]", ylabel = L"Volume\ [km³]",
                        titlesize = 40,
                        yticklabelsize = 25,
                        xticklabelsize = 25,
                        xlabelsize = 25,
                        ylabelsize = 25
                    )
                    scatterlines!(ax1, volume_times, (round.(ustrip.(uconvert.(u"km^3", (Volume)u"m^3")); digits = 5)), color = :blue, markersize = 5)
                    fig1
                    save(joinpath(figdir, "volume_change.png"), fig1)
                    save(joinpath(figdir, "volume_change.svg"), fig1)
                end

                fig4 = let
                    fig1 = Figure(; size = (1200, 900))
                    ax1 = Axis(
                        fig1[1, 1]; title = L"Overpressure\ \Delta P_c", xlabel = L"Time\ [Kyrs]", ylabel = L"Pressure\ [MPa]",
                        titlesize = 40,
                        yticklabelsize = 25,
                        xticklabelsize = 25,
                        xlabelsize = 25,
                        ylabelsize = 25
                    )
                    scatterlines!(ax1, overpressure_t, overpressure./1e6, color = :violet, markersize = 5)
                    fig1
                    save(joinpath(figdir, "Overpressure.png"), fig1)
                    save(joinpath(figdir, "Overpressure.svg"), fig1)
                end
            end
            # ------------------------------
        end
    end

    return nothing
end

## END OF MAIN SCRIPT ----------------------------------------------------------------
const plotting = true
const progressiv_extension = false
const displacement = false  #set solver to displacement or velocity
do_vtk = true # set to true to generate VTK files for ParaView

# figdir is defined as Systematics_depth_radius_ar_extension
figdir = "Analogue_Modelling_Benchmark"
n = 480
nx, ny = n, n >> 1

# IO -------------------------------------------------
# if it does not exist, make folder where figures are stored
if plotting
    if do_vtk
        vtk_dir = joinpath(figdir, "vtk")
        take(vtk_dir)
    end
    take(figdir)
    checkpoint = joinpath(figdir, "checkpoint")
    take(checkpoint)
end
# ----------------------------------------------------

li, origin, phases_GMG, T_GMG, T_bg, Grid, V_total, V_eruptible, layers, air_phase = setup2D(
    nx + 1, ny + 1;
    sticky_air = 2.0e0,
    dimensions = (35.0e0, 20.0e0), # extent in x and y in km
    flat = true, # flat or volcano cone
    chimney = false, # conduit or not
    layers = 0, # number of layers
    volcano_size = (3.0e0, 7.0e0),    # height, radius
    conduit_radius = 1.0e-2, # radius of the conduit
    chamber_T = 850.0e0, # temperature of the chamber
    chamber_radius = 2.5, # radius of the chamber
    chamber_depth  = 6.0, # depth of the chamber
    aspect_x       = 1.5, # aspect ratio of the chamber
)

igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI = true, select_device = false)...)
else
    igg
end

main(li, origin, phases_GMG, T_GMG, T_bg, igg; figdir = figdir, nx = nx, ny = ny, do_vtk = do_vtk, fric_angle = 30.0, extension = 0.0, cutoff_visc = (1.0e17, 1.0e23), V_total = V_total, V_eruptible = V_eruptible, layers = layers, air_phase = air_phase, progressiv_extension = progressiv_extension, plotting = plotting, displacement=displacement);
