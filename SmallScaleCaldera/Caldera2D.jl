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
include("Caldera_setup.jl")
include("Caldera_rheology.jl")

## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------

import ParallelStencil.INDICES
const idx_k = INDICES[2]
macro all_k(A)
    return esc(:($A[$idx_k]))
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
    f, ax, h = scatter(Array(pxv[idxv]), Array(pyv[idxv]), color = Array(clr[idxv]), colormap = clrmap, markersize = 1)
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

function make_it_go_boom!(Q, threshold, cells, ϕ, ρg, Δρ, V_erupt, V_tot, di, phase_ratios, magma_phase, anomaly_phase)

    @parallel_indices (i, j) function _make_it_go_boom!(Q, threshold, cells, ϕ, ρg, Δρ, V_erupt, V_tot, dx, dy, center_ratio, magma_phase, anomaly_phase)

        magma_ratio_ij = @index center_ratio[magma_phase, i, j]
        anomaly_ratio_ij = @index center_ratio[anomaly_phase, i, j]
        total_fraction = magma_ratio_ij + anomaly_ratio_ij
        ϕ_ij = ϕ[i, j]
        cells_ij = cells[i, j]
        ρg_ij = ρg[i, j]

        if (anomaly_ratio_ij > 0.5 || magma_ratio_ij > 0.5) && ϕ_ij ≥ threshold
            Q[i, j] =(((Δρ-ρg_ij) * inv(ρg_ij)) + (V_erupt * inv(V_tot))) * ((total_fraction * cells_ij) * inv(numcells(cells)))
        end
        return nothing
    end

    ni = size(phase_ratios.center)

    @parallel (@idx ni) _make_it_go_boom!(Q, threshold, cells, ϕ, ρg, Δρ, V_erupt, V_tot, di..., phase_ratios.center, magma_phase, anomaly_phase)
    V_tot += V_erupt

    return V_tot, V_erupt
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

numcells(A::AbstractArray) = count(x -> x == 1.0, A)

function compute_thermal_source!(H, T_erupt, threshold, V_erupt, cells, ϕ, phase_ratios, dt, args, di, magma_phase, anomaly_phase, rheology)
    @parallel_indices (I...) function _compute_thermal_source!(H, T_erupt, threshold, V_erupt, cells, center_ratio, dt, args, dx, dy, magma_phase, anomaly_phase, heology)
        magma_ratio_ij = @index center_ratio[magma_phase, I...]
        anomaly_ratio_ij = @index center_ratio[anomaly_phase, I...]
        phase_ij = center_ratio[I...]
        total_fraction = magma_ratio_ij + anomaly_ratio_ij
        V_eruptij = V_erupt * ((total_fraction * cells[I...]) * inv(numcells(cells)))

        ϕ_ij = ϕ[I...]
        args_ij = (; T = args.T[I...], P = args.P[I...])
        Tij_bg = args.Temp_bg[I...]
        Tij = args_ij.T
        ρCp = JustRelax.JustRelax2D.compute_ρCp(rheology, phase_ij, args_ij)

        if ((anomaly_ratio_ij > 0.5 || magma_ratio_ij > 0.5) && ϕ_ij ≥ threshold)
            # Ensure temperature does not go below Temp_bg
            ΔT = max(T_erupt - max(Tij, Tij_bg), 0.0)
            H[I...] = ((V_eruptij / dt) * ρCp[I...] * ΔT) / (dx * dy * dx)
            # H[I...] = ((V_eruptij / dt) * ρCp[I...] * (max(T_erupt - Tij, 0.0))) / (dx * dy * dx)

        #   [W/m^3] = [[m3/[s]] * [[kg/m^3] * [J/kg/K]] * [K]] / [[m] * [m] * [m]]
        end
        return nothing
    end

    ni = size(phase_ratios.center)

    @parallel (@idx ni) _compute_thermal_source!(H, T_erupt, threshold, V_erupt, cells, phase_ratios.center, dt, args, di..., magma_phase, anomaly_phase, rheology)
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

function d18O_anomaly!(
    d18O, z, phase_ratios,
    magma_phase,
    anomaly_phase,
    lower_crust,
    air_phase;
    crust_gradient::Bool = true,
    crust_min::Float64 = -10.0,
    crust_max::Float64 = 3.0,
    crust_const::Float64 = 0.0,
    magma_const::Float64 = 5.5,

)
    ni = size(phase_ratios.vertex)

    @parallel_indices (i, j) function _d18O_anomaly!(d18O, z, vertex_ratio, magma_phase, anomaly_phase, lower_crust, air_phase)

        magma_ratio_ij = @index vertex_ratio[magma_phase, i, j]
        anomaly_ratio_ij = @index vertex_ratio[anomaly_phase, i, j]
        lower_crust_ratio_ij = @index vertex_ratio[lower_crust, i, j]
        air_ratio_ij = @index vertex_ratio[air_phase, i, j]

        if magma_ratio_ij > 0.5 || lower_crust_ratio_ij > 0.5 || anomaly_ratio_ij > 0.5
            d18O[i,j] = magma_const
        elseif air_ratio_ij > 0.5
            d18O[i,j] = 3.0
        elseif z[j] .> -3e3
            if crust_gradient
                # Linear gradient from crust_min at shallowest to crust_max at deepest
                zmin = z[1]
                zmax = z[end]
                d18O[i, j] = crust_min + (crust_max - crust_min) * (z[j] - zmin) / (zmax - zmin)
            else
                d18O[i, j] = crust_const
            end
        elseif z[j] .< -3e3
            d18O[i,j] = 5.0
        end
        return nothing
    end

    @parallel (@idx ni) _d18O_anomaly!(d18O, z, phase_ratios.vertex, magma_phase, anomaly_phase, lower_crust, air_phase)

    return nothing
end

# ## END OF HELPER FUNCTION ------------------------------------------------------------
# ## Custom Colormap by T.Keller
# using MAT
# using Colors, ColorSchemes

# matfile = matopen("SmallScaleCaldera/ocean.mat")
# my_cmap_data = read(matfile, "ocean") # e.g., "colormap_variable" is the variable name in the .mat file
# close(matfile)
# ocean = ColorScheme([RGB(r...) for r in eachrow(my_cmap_data)])
# ocean_rev = ColorScheme([RGB(r...) for r in eachrow(reverse(my_cmap_data, dims=1))])
# --------------------------------------------------------

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main(li, origin, phases_GMG, T_GMG, T_bg, igg; nx = 16, ny = 16, figdir = "figs2D", do_vtk = false, fric_angle = 30, extension = 1.0e-15 * 0, cutoff_visc = (1.0e16, 1.0e23), V_total = 0.0, V_eruptible = 0.0, layers = 1, air_phase = 6, progressiv_extension = false, plotting = true)

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
    κ = (4 / (rheology[1].HeatCapacity[1].Cp.val * rheology[1].Density[1].ρ0.val)) # thermal diffusivity                                 # thermal diffusivity
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

    # ----------------------------------------------------
    # Track isotope ratios
    d18O = @zeros(ni.+ 1...)
    d18O_anomaly!(d18O, xvi[2], phase_ratios, 3, 4, 2, air_phase)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ_rel = 1.0e-5, ϵ_abs = 1.0e-6, Re = 3.0, r = 0.7, CFL = 0.8 / √2.1) # Re=3π, r=0.7
    # pt_stokes = PTStokesCoeffs(li, di; ϵ = 1.0e-4, Re = 3*√10*π/2, r = 0.5, CFL = 0.8 / √2.1) # Re=3π, r=0.7

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
    # flow_bcs         = DisplacementBoundaryConditions(;
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        free_surface = false,
    )

    εbg = extension
    apply_pure_shear(@velocity(stokes)..., εbg, xvi, li...)

    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
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
    let
        compo = [oxd_wt[1] (oxd_wt[7]+oxd_wt[8])]
        fig=Plot_TAS_diagram(compo; sz=(1000, 1000))
        save(joinpath(figdir, "TAS_diagram.png"), fig)
    end

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
    volume = Float64[]
    erupted_volume = Float64[]
    volume_times = Float64[]
    overpressure = Float64[]
    overpressure_t = Float64[]
    local iters, er_it, eruption_counter, Vx_v, Vy_v, d18O

    while it < 500 #000 # run only for 5 Myrs
        if it == 1
            P_lith .= stokes.P
        end

        if it >1 && iters.iter > iterMax && iters.err_evo1[end] > pt_stokes.ϵ * 5
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
        # args = (; ϕ=ϕ_m, T=thermal.Tc, P=stokes.P, dt=Inf)

        if it > 3
            CUDA.@allowscalar pp = [p[3] > 0 || p[4] > 0 for p in phase_ratios.center]
            # pp = [p[3] > 0 || p[4] > 0 for p in phase_ratios.center]
            V_max_eruptable = V_total / 2
            V_erupt_fast = -V_total / 3
            # if eruption == false && ((any((Array(stokes.P)[pp] .- Array(P_lith)[pp]) .≥ ΔPc .&& (Array(ϕ_m)[pp] .≥ 0.5)) .&& any(Array(ϕ_m) .≥ 0.5)) .|| rem(it, 30) == 0.0).&& (V_total - abs(V_erupt_fast)) ≥ V_max_eruptable
            if eruption == false && ((any(maximum(Array(stokes.P)[pp][Array(ϕ_m)[pp]  .≥ 0.3] .- Array(P_lith)[pp][Array(ϕ_m)[pp]  .≥ 0.3]) .≥ ΔPc .&& Array(ϕ_m)[pp] .≥ 0.5) .&& any(Array(ϕ_m) .≥ 0.5)) .|| rem(it, 30) == 0.0).&& (V_total - abs(V_erupt_fast)) ≥ V_max_eruptable
                println("Critical overpressure reached - erupting with fast rate")
                @views stokes.Q .= 0.0
                @views thermal.H .= 0.0
                eruption_counter += 1
                er_it = 0
                eruption = true
                dt *= 0.1
                if rand() < 0.1
                    V_erupt = V_erupt_fast
                else
                    V_erupt = rand() * V_erupt_fast
                end
                V_tot = V_total
                if (V_total - V_erupt_fast) > 0.0
                    compute_cells_for_Q!(cells, 0.5, phase_ratios, 3, 4, ϕ_m)
                    T_erupt = mean(thermal.Tc[cells .== true])
                    V_total, V_erupt = make_it_go_boom!(stokes.Q, 0.5, cells, ϕ_m, V_erupt, V_tot, di, phase_ratios, 3, 4)
                    # ρ_out = mean(ρg[end][cells .== true])
                    # V_total, V_erupt = make_it_go_boom!(stokes.Q, 0.5, cells, ϕ_m, ρg[end], ρ_out, V_erupt, V_tot, di, phase_ratios, 3, 4)
                    compute_thermal_source!(thermal.H, T_erupt, 0.5, V_erupt, cells, ϕ_m, phase_ratios, dt, args, di,  3, 4, rheology)
                end
                println("Volume total: $(round(ustrip.(uconvert(u"km^3", (V_total)u"m^3")); digits = 5)) km³")
                println("Erupted Volume: $(round(ustrip.(uconvert(u"km^3", (V_erupt)u"m^3")); digits = 2)) km³")

                # Compute VEI and update arrays
                VEI = compute_VEI!(abs(V_erupt))
                push!(VEI_array, VEI)
                push!(erupted_volume, abs(V_erupt))
                push!(eruption_times, (t / (3600 * 24 * 365.25) / 1.0e3))
                push!(eruption_counters, eruption_counter)
                # push!(overpressure, maximum(Array(stokes.P)[pp][Array(ϕ_m)[pp]  .≥ 0.3] .- Array(P_lith)[pp][Array(ϕ_m)[pp]  .≥ 0.3]))
                # push!(overpressure_t, t / (3600 * 24 * 365.25) / 1.0e3)

            elseif eruption == false && any((maximum(Array(stokes.P)[pp][Array(ϕ_m)[pp]  .≥ 0.3] .- Array(P_lith)[pp][Array(ϕ_m)[pp]  .≥ 0.3])) .< ΔPc .&& Array(ϕ_m)[pp]  .≥ 0.3) #.&& depth .≤ -2500)
                println("Adding volume to the chamber ")
                @views stokes.Q .= 0.0
                @views thermal.H .= 0.0
                V_tot = V_total
                T_addition = 1000+273e0
                V_erupt = if rand() < 0.1
                    0.0 / (3600 * 24 * 365.25) * dt
                else
                    (rand(1e-6:1e-5:6e-3) * 1.0e9) / (3600 * 24 * 365.25) * dt # [m3/s * dt] Constrained by  https://doi.org/10.1029/2018GC008103
                end
                compute_cells_for_Q!(cells, 0.5, phase_ratios, 3, 4, ϕ_m)
                V_total, V_erupt = make_it_go_boom!(stokes.Q, 0.5, cells, ϕ_m, V_erupt, V_tot, di, phase_ratios, 3, 4)
                # ρ_in = mean(ρg[end][ϕ.center .== 1.0])
                # V_total, V_erupt = make_it_go_boom!(stokes.Q, 0.5, cells, ϕ_m, ρg[end], ρ_in, V_erupt, V_tot, di, phase_ratios, 3, 4)
                compute_thermal_source!(thermal.H, T_addition, 0.5, V_erupt, cells, ϕ_m, phase_ratios, dt, args, di,  3, 4, rheology)
                println("Added Volume: $(round(ustrip.(uconvert(u"km^3", (V_erupt)u"m^3")); digits = 5)) km³")
                println("Volume total: $(round(ustrip.(uconvert(u"km^3", (V_total)u"m^3")); digits = 5)) km³")
                # push!(overpressure, maximum(Array(stokes.P)[pp][Array(ϕ_m)[pp]  .≥ 0.3] .- Array(P_lith)[pp][Array(ϕ_m)[pp]  .≥ 0.3]))
                # push!(overpressure_t, t / (3600 * 24 * 365.25) / 1.0e3)
            end
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
                    iterMax = it < 5 || eruption == true ? 250.0e3 : iterMax,
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
        if er_it > 4
            println("Eruption stopped")
            eruption = false
        end
        if eruption == true
            er_it += 1
            dtmax = 100 * 3600 * 24 * 365.25
        else
            # dtmax = 5e2 * 3600 * 24 * 365.25
            dtmax = 1.0e3 * 3600 * 24 * 365.25
            # dtmax = 100 * 3600 * 24 * 365.25
        end
        dt = compute_dt(stokes, di, dtmax)

        println("dt = $(dt / (3600 * 24 * 365.25)) years")
        # ------------------------------

        # Thermal solver ---------------
        if it > 3
        heatdiffusion_PT!(
            thermal,
            pt_thermal,
            thermal_bc,
            it <= 3 ? rheology_incomp : rheology,
            args,
            dt,
            di;
            kwargs = (
                igg = igg,
                phase = phase_ratios,
                iterMax = 100.0e3,
                nout = 1.0e2,
                verbose = true,
            )
        )
        @. thermal.T[2:end-2, 1:end-1] = max(thermal.T[2:end-2, 1:end-1], Temp_bg)
        temperature2center!(thermal)
        thermal.ΔT .= thermal.T .- thermal.Told
        vertex2center!(thermal.ΔTc, thermal.ΔT[2:(end - 1), :])

        subgrid_characteristic_time!(
            subgrid_arrays, particles, dt₀, phase_ratios, rheology, thermal, stokes, xci, di
        )
        centroid2particle!(subgrid_arrays.dt₀, xci, dt₀, particles)
        subgrid_diffusion!(
            pT, thermal.T, thermal.ΔT, subgrid_arrays, particles, xvi, di, dt
        )
        end
        # ------------------------------
        # Isotope advection
        velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
        WENO_advection!(d18O, (Vx_v, Vy_v), weno, di, dt)

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
        # update_rock_ratio!(ϕ, phase_ratios, air_phase)
        compute_rock_fraction!(ϕ, chain, xvi, di)

        tensor_invariant!(stokes.τ)

        @show it += 1
        t += dt
        if igg.me == 0
            push!(volume, V_total)
            push!(volume_times, (t / (3600 * 24 * 365.25) / 1.0e3))
            CUDA.@allowscalar pp = [p[3] > 0 || p[4] > 0 for p in phase_ratios.center]
            # pp = [p[3] > 0 || p[4] > 0 for p in phase_ratios.center]
            push!(overpressure, maximum(Array(stokes.P)[pp][Array(ϕ_m)[pp]  .≥ 0.3] .- Array(P_lith)[pp][Array(ϕ_m)[pp]  .≥ 0.3]))
            push!(overpressure_t, t / (3600 * 24 * 365.25) / 1.0e3)
        end

        if it == 1
            stokes.EII_pl .= 0.0
        end

        if plotting
            # Data I/O and plotting ---------------------
            if it == 1 || rem(it, 1) == 0
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
                        d18O = Array(d18O),
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
                        t = round(t / (1.0e3 * 3600 * 24 * 365.25); digits = 3)
                    )
                    save_particles(particles, pPhases; conversion = 1.0e3, fname = joinpath(vtk_dir, "particles_" * lpad("$it", 6, "0")))
                    save_marker_chain(joinpath(vtk_dir, "chain_" * lpad("$it", 6, "0")), xvi[1] ./ 1.0e3, Array(chain.h_vertices) ./ 1.0e3)
                end

                # Make particles plottable
                p = particles.coords
                ppx, ppy = p
                pxv = ppx.data[:] ./ 1.0e3
                pyv = ppy.data[:] ./ 1.0e3
                clr = pPhases.data[:]
                # clr      = pT.data[:]
                idxv = particles.index.data[:]

                chain_x = chain.coords[1].data[:] ./ 1.0e3
                chain_y = chain.coords[2].data[:] ./ 1.0e3

                scatter!(pxv, pyv, color = clr, markersize = 10)
                scatter!(Array(chain_x), Array(chain_y), color = :red, markersize = 3)

                # Make Makie figure
                ar = DataAspect()
                fig = Figure(size = (1200, 900), title = "t = $t")
                ax1 = Axis(fig[1, 1], aspect = ar, title = "T [C]  (t=$(round(t / (1.0e3 * 3600 * 24 * 365.25); digits = 3)) Kyrs)")
                ax2 = Axis(fig[2, 1], aspect = ar, title = "Vy [cm/yr]")
                # ax2 = Axis(fig[2,1], aspect = ar, title = "Phase")
                ax3 = Axis(fig[1, 3], aspect = ar, title = "τII [MPa]")
                # ax4 = Axis(fig[2,3], aspect = ar, title = "log10(εII)")
                ax4 = Axis(fig[2, 3], aspect = ar, title = "log10(η)")
                ax5 = Axis(fig[3, 1], aspect = ar, title = "EII_pl")
                ax6 = Axis(fig[3, 3], aspect = ar, title = "ΔP from P_lith")
                # ax6 = Axis(fig[3,3], aspect = ar, title = "Melt fraction ϕ")
                # Plot temperature
                h1 = heatmap!(ax1, xvi[1] .* 1.0e-3, xvi[2] .* 1.0e-3, Array(thermal.T[2:(end - 1), :] .- 273), colormap = :batlow)
                # Plot velocity
                h2 = heatmap!(ax2, xvi[1] .* 1.0e-3, xvi[2] .* 1.0e-3, ustrip.(uconvert.(u"cm/yr", Array(stokes.V.Vy)u"m/s")), colormap = :vik) #, colorrange= (-(ustrip.(1u"cm/yr")), (ustrip(1u"cm/yr"))))
                scatter!(ax2, Array(chain_x), Array(chain_y), color = :red, markersize = 3)
                # Plot 2nd invariant of stress
                # h3  = heatmap!(ax3, xci[1].*1e-3, xci[2].*1e-3, Array(log10.(stokes.ε_pl.II)) , colormap=:batlow)
                h3 = heatmap!(ax3, xci[1] .* 1.0e-3, xci[2] .* 1.0e-3, Array(stokes.τ.II) ./ 1.0e6, colormap = :batlow)
                # Plot effective viscosity
                # h4  = heatmap!(ax4, xci[1].*1e-3, xci[2].*1e-3, Array(log10.(stokes.ε.II)) , colormap=:lipari)
                h4 = heatmap!(ax4, xci[1] .* 1.0e-3, xci[2] .* 1.0e-3, Array(log10.(stokes.viscosity.η_vep)), colorrange = log10.(viscosity_cutoff), colormap = :batlow)
                # h4 = heatmap!(ax4, xci[1] .* 1.0e-3, xci[2] .* 1.0e-3, Array(log10.(η_eff)), colorrange = log10.(viscosity_cutoff), colormap = :batlow)
                h5 = heatmap!(ax5, xci[1] .* 1.0e-3, xci[2] .* 1.0e-3, Array(stokes.EII_pl), colormap = :batlow)
                contour!(ax5, xci[1] .* 1.0e-3, xci[2] .* 1.0e-3, Array(ϕ_m), levels = [0.5, 0.75, 1.0], color = :white, linewidth = 1.5, labels=true)

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

                ## Plot initial T and P profile
                # fig = let
                #     Yv = [y for x in xvi[1], y in xvi[2]][:]
                #     Y = [y for x in xci[1], y in xci[2]][:]
                #     fig = Figure(; size=(1200, 900))
                #     ax1 = Axis(fig[1, 1]; aspect=2 / 3, title="T")
                #     ax2 = Axis(fig[1, 2]; aspect=2 / 3, title="Pressure")
                #     scatter!(
                #         ax1,
                #         Array(thermal.T[2:(end - 1), :][:].-273.15),
                #         Yv./1e3,
                #     )
                #     lines!(
                #         ax2,
                #         Array(stokes.P[:]./1e6),
                #         Y./1e3,
                #     )
                #     hideydecorations!(ax2)
                #     # save(joinpath(figdir, "thermal_profile_$it.png"), fig)
                #     fig
                # end

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
                        ax1 = Axis(
                            fig[1, 1]; title = "VEI over Time - Eruption total = $eruption_counter", xlabel = "Time [Kyrs]", ylabel = "VEI",
                            titlesize = 40,
                            yticklabelsize = 25,
                            xticklabelsize = 25,
                            xlabelsize = 25,
                            ylabelsize = 25
                        )
                        ax2 = Axis(
                            fig[1, 1], yaxisposition = :right, ylabel = "Erupted volume [km³]", yscale = log10,
                            yticklabelsize = 25,
                            xticklabelsize = 25,
                            xlabelsize = 25,
                            ylabelsize = 25
                        )
                        scatterlines!(
                            ax1, eruption_times, VEI_array,
                            color = VEI_array,
                            colormap = :lipari10,
                            markersize = VEI_array * 5
                        )
                        scatterlines!(
                            ax2, eruption_times, (round.(ustrip.(uconvert.(u"km^3", (erupted_volume)u"m^3")); digits = 5)),
                            color = :blue,
                            markersize = 5
                        )
                        ylims!(ax1, 0, 8.5)
                        ylims!(ax2, 0.00001, 1100)
                        hidexdecorations!(ax2)
                        fig
                        save(joinpath(figdir, "eruption_data.png"), fig)
                        # save(joinpath(figdir, "eruption_data.svg"), fig)
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
                    scatterlines!(ax1, volume_times, (round.(ustrip.(uconvert.(u"km^3", (volume)u"m^3")); digits = 5)), color = :blue, markersize = 5)
                    fig1
                    save(joinpath(figdir, "volume_change.png"), fig1)
                    # save(joinpath(figdir, "volume_change.svg"), fig1)
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
                    # save(joinpath(figdir, "Overpressure.svg"), fig1)
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
do_vtk = true # set to true to generate VTK files for ParaView

conduit, depth, radius, ar, extension, fric_angle = parse.(Float64, ARGS[1:end])

# figdir is defined as Systematics_depth_radius_ar_extension
figdir   = "Systematics/Caldera2D_$(today())_strong_diabase_no_softening_$(depth)_$(radius)_$(ar)_$(extension)_$(fric_angle)"
# figdir = "Systematics/Caldera2D_$(today())"
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
end
# ----------------------------------------------------

open(joinpath(checkpoint, "setup_args.txt"), "w") do io
    println(io, "conduit: $conduit")
    println(io, "depth: $depth")
    println(io, "radius: $radius")
    println(io, "aspect ratio (ar): $ar")
    println(io, "extension: $extension")
    println(io, "friction angle: $fric_angle")
end


li, origin, phases_GMG, T_GMG, T_bg, _, V_total, V_eruptible, layers, air_phase = setup2D(
    nx + 1, ny + 1;
    sticky_air = 4.0e0,
    dimensions = (40.0e0, 20.0e0), # extent in x and y in km
    flat = false, # flat or volcano cone
    chimney = false, # conduit or not
    layers = 3, # number of layers
    volcano_size = (3.0e0, 7.0e0),    # height, radius
    conduit_radius = 1.0e-2, # radius of the conduit
    chamber_T = 1050.0e0, # temperature of the chamber
    chamber_depth  = depth, # depth of the chamber
    chamber_radius = radius, # radius of the chamber
    aspect_x       = ar, # aspect ratio of the chamber
)

igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI = true, select_device = false)...)
else
    igg
end
# extension = 0.0
# cutoff_visc = (1.0e17, 1.0e23)
# fric_angle = 30.0e0 # friction angle in degrees
main(li, origin, phases_GMG, T_GMG, T_bg, igg; figdir = figdir, nx = nx, ny = ny, do_vtk = do_vtk, fric_angle = fric_angle, extension = extension, cutoff_visc = (1.0e17, 1.0e23), V_total = V_total, V_eruptible = V_eruptible, layers = layers, air_phase = air_phase, progressiv_extension = progressiv_extension, plotting = plotting);
