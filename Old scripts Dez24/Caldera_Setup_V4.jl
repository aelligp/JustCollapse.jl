const isCUDA = false

@static if isCUDA
    using CUDA
    # CUDA.allowscalar(true)
end

using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO

const backend_JR = @static if isCUDA
    CUDABackend          # Options: CPUBackend, CUDABackend, AMDGPUBackend
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

const backend = @static if isCUDA
    CUDABackend        # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

using Printf, Statistics, LinearAlgebra, GeoParams, CairoMakie
using MPI
import GeoParams.Dislocation
using GeophysicalModelGenerator, WriteVTK, JLD2
using Dates

# -----------------------------------------------------
include("CalderaModelSetup.jl")
include("CalderaRheology.jl")
# -----------------------------------------------------
## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------

import ParallelStencil.INDICES
const idx_j = INDICES[2]
macro all_j(A)
    esc(:($A[$idx_j]))
end

function copyinn_x!(A, B)
    @parallel function f_x(A, B)
        @all(A) = @inn_x(B)
        return nothing
    end

    @parallel f_x(A, B)
end

function BC_velo!(Vx,Vy, εbg, xvi, lx,ly)
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
    @parallel (1:nx, 1:(ny - 2)) pure_shear_x!(Vx, εbg,lx)
    nx, ny = size(Vy)
    @parallel (1:(nx - 2), 1:ny) pure_shear_y!(Vy,εbg, ly)

    return nothing
end

function BC_displ!(Ux,Uy, εbg, xvi, lx,ly, dt)
    xv, yv = xvi


    @parallel_indices (i, j) function pure_shear_x!(Ux, εbg, lx,dt)
        xi = xv[i]
        Ux[i, j + 1] = εbg * (xi - lx * 0.5) * dt
        return nothing
    end

    @parallel_indices (i, j) function pure_shear_y!(Uy, εbg, ly, dt)
        yi = yv[j]
        Uy[i + 1, j] = abs(yi) * εbg * dt
        return nothing
    end


    nx, ny = size(Ux)
    @parallel (1:nx, 1:(ny - 2)) pure_shear_x!(Ux, εbg, lx,dt)
    nx, ny = size(Uy)
    @parallel (1:(nx - 2), 1:ny) pure_shear_y!(Uy, εbg, ly, dt)

    return nothing
end

@parallel_indices (i, j) function init_P!(P, ρg, z)
    @all(P) = abs(@all(ρg) * (@all_j(z))) * <((@all_j(z)), 0.0)
    return nothing
end

function phase_change!(phases, particles)
    ni = size(phases)
    @parallel_indices (I...) function _phase_change!(phases, px, py, index)

        @inbounds for ip in cellaxes(phases)
            #quick escape
            @index(index[ip, I...]) == 0 && continue

            x = @index px[ip,I...]
            y = (@index py[ip,I...])
            phase_ij = @index phases[ip, I...]
            if y > 0.0 && (phase_ij  == 2.0 || phase_ij  == 3.0)
                @index phases[ip, I...] = 4.0
            end
        end
        return nothing
    end

    @parallel (@idx ni) _phase_change!( phases, particles.coords..., particles.index)
end

function phase_change!(phases, particles, chain, air_phase; init=true)
    ni = size(phases)
    @parallel_indices (I...) function _phase_change!(phases, px, py , index, chain, air_phase, init)

        @inbounds for ip in cellaxes(phases)
            # quick escape
            @index(index[ip, I...]) == 0 && continue

            x = @index px[ip, I...]
            y = @index py[ip, I...]
            chain_x = @index chain.coords[1][ip, I[1]]
            chain_y = @index chain.coords[2][ip, I[1]]


            phase_ij = @index phases[ip, I...]

            if init== true && (phase_ij > air_phase || phase_ij < air_phase) && y > chain_y
                @index phases[ip, I...] = Float64(air_phase)
            elseif phase_ij == 1.0 && y > chain_y
                @index phases[ip, I...] = Float64(air_phase)
            elseif phase_ij == air_phase && y < chain_y
                @index phases[ip, I...] = 1.0
            end
        end
        return nothing
    end

    @parallel (@idx ni) _phase_change!(phases, particles.coords..., particles.index, chain, air_phase, init)
end


function circular_perturbation!(T, δT, max_temperature, xc_anomaly, yc_anomaly, r_anomaly, xvi)

    @parallel_indices (i, j) function _circular_perturbation!(T, δT, max_temperature, xc_anomaly, yc_anomaly, r_anomaly, x, y)
        @inbounds if  ((x[i] - xc_anomaly)^2 + (y[j] - yc_anomaly)^2 ≤ r_anomaly^2)
            new_temperature = T[i+1, j] * (δT / 100 + 1)
            T[i+1, j] = new_temperature > max_temperature ? max_temperature : new_temperature
        end
        return nothing
    end

    nx, ny = size(T)

    @parallel (1:nx-2, 1:ny) _circular_perturbation!(T, δT, max_temperature, xc_anomaly, yc_anomaly, r_anomaly, xvi...)
end

function new_thermal_anomaly!(phases, particles, xc_anomaly, yc_anomaly, r_anomaly)
    ni = size(phases)

    @parallel_indices (I...) function new_anomlay_particles(phases, px, py, index, xc_anomaly, yc_anomaly, r_anomaly)
        @inbounds for ip in cellaxes(phases)
            @index(index[ip, I...]) == 0 && continue

            x = @index px[ip, I...]
            y = @index py[ip, I...]

            # thermal anomaly - circular
            if ((x - xc_anomaly)^2 + (y - yc_anomaly)^2 ≤ r_anomaly^2)
                @index phases[ip, I...] = 3.0
            end
        end
        return nothing
    end
    @parallel (@idx ni) new_anomlay_particles(phases, particles.coords..., particles.index, xc_anomaly, yc_anomaly, r_anomaly)
end

function plot_particles(particles, pPhases)
    p = particles.coords
    # pp = [argmax(p) for p in phase_ratios.center] #if you want to plot it in a heatmap rather than scatter
    ppx, ppy = p
    # pxv = ustrip.(dimensionalize(ppx.data[:], km, CharDim))
    # pyv = ustrip.(dimensionalize(ppy.data[:], km, CharDim))
    pxv = ppx.data[:]
    pyv = ppy.data[:]
    clr = pPhases.data[:]
    # clr = pϕ.data[:]
    idxv = particles.index.data[:]
    f,ax,h=scatter(Array(pxv[idxv]), Array(pyv[idxv]), color=Array(clr[idxv]), colormap=:roma, markersize=1)
    Colorbar(f[1,2], h)
    f
end

# function extract_topography2D!(grid, air_phase)

#         phases = grid.fields.Phases
#         topography = zeros(Float64, size(grid.z.val[:,1,1]))
#         for i in eachindex(topography)
#             ind = findfirst(x -> x == air_phase, phases[i,1,:])
#             topography[i] = grid.z.val[i,1,ind] - 0.5 * (grid.z.val[i,1,ind] - grid.z.val[i,1,ind-1])
#         end

#     return topography
# end

function smooth(data::Vector{Float64}, window_size::Int)
    smoothed_data = similar(data)
    half_window = div(window_size, 2)
    n = length(data)

    for i in 1:n
        start_idx = max(1, i - half_window)
        end_idx = min(n, i + half_window)
        smoothed_data[i] = mean(data[start_idx:end_idx])
    end

    return smoothed_data
end

function extract_topo_particles2D!(particles, air_phase, phase_ratios::JustPIC.PhaseRatios; smoothing_factor=7)
    ni = size(phase_ratios.center)
    topo = @fill(NaN, ni[1])
    py = particles.coords[2]

    for i in 1:ni[1]
        for j in 1:ni[2]
            phase_i = phase_ratios.center[i, j]
            if phase_i[air_phase] > 0.0
                if isnan(topo[i])
                    y = py[i, j][1]  # Extract the first element of the vector
                    topo[i] = y
                end
                break
            end
        end
    end

    # Apply smoothing filter
    topo = smooth(topo, smoothing_factor)

    return topo
end

# topo = extract_topo_particles2D!(particles, air_phase, phase_ratios)

function extract_topography3D!(grid, air_phase)
    phases = grid.fields.Phases
    topography = zeros(Float64, size(grid.z.val, 1), size(grid.z.val, 2))

    for i in axes(topography, 1)
        for j in axes(topography, 2)
            ind = findfirst(x -> x == air_phase, phases[i, j, :])
            if ind !== nothing && ind > 1
                topography[i, j] = grid.z.val[i, j, ind] - 0.5 * (grid.z.val[i, j, ind] - grid.z.val[i, j, ind-1])
            end
        end
    end

    return topography
end
# [...]


@views function Caldera_2D(li_GMG, origin_GMG, phases_GMG, T_GMG, Grid,igg;
    figname="Caldera2D",
    nx=64, ny=64,
    do_vtk=false, sticky_air=5e0,
    εbg_dim= 1e-15 /s,
    shear=false,
    x_global = range(0e0, 40, nx_g()),
    z_global = range(-25e0, sticky_air, ny_g()),
    )

    #-----------------------------------------------------
    # USER INPUTS
    #-----------------------------------------------------
    # Characteristic lengths for nondimensionalisation
    CharDim         = GEO_units(; length=40km, viscosity=1e20Pa * s, temperature=1000C)
    #-----------------------------------------------------
    # Define model to be run
    # nt              = 500                       # number of timesteps
    # IO --------------------------------------------------
    # if it does not exist, make folder where figures are stored
    figdir = "./fig2D/$figname/"
    checkpoint = joinpath(figdir, "checkpoint")
    if do_vtk
        vtk_dir      = joinpath(figdir,"vtk")
        take(vtk_dir)
    end
    take(figdir)

    # -----------------------------------------------------
    # Set up the JustRelax model
    # -----------------------------------------------------
    sticky_air      = nondimensionalize(sticky_air*km, CharDim)             # nondimensionalize sticky air
    lx              = nondimensionalize(li_GMG[1]*km, CharDim)              # nondimensionalize domain length in x-direction
    lz              = nondimensionalize(li_GMG[end]*km, CharDim)            # nondimensionalize domain length in y-direction
    li              = (lx, lz)                                              # domain length in x- and y-direction
    ni              = (nx, ny)                                              # number of grid points in x- and y-direction
    di              = @. li / (nx_g(),nz_g())                                            # grid spacing in x- and y-direction
    origin          = ntuple(Val(2)) do i
        nondimensionalize(origin_GMG[i] * km,CharDim)                       # origin coordinates of the domain
    end
    grid         = Geometry(ni, li; origin=origin)
    (; xci, xvi) = grid                                                     # nodes at the center and vertices of the cells

    εbg          = nondimensionalize((εbg_dim * shear), CharDim)                      # background strain rate
    perturbation_C = @rand(ni...);                                          # perturbation of the cohesion

    # Physical Parameters
    rheology     = init_rheology(CharDim; is_compressible=true, linear = true)
    rheology_incomp = init_rheology(CharDim; is_compressible=false, linear = true)
    cutoff_visc  = nondimensionalize((1e15Pa*s, 1e24Pa*s),CharDim)
    κ            = (4 / (rheology[5].HeatCapacity[1].Cp.val * rheology[1].Density[1].ρsolid.ρ0.val))                                 # thermal diffusivity
    # κ            = (4 / (rheology[1].HeatCapacity[1].Cp.Cp.val * rheology[1].Density[1].ρsolid.ρ0.val))                                 # thermal diffusivity
    # κ            = (4 / (rheology[1].HeatCapacity[1].Cp.val * rheology[1].Density[1].ρ0.val))                                 # thermal diffusivity
    # κ            = (4 / (rheology[2].HeatCapacity[1].Cp.Cp.val * rheology[2].Density[1].ρ0.val))                                 # thermal diffusivity
    dt           = dt_diff = 0.5 * min(di...)^2 / κ / 2.01

    # Initialize particles ----------------------------------
    nxcell           = 60
    max_xcell        = 80
    min_xcell        = 40
    particles        = init_particles(backend, nxcell, max_xcell, min_xcell, xvi...);

    subgrid_arrays   = SubgridDiffusionCellArrays(particles);
    # velocity grids
    grid_vxi         = velocity_grids(xci, xvi, di);
    # temperature
    pT, pPhases = init_cell_arrays(particles, Val(2));
    # particle fields for the stress rotation
    pτ  = pτxx, pτyy, pτxy        = init_cell_arrays(particles, Val(3)) # stress
    # pτ_o = pτxx_o, pτyy_o, pτxy_o = init_cell_arrays(particles, Val(3)) # old stress
    pω   = pωxy,                  = init_cell_arrays(particles, Val(1)) # vorticity
    particle_args                 = (pT, pPhases, pτ..., pω...)
    particle_args_reduced         = (pT, pτ..., pω...)

    # Assign material phases --------------------------
    phases_dev   = PTArray(backend_JR)(phases_GMG)
    phase_ratios = PhaseRatios(backend, length(rheology), ni);
    init_phases!(pPhases, phases_dev, particles, xvi)
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)

    update_cell_halo!(particles.coords..., particle_args...);
    update_cell_halo!(particles.index)

    # RockRatios for variational stokes
    air_phase   = 5
    ϕ_R         = RockRatio(backend_JR, ni)
    update_rock_ratio!(ϕ_R, phase_ratios, (phase_ratios.Vx, phase_ratios.Vy), air_phase)
    topo = extract_topo_particles2D!(particles, air_phase, phase_ratios; smoothing_factor=7)

    chain             = init_markerchain(backend, nxcell, min_xcell, max_xcell, xvi[1], topo);
    phase_change!(pPhases, particles, chain, air_phase; init=true)
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
    update_cell_halo!(particles.coords..., particle_args...);
    update_cell_halo!(particles.index)
    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes    = StokesArrays(backend_JR, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ=1e-4, Re = 3e0, r=0.7, CFL = 0.9 / √2.1) # Re=3π, r=0.7

    # Boundary conditions of the flow
    flow_bcs = VelocityBoundaryConditions(;
        free_slip=(left=true, right=true, top=true, bot=true),
        free_surface=true,
    )
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...) # update halo cells

    # -----------------------------------------------------

    # THERMAL --------------------------------------------
    Tsurf           = nondimensionalize(273K,CharDim)
    Tbot            = nondimensionalize((700+273)*K,CharDim)
    thermal         = ThermalArrays(backend_JR, ni)
    @views thermal.T[2:end-1, :] .= PTArray(backend_JR)(nondimensionalize(T_GMG.*K, CharDim)) # temperature is in kelvin now !!!
    thermal_bc      = TemperatureBoundaryConditions(;
        no_flux     = (left = true, right = true, top = false, bot = false),
    )
    thermal_bcs!(thermal, thermal_bc)
    temperature2center!(thermal)
    # -----------------------------------------------------

    # Melt Fraction
    ϕ = @zeros(ni...)

    # Buoyancy force
    ρg= ntuple(_ -> @zeros(ni...), Val(2))           # ρg[1] is the buoyancy force in the x direction, ρg[2] is the buoyancy force in the y direction
    for _ in 1:5
        compute_ρg!(ρg[end], phase_ratios, rheology, (T=thermal.Tc, P=stokes.P))
        @parallel (@idx ni) init_P!(stokes.P, ρg[2], xci[2])
        compute_melt_fraction!(
            ϕ, phase_ratios, rheology, (T=thermal.Tc, P=stokes.P)
        )
    end

    args0 = (; ϕ=ϕ, T=thermal.Tc, P=stokes.P, dt=dt, perturbation_C = perturbation_C)

    compute_viscosity!(stokes, phase_ratios, args0, rheology, air_phase, cutoff_visc)

    pt_thermal = PTThermalCoeffs(
        backend_JR, rheology, phase_ratios, args0, dt, ni, di, li; ϵ=1e-5, CFL=0.98 / √2.1
    )
    # MPI ------------------------------------------------
    # global array
    nx_v         = (nx - 2) * igg.dims[1]
    ny_v         = (ny - 2) * igg.dims[2]
    # center
    P_v          = zeros(nx_v, ny_v)
    τII_v        = zeros(nx_v, ny_v)
    η_v          = zeros(nx_v, ny_v)
    η_vep_v      = zeros(nx_v, ny_v)
    εII_v        = zeros(nx_v, ny_v)
    εII_pl_v     = zeros(nx_v, ny_v)
    EII_pl_v     = zeros(nx_v, ny_v)
    ϕ_v          = zeros(nx_v, ny_v)
    ρg_v         = zeros(nx_v, ny_v)
    phases_c_v   = zeros(nx_v, ny_v)
    #center nohalo
    P_nohalo     = zeros(nx-2, ny-2)
    τII_nohalo   = zeros(nx-2, ny-2)
    η_nohalo     = zeros(nx-2, ny-2)
    η_vep_nohalo = zeros(nx-2, ny-2)
    εII_nohalo   = zeros(nx-2, ny-2)
    εII_pl_nohalo= zeros(nx-2, ny-2)
    EII_pl_nohalo= zeros(nx-2, ny-2)
    ϕ_nohalo     = zeros(nx-2, ny-2)
    ρg_nohalo    = zeros(nx-2, ny-2)
    phases_c_nohalo = zeros(nx-2, ny-2)
    #vertex
    Vxv_v        = zeros(nx_v, ny_v)
    Vyv_v        = zeros(nx_v, ny_v)
    T_v          = zeros(nx_v, ny_v)
    #vertex nohalo
    Vxv_nohalo   = zeros(nx-2, ny-2)
    Vyv_nohalo   = zeros(nx-2, ny-2)
    T_nohalo     = zeros(nx-2, ny-2)

    xci_v        = LinRange(minimum(x_global).*1e3, maximum(x_global).*1e3, nx_v), LinRange(minimum(z_global).*1e3, maximum(z_global).*1e3, ny_v)

    @copy stokes.P0 stokes.P
    @copy thermal.Told thermal.T

    t, it      = 0.0, 0
    interval   = 1.0
    local Vx, Vy
    Vx = @zeros(ni...)
    Vy = @zeros(ni...)
    τxx_v = @zeros(ni.+1...)
    τyy_v = @zeros(ni.+1...)

    T_buffer    = @zeros(ni.+1)
    Told_buffer = similar(T_buffer)
    dt₀         = similar(stokes.P)
    for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
        copyinn_x!(dst, src)
    end
    grid2particle!(pT, xvi, T_buffer, particles)

    ## Plot initial T and P profile
    fig = let
        Yv = [y for x in xvi[1], y in xvi[2]][:]
        Y = [y for x in xci[1], y in xci[2]][:]
        fig = Figure(; size=(1200, 900))
        ax1 = Axis(fig[1, 1]; aspect=2 / 3, title="T")
        ax2 = Axis(fig[1, 2]; aspect=2 / 3, title="Pressure")
        scatter!(
            ax1,
            Array(ustrip.(dimensionalize(thermal.T[2:(end - 1), :][:], K, CharDim))),
            ustrip.(dimensionalize(Yv, km, CharDim)),
        )
        lines!(
            ax2,
            Array(ustrip.(dimensionalize(stokes.P[:], MPa, CharDim))),
            ustrip.(dimensionalize(Y, km, CharDim)),
        )
        hideydecorations!(ax2)
        save(joinpath(figdir, "initial_profile.png"), fig)
        fig
    end
    ## Do a incompressible stokes solve, to get a better initial guess for the compressible stokes
    ## solver. And after that apply the extension/compression BCs. This is done to avoid the
    ## incompressible stokes solver to blow up.
    igg.me ==0 && println("Starting incompressible stokes solve")

    solve_VariationalStokes!(
        stokes,
        pt_stokes,
        di,
        flow_bcs,
        ρg,
        phase_ratios,
        ϕ_R,
        rheology_incomp,
        args0,
        dt*0.1,
        igg;
        kwargs = (;
            iterMax          = 50e3,#250e3,
            # free_surface     = false,
            nout             = 2e3,#5e3,
            viscosity_cutoff = cutoff_visc,
        )
    )
    # tensor_invariant!(stokes.ε)
    # heatdiffusion_PT!(
    #     thermal,
    #     pt_thermal,
    #     thermal_bc,
    #     rheology_incomp,
    #     args0,
    #     dt,
    #     di;
    #     kwargs =(;
    #         igg     = igg,
    #         phase   = phase_ratios,
    #         iterMax = 150e3,
    #         nout    = 1e3,
    #         verbose = true,
    #     )
    # )

    if shear == true
        BC_velo!(@velocity(stokes)..., εbg, xvi,lx,lz)
        flow_bcs = VelocityBoundaryConditions(;
            free_slip   =(left=true, right=true, top=true, bot=true),
            free_surface= true,
        )
        flow_bcs!(stokes, flow_bcs) # apply boundary conditions
        update_halo!(@velocity(stokes)...) # update halo cells
    end
    igg.me == 0 && println("Starting main loop")

    while it < 150 #nt

        # interpolate fields from particle to grid vertices
        particle2grid!(T_buffer, pT, xvi, particles)
        @views T_buffer[:, end]      .= Tsurf
        @views T_buffer[:, 1]        .= Tbot
        @views thermal.T[2:end-1, :] .= T_buffer
        thermal_bcs!(thermal, thermal_bc)
        temperature2center!(thermal)
        vertex2center!(thermal.ΔTc, thermal.ΔT[2:end-1, :])

        particle2centroid!(stokes.τ.xx, pτxx, xci, particles)
        particle2centroid!(stokes.τ.yy, pτyy, xci, particles)
        particle2grid!(stokes.τ.xy, pτxy, xvi, particles)

        # if DisplacementFormulation == true
        #     BC_displ!(@displacement(stokes)..., εbg, xvi,lx,lz,dt)
        #     flow_bcs!(stokes, flow_bcs) # apply boundary conditions
        #     BC_velo!(@velocity(stokes)..., εbg, xvi,lx,lz)
        # end

        args = (; ϕ=ϕ, T=thermal.Tc, P=stokes.P, dt=dt, ΔTc=thermal.ΔTc, perturbation_C = perturbation_C)
        ## Stokes solver -----------------------------------
        t_stokes = @elapsed solve_VariationalStokes!(
            stokes,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            phase_ratios,
            ϕ_R,
            rheology,
            args,
            dt,
            igg;
            kwargs = (;
                iterMax          = 10e3,
                nout             = 2e3,
                viscosity_cutoff = cutoff_visc,
                air_phase        = air_phase
            )
        )

        center2vertex!(τxx_v, stokes.τ.xx)
        center2vertex!(τyy_v, stokes.τ.yy)
        centroid2particle!(pτxx , xci, stokes.τ.xx, particles)
        centroid2particle!(pτyy , xci, stokes.τ.yy, particles)
        grid2particle!(pτxy, xvi, stokes.τ.xy, particles)
        rotate_stress_particles!(pτ, pω, particles, dt)

        igg.me == 0 && println("Stokes solver time             ")
        igg.me == 0 && println("   Total time:      $t_stokes s")
        tensor_invariant!(stokes.ε)
        tensor_invariant!(stokes.ε_pl)
        dt= compute_dt(stokes, di, dt_diff, igg) #/ 9.81

        ## Save the checkpoint file before a possible thermal solver blow up
        checkpointing_jld2(joinpath(checkpoint, "thermal"), stokes, thermal, t, dt, igg)

        # ------------------------------
        # compute_shear_heating!(
        #     thermal,
        #     stokes,
        #     phase_ratios,
        #     rheology, # needs to be a tuple
        #     dt,
        # )
        # Thermal solver ---------------
        heatdiffusion_PT!(
            thermal,
            pt_thermal,
            thermal_bc,
            rheology,
            args,
            dt,
            di;
            kwargs =(;
                igg     = igg,
                phase   = phase_ratios,
                iterMax = 50e3,
                nout    = 1e3,
                verbose = true,
            )
        )

        for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
            copyinn_x!(dst, src)
        end
        subgrid_characteristic_time!(
            subgrid_arrays, particles, dt₀, phase_ratios, rheology, thermal, stokes, xci, di
        )
        centroid2particle!(subgrid_arrays.dt₀, xci, dt₀, particles)
        subgrid_diffusion!(
            pT, thermal.T, thermal.ΔT, subgrid_arrays, particles, xvi,  di, dt
        )

        compute_melt_fraction!(
            ϕ, phase_ratios, rheology, (T=thermal.Tc, P=stokes.P)
        )

        # Advection --------------------
        # advect particles in space
        # advection!(particles, RungeKutta2(), @velocity(stokes), grid_vxi, dt)
        # advection_LinP!(particles, RungeKutta2(), @velocity(stokes), grid_vxi, dt)
        advection_MQS!(particles, RungeKutta2(), @velocity(stokes), grid_vxi, dt)
        advect_markerchain!(chain, RungeKutta2(), @velocity(stokes), grid_vxi, dt)
        # update halos
        update_cell_halo!(particles.coords..., particle_args...);
        update_cell_halo!(particles.index)
        # advect particles in memory
        move_particles!(particles, xvi, particle_args)

        # check if we need to inject particles
        # inject_particles_phase!(particles, pPhases, (pT, ), (T_buffer, ), xvi)
        inject_particles_phase!(
            particles,
            pPhases,
            particle_args_reduced,
            (T_buffer, τxx_v, τyy_v, stokes.τ.xy, stokes.ω.xy),
            xvi
        )

        # phase change above the marker chain
        # phase_change!(pPhases, particles, chain, air_phase; init=false)
        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
        update_rock_ratio!(ϕ_R, phase_ratios, (phase_ratios.Vx, phase_ratios.Vy), air_phase)

        @show it += 1
        t += dt

        #MPI gathering
        phase_center = [argmax(p) for p in Array(phase_ratios.center)]
        #centers
        @views P_nohalo     .= Array(stokes.P[2:end-1, 2:end-1]) # Copy data to CPU removing the halo
        @views τII_nohalo   .= Array(stokes.τ.II[2:end-1, 2:end-1]) # Copy data to CPU removing the halo
        @views η_nohalo     .= Array(stokes.viscosity.η[2:end-1, 2:end-1]) # Copy data to CPU removing the halo
        @views η_vep_nohalo .= Array(stokes.viscosity.η_vep[2:end-1, 2:end-1])       # Copy data to CPU removing the halo
        @views εII_nohalo   .= Array(stokes.ε.II[2:end-1, 2:end-1]) # Copy data to CPU removing the halo
        @views εII_pl_nohalo .= Array(stokes.ε_pl.II[2:end-1, 2:end-1]) # Copy data to CPU removing the halo
        @views EII_pl_nohalo.= Array(stokes.EII_pl[2:end-1, 2:end-1]) # Copy data to CPU removing the halo
        @views ϕ_nohalo     .= Array(ϕ[2:end-1, 2:end-1]) # Copy data to CPU removing the halo
        @views ρg_nohalo    .= Array(ρg[end][2:end-1, 2:end-1]) # Copy data to CPU removing the halo
        @views phases_c_nohalo   .= Array(phase_center[2:end-1, 2:end-1])
        gather!(P_nohalo, P_v)
        gather!(τII_nohalo, τII_v)
        gather!(η_nohalo, η_v)
        gather!(η_vep_nohalo, η_vep_v)
        gather!(εII_nohalo, εII_v)
        gather!(εII_pl_nohalo, εII_pl_v)
        gather!(EII_pl_nohalo, EII_pl_v)
        gather!(ϕ_nohalo, ϕ_v)
        gather!(ρg_nohalo, ρg_v)
        gather!(phases_c_nohalo, phases_c_v)
        #vertices
        velocity2center!(Vx, Vy, @velocity(stokes)...)
        @views Vxv_nohalo   .= Array(Vx[2:end-1, 2:end-1]) # Copy data to CPU removing the halo
        @views Vyv_nohalo   .= Array(Vy[2:end-1, 2:end-1]) # Copy data to CPU removing the halo
        gather!(Vxv_nohalo, Vxv_v)
        gather!(Vyv_nohalo, Vyv_v)
        @views T_nohalo     .= Array(thermal.Tc[2:end-1, 2:end-1]) # Copy data to CPU removing the halo
        gather!(T_nohalo, T_v)
        ## Plotting -------------------------------------------------------
        if it == 1 || rem(it, 1) == 0
            if igg.me == 0 && it == 1
                metadata(pwd(), checkpoint, basename(@__FILE__), "CalderaModelSetup.jl", "CalderaRheology.jl")
            end
            checkpointing_jld2(checkpoint, stokes, thermal, t, dt, igg)
            ## Somehow fails to open with load("particles.jld2")
            mktempdir() do tmpdir
                # Save the checkpoint file in the temporary directory
                tmpfname = joinpath(tmpdir, basename(joinpath(checkpoint, "particles.jld2")))
                jldsave(
                    tmpfname;
                    particles=Array(particles),
                    pPhases=Array(pPhases),
                    time=t,
                    timestep=dt,
                )
                # Move the checkpoint file from the temporary directory to the destination directory
                mv(tmpfname, joinpath(checkpoint, "particles.jld2"); force=true)
            end

            t_yrs = dimensionalize(t, yr, CharDim)
            t_Kyrs = t_yrs / 1e3
            t_Myrs = t_Kyrs / 1e3

            xci_dim = ntuple(Val(2)) do i
                ustrip.(dimensionalize(xci_v[i], km, CharDim))
            end


            if igg.me == 0 && do_vtk
                data_c = (;
                    T = ustrip.(dimensionalize(T_v, K, CharDim)),
                    P = ustrip.(dimensionalize(P_v, MPa, CharDim)),
                    τII = ustrip.(dimensionalize(τII_v, MPa, CharDim)),
                    εII = ustrip.(dimensionalize(εII_v, s^-1, CharDim)),
                    εII_pl = ustrip.(dimensionalize(EII_pl_v, s^-1, CharDim)),
                    EII_pl = EII_pl_v,
                    ϕ = ϕ_v,
                    ρ = ustrip.(dimensionalize(ρg_v, kg / m^3 * m / s^2, CharDim))./ 9.81,
                    η = ustrip.(dimensionalize(η_v, Pas, CharDim)),
                    η_vep   = ustrip.(dimensionalize(η_vep_v, Pas, CharDim)),
                    Vx = ustrip.(dimensionalize(Array(Vxv_v), cm / yr, CharDim)),
                    Vy = ustrip.(dimensionalize(Array(Vyv_v), cm / yr, CharDim)),
                    phases = phases_c_v
                )
                velocity = (
                    ustrip.(dimensionalize(Array(Vxv_v), cm / yr, CharDim)),
                    ustrip.(dimensionalize(Array(Vyv_v), cm / yr, CharDim)),
                )
                save_vtk(
                    joinpath(vtk_dir, "vtk_" * lpad("$(it)_$(igg.me)", 6, "0")),
                    xci_dim,
                    data_c,
                    velocity,
                    t=(round.(ustrip.(t_Kyrs); digits=3))
                )
            end


            p = particles.coords
            # pp = [argmax(p) for p in phase_ratios.center] #if you want to plot it in a heatmap rather than scatter
            ppx, ppy = p
            pxv = ustrip.(dimensionalize(ppx.data[:], km, CharDim))
            pyv = ustrip.(dimensionalize(ppy.data[:], km, CharDim))
            clr = pPhases.data[:]
            clrT = pT.data[:]
            idxv = particles.index.data[:]


            if igg.me == 0
                fig = Figure(; size=(2000, 1800), createmissing=true)
                ar = li[1] / li[2]
                # ar = DataAspect()

                ax0 = Axis(
                    fig[1, 1:2];
                    aspect=ar,
                    title="t = $(round.(ustrip.(t_Kyrs); digits=3)) Kyrs",
                    titlesize=50,
                    height=0.0,
                )
                ax0.ylabelvisible = false
                ax0.xlabelvisible = false
                ax0.xgridvisible = false
                ax0.ygridvisible = false
                ax0.xticksvisible = false
                ax0.yticksvisible = false
                ax0.yminorticksvisible = false
                ax0.xminorticksvisible = false
                ax0.xgridcolor = :white
                ax0.ygridcolor = :white
                ax0.ytickcolor = :white
                ax0.xtickcolor = :white
                ax0.yticklabelcolor = :white
                ax0.xticklabelcolor = :white
                ax0.yticklabelsize = 0
                ax0.xticklabelsize = 0
                ax0.xlabelcolor = :white
                ax0.ylabelcolor = :white

                ax1 = Axis(
                    fig[2, 1][1, 1];
                    aspect=ar,
                    title=L"T [\mathrm{K}]",
                    titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,
                )
                ax2 = Axis(
                    fig[2, 2][1, 1];
                    aspect=ar,
                    title=L"\log_{10}(\eta_{vep}) [\mathrm{Pas}]",
                    titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,
                )
                ax3 = Axis(
                    fig[3, 1][1, 1];
                    aspect=ar,
                    title=L"Vy [\mathrm{cm/yr}]",
                    # title=L"\tau_{\textrm{II}} [MPa]",
                    xlabel="Width [km]",
                    titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,
                )
                ax4 = Axis(
                    fig[3, 2][1, 1];
                    aspect=ar,
                    title=L"\log_{10}(\dot{\varepsilon}_{\textrm{II}}) [\mathrm{s}^{-1}]",
                    xlabel="Width [km]",
                    titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,
                )
                ax5 = Axis(
                    fig[4, 1][1, 1];
                    aspect=ar,
                    title=L"Plastic Strain",
                    # title=L"Phases",
                    xlabel="Width [km]",
                    titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,
                )
                ax6 = Axis(
                    fig[4, 2][1, 1];
                    aspect=ar,
                    title=L"\tau_{\textrm{II}} [MPa]",
                    xlabel="Width [km]",
                    titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,
                )

                linkyaxes!(ax1, ax2)
                linkyaxes!(ax3, ax4)
                hidexdecorations!(ax1; grid=false)
                hideydecorations!(ax2; grid=false)
                hidexdecorations!(ax3; grid=false)
                hidexdecorations!(ax2; grid=false)

                @views T_v[ϕ_R.center[2:end-1, 2:end-1] .== 0.0] .= NaN
                @views ρg_v[ϕ_R.center[2:end-1, 2:end-1] .== 0.0] .= NaN
                @views η_vep_v[ϕ_R.center[2:end-1, 2:end-1] .== 0.0] .= NaN
                @views τII_v[ϕ_R.center[2:end-1, 2:end-1] .== 0.0] .= NaN
                @views εII_v[ϕ_R.center[2:end-1, 2:end-1] .== 0.0] .= NaN
                @views EII_pl_v[ϕ_R.center[2:end-1, 2:end-1] .== 0.0] .= NaN
                @views ϕ_v[ϕ_R.center[2:end-1, 2:end-1] .== 0.0] .= NaN
                @views Vyv_v[ϕ_R.center[2:end-1, 2:end-1] .== 0.0] .=NaN


                p1 = heatmap!(ax1, xci_dim..., ustrip.(dimensionalize(T_v, K, CharDim)); colormap=:batlow, colorrange=(000, 1200))
                # scatter!(ax1, x_c, topo[2:end-2])
                contour!(ax1, xci_dim..., ustrip.(dimensionalize(T_v, K, CharDim)); color=:white, levels=600+273:200:1200+273)
                p2 = heatmap!(ax2, xci_dim..., log10.(ustrip.(dimensionalize(η_vep_v, Pas, CharDim))); colormap=:glasgow)#, colorrange= (log10(1e16), log10(1e22)))
                contour!(ax1, xci_dim..., ustrip.(dimensionalize(T_v, K, CharDim)); color=:white, levels=600+273:200:1200+273)
                # scatter!(ax2, x_c, topo[2:end-2])
                p3 = heatmap!(ax3, xci_dim..., ustrip.(dimensionalize(Array(Vyv_v), cm / yr, CharDim)); colormap=:vik)
                # scatter!(ax3, x_v, topo[2:end-1])
                p4 = heatmap!(ax4, xci_dim..., log10.(ustrip.(dimensionalize(εII_v, s^-1, CharDim))); colormap=:glasgow, colorrange= (log10(5e-15), log10(5e-12)))
                # p5 = heatmap!(ax5, x_c, y_c, P_d; colormap=:glasgow)
                p5 = heatmap!(ax5, xci_dim..., EII_pl_v; colormap=:glasgow)
                contour!(ax5, xci_dim..., ustrip.(dimensionalize(T_v, K, CharDim)); color=:white, levels=600+273:200:1200+273, labels = true)
                p6 = heatmap!(ax6, xci_dim..., ustrip.(dimensionalize(τII_v, MPa, CharDim)); colormap=:batlow)
                # scatter!(ax6, x_c, topo[2:end-2])
                Colorbar(
                    fig[2, 1][1, 2], p1; height=Relative(0.7), ticklabelsize=25, ticksize=15
                )
                Colorbar(
                    fig[2, 2][1, 2], p2; height=Relative(0.7), ticklabelsize=25, ticksize=15
                )
                Colorbar(
                    fig[3, 1][1, 2], p3; height=Relative(0.7), ticklabelsize=25, ticksize=15
                )
                Colorbar(
                    fig[3, 2][1, 2], p4; height=Relative(0.7), ticklabelsize=25, ticksize=15
                )
                Colorbar(
                    fig[4, 1][1, 2], p5; height=Relative(0.7), ticklabelsize=25, ticksize=15
                )
                Colorbar(
                    fig[4, 2][1, 2], p6; height=Relative(0.7), ticklabelsize=25, ticksize=15
                )
                rowgap!(fig.layout, 1)
                colgap!(fig.layout, 1)
                colgap!(fig.layout, 1)
                colgap!(fig.layout, 1)
                fig
                figsave = joinpath(figdir, @sprintf("%06d.png", it))
                save(figsave, fig)

                let
                    Y = [y for x in xci_dim[1], y in xci_dim[2]][:]
                    fig = Figure(; size=(1200, 900))
                    ax1 = Axis(fig[1, 1]; aspect=2 / 3, title="T [K]")
                    ax2 = Axis(fig[1, 2]; aspect=2 / 3, title="Pressure")

                    scatter!(
                        ax1,
                        Array(ustrip.(dimensionalize(T_v[:], K, CharDim))),
                        ustrip.(dimensionalize(Y, km, CharDim)),
                    )
                    lines!(
                        ax2,
                        Array(ustrip.(dimensionalize(P_v[:], MPa, CharDim)),),
                        ustrip.(dimensionalize(Y, km, CharDim)),
                    )

                    hideydecorations!(ax2)
                    save(joinpath(figdir, "pressure_profile_$it.png"), fig)
                    fig
                end

                let
                    p = particles.coords
                    # pp = [argmax(p) for p in phase_ratios.center] #if you want to plot it in a heatmap rather than scatter
                    ppx, ppy = p
                    pxv = ustrip.(dimensionalize(ppx.data[:], km, CharDim))
                    pyv = ustrip.(dimensionalize(ppy.data[:], km, CharDim))
                    # pxv = ppx.data[:]
                    # pyv = ppy.data[:]
                    # clr = pPhases.data[:]
                    # clrT = pT.data[:]
                    px = ustrip.(dimensionalize(chain.coords[1].data[:], km, CharDim));
                    py = ustrip.(dimensionalize(chain.coords[2].data[:], km, CharDim));
                    idxv = particles.index.data[:]
                    f,ax,h=scatter(Array(pxv[idxv]), Array(pyv[idxv]), color=Array(clr[idxv]), colormap=:roma, markersize=1)
                    scatter!(ax,ustrip.(dimensionalize(xci[1], km, CharDim)), ustrip.(dimensionalize(topo, km, CharDim)), color=:black, markersize=3)
                    # lines!(px, py, color=:black)#, markersize=1)
                    Colorbar(f[1,2], h)
                    f
                    save(joinpath(figdir, "particles_$it.png"), f)
                end
             end
        end
    end
end

shear           = true                     # specify if you want to use pure shear boundary conditions
εbg_dim         = 5e-15 / s * shear         # specify the background strain rate

do_vtk          = true
ar              = 1 # aspect ratio
n               = 128
nx              = n * ar
ny              = n
igg             = if !(JustRelax.MPI.Initialized())
                    IGG(init_global_grid(nx, ny, 1; init_MPI=true)...)
                  else
                    igg
                  end

# GLOBAL Physical domain ------------------------------------
sticky_air = 5.0
x_global = range(0.0, 40, nx_g());
z_global      = range(-25e0, sticky_air, ny_g());
origin = (x_global[1], z_global[1])
li = (abs(last(x_global)-first(x_global)), abs(last(z_global)-first(z_global)))

ni           = nx, ny           # number of cells
di           = @. li / (nx_g(), ny_g())           # grid steps
grid_global  = Geometry(ni, li; origin = origin)

li_GMG, origin_GMG, phases_GMG, T_GMG, Grid = volcano_setup2D(grid_global.xvi, nx+1,ny+1;
    flat           = false,
    chimney        = true,
    volcano_size   = (3e0, 5e0),
    chamber_T      = 1e3,
    chamber_depth  = 5e0,
    chamber_radius = 2e0,
    aspect_x       = 1.5,
)

figname = "$(today())_V4_variational_stokes_$(nx_g())x$(ny_g())"

Caldera_2D(li_GMG, origin_GMG, phases_GMG, T_GMG, Grid, igg;
           figname=figname, nx=nx, ny=ny, do_vtk=do_vtk, sticky_air=sticky_air,
           εbg_dim=εbg_dim, shear=shear, x_global, z_global)


# # pc = [argmax(p) for p in Array(phase_ratios.center)]
# # pv = [argmax(p) for p in Array(phase_ratios.vertex)]

# # heatmap(xci..., pc)
# # f,ax,h= scatter!(Array(pxv[idxv]), Array(pyv[idxv]), color=Array(clr[idxv]), colormap=:roma, markersize=3)

# # heatmap(pv)


# p = copy(stokes.P)
# p = copy(thermal.Tc)
# p = copy(thermal.Tc)
# p[ϕ_R.center .== 0] .= NaN
# heatmap(p)
# heatmap(stokes.V.Vy, colormap=:vikO)

# heatmap( x_v, y_v, Vy_d; colormap=:vik)
