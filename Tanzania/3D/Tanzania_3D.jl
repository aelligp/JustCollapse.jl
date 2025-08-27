const isCUDA = false
# const isCUDA = true

@static if isCUDA
    using CUDA
    # CUDA.allowscalar(true)
end

using JustRelax, JustRelax.JustRelax3D, JustRelax.DataIO
# import @index

const backend_JR = @static if isCUDA
    CUDABackend          # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustRelax.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

using ParallelStencil, ParallelStencil.FiniteDifferences3D

@static if isCUDA
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end

using JustPIC, JustPIC._3D

# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
const backend = @static if isCUDA
    CUDABackend        # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

using Printf, Statistics, LinearAlgebra, GeoParams, CairoMakie, CellArrays
import GeoParams.Dislocation
using StaticArrays, GeophysicalModelGenerator, WriteVTK, JLD2
using Dates

# -----------------------------------------------------
include("TanzaniaModelSetup.jl")
include("TanzaniaRheology.jl")
# -----------------------------------------------------
## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------

import ParallelStencil.INDICES
const idx_k = INDICES[3]
macro all_k(A)
    esc(:($A[$idx_k]))
end

# Initial pressure profile - not accurate
@parallel function init_P!(P, ρg, z)
    @all(P) = abs(@all(ρg) * @all_k(z)) * <(@all_k(z), 0.0)
    return nothing
end

function BC_velo!(Vx,Vy, Vz, εbg, xvi, lx, ly,lz)
    xv, yv, zv = xvi

    @parallel_indices (i, j, k) function pure_shear_x!(Vx, εbg, lx)
        xi = xv[i]
        Vx[i, j + 1, k + 1] = εbg * (xi - lx * 0.5)
        return nothing
    end

    # @parallel_indices (i, j, k) function pure_shear_y!(Vy, εbg, ly)
    #     yi = yv[j]
    #     Vy[i + 1, j, k+1] = abs(yi) * εbg
    #     return nothing
    # end

    @parallel_indices (i, j, k) function pure_shear_z!(Vz, εbg, ly)
        zi = zv[k]
        Vz[i + 1, j+1, k] = abs(zi) * εbg
        return nothing
    end

    nx, ny, nz = size(Vx)
    @parallel (1:nx, 1:ny - 2, 1:nz - 2) pure_shear_x!(Vx, εbg, lx)
    # nx, ny, nz = size(Vy)
    # @parallel (1:(nx - 2), 1:ny, 1:nz-2) pure_shear_y!(Vy, εbg, ly)
    nx, ny, nz = size(Vz)
    @parallel (1:nx - 2, 1:ny-2, 1:nz) pure_shear_z!(Vz, εbg, lz)

    return nothing
end

function BC_displ!(Ux,Uy, Uz,εbg, xvi, lx,ly, lz, dt)
    xv, yv, zv = xvi

    @parallel_indices (i, j) function pure_shear_x!(Ux, εbg, lx, dt)
        xi = xv[i]
        Ux[i, j + 1] = εbg * (xi - lx * 0.5) * dt
        return nothing
    end

    # @parallel_indices (i, j) function pure_shear_y!(Uy, εbg, ly, dt)
    #     yi = yv[j]
    #     Uy[i + 1, j, k+1] = abs(yi) * εbg * dt
    #     return nothing
    # end


    @parallel_indices (i, j) function pure_shear_z!(Uy, εbg, lz, dt)
        zi = zv[k]
        Uz[i + 1, j+1, k] = abs(zi) * εbg * dt
        return nothing
    end

    nx, ny = size(Ux)
    @parallel (1:nx, 1:ny - 2, 1:nz - 2) pure_shear_x!(Ux, εbg, lx, dt)
    # nx, ny = size(Uy)
    # @parallel (1:(nx - 2), 1:ny, 1:nz-2) pure_shear_y!(Uy, εbg, ly, dt)
    nx, ny = size(Uz)
    @parallel (1:nx - 2, 1:ny-2, 1:nz) pure_shear_z!(Uy, εbg, lz, dt)

    return nothing

end

# [...]


@views function Tanzania_3D(li_GMG, origin_GMG, phases_GMG, T_GMG, εbg_dim, igg; x_global, y_global, z_global, figname=figname, nx=64, ny=64, nz=64, do_vtk=false)

    #-----------------------------------------------------
    # USER INPUTS
    #-----------------------------------------------------
    # Characteristic lengths for nondimensionalisation
    CharDim         = GEO_units(; length=100km, viscosity=1e20Pa * s, temperature=1000C)
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
    # sticky_air      = nondimensionalize(sticky_air*km, CharDim)             # nondimensionalize sticky air
    lx              = nondimensionalize(li_GMG[1]*km, CharDim)              # nondimensionalize domain length in x-direction
    ly              = nondimensionalize(li_GMG[2]*km, CharDim)              # nondimensionalize domain length in x-direction
    lz              = nondimensionalize(li_GMG[end]*km, CharDim)            # nondimensionalize domain length in z-direction
    li              = (lx, ly, lz)                                      # domain length in x-, y- and z-direction
    ni              = (nx, ny, nz)                                      # number of grid points in x-, y- and z-direction
    di           = @. li / (nx_g(), ny_g(), nz_g())                     # grid spacing in x-, y- and z-direction
    origin          = ntuple(Val(3)) do i
        nondimensionalize(origin_GMG[i] * km,CharDim)                       # origin coordinates of the domain
    end
    grid         = Geometry(ni, li; origin=origin)
    (; xci, xvi) = grid                                                     # nodes at the center and vertices of the cells

    εbg          = nondimensionalize(εbg_dim, CharDim)                      # background strain rate
    perturbation_C = @rand(ni...);                                          # perturbation of the cohesion

    # Physical Parameters
    rheology     = init_rheology(CharDim; is_compressible=true, linear = false)
    rheology_incomp = init_rheology(CharDim; is_compressible=false, linear = false)
    cutoff_visc  = nondimensionalize((1e16Pa*s, 1e23Pa*s),CharDim)
    κ            = (4 / (rheology[1].HeatCapacity[1].Cp.Cp.val * rheology[1].Density[1].ρsolid.ρ0.val))                                 # thermal diffusivity
    # κ            = (4 / (rheology[1].HeatCapacity[1].Cp.val * rheology[1].Density[1].ρ0.val))                                 # thermal diffusivity
    # κ            = (4 / (rheology[2].HeatCapacity[1].Cp.Cp.val * rheology[2].Density[1].ρ0.val))                                 # thermal diffusivity
    dt           = dt_diff = 0.5 * min(di...)^2 / κ / 2.01

    # Initialize particles ----------------------------------
    nxcell, max_xcell, min_xcell = 150, 175, 125
    particles        = init_particles(backend, nxcell, max_xcell, min_xcell, xvi, di, ni);
    subgrid_arrays   = SubgridDiffusionCellArrays(particles);
    # velocity grids
    grid_vxi         = velocity_grids(xci, xvi, di);
    # temperature
    pT, pPhases = init_cell_arrays(particles, Val(2));

    pτ                    = StressParticles(particles)
    particle_args         = (pT, pPhases, unwrap(pτ)...)
    particle_args_reduced = (pT,  unwrap(pτ)...)

    particle_args       = (pT, pT0, pPhases);

    # Assign material phases --------------------------
    phases_dev   = PTArray(backend_JR)(phases_GMG)
    phase_ratios = PhaseRatios(backend, length(rheology), ni);
    init_phases!(pPhases, phases_dev, particles, xvi)
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)

    update_cell_halo!(particles.coords..., particle_args...);
    update_cell_halo!(particles.index)

    # RockRatios
    air_phase   = 5
    ϕ           = RockRatio(backend, ni)
    # update_rock_ratio!(ϕ, phase_ratios, air_phase)
    compute_rock_fraction!(ϕ, chain, xvi, di)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes          = StokesArrays(backend_JR, ni)
    pt_stokes       = PTStokesCoeffs(li, di; ϵ=1e-5,  CFL = 0.99 / √3.1) #ϵ=1e-4,  CFL=1 / √2.1 CFL=0.27 / √2.1
    # -----------------------------------------------------
    # THERMAL --------------------------------------------
    thermal         = ThermalArrays(backend_JR, ni)
    thermal.T       .= PTArray(backend_JR)(nondimensionalize(T_GMG.*C, CharDim))
    thermal_bc      = TemperatureBoundaryConditions(;
        no_flux     = (left = true , right = true , top = false, bot = false, front = true , back = true),
    )
    thermal_bcs!(thermal, thermal_bc)
    temperature2center!(thermal)

    args = (; T=thermal.Tc, P=stokes.P, dt=dt,  ΔTc=thermal.ΔTc, perturbation_C = perturbation_C)

    pt_thermal = PTThermalCoeffs(
        backend_JR, rheology, phase_ratios, args, dt, ni, di, li; ϵ=1e-5, CFL=0.98 / √2.1
    )
    # Boundary conditions of the flow
    flow_bcs = VelocityBoundaryConditions(;
        free_slip    = (left = true , right = true , top = true , bot = true , front = true , back = true ),
        no_slip      = (left = false, right = false, top = false, bot = false, front = false, back = false),
    )
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...) # update halo cells

    # Melt Fraction
    ϕ_m = @zeros(ni...)

    # Buoyancy force
    ρg= ntuple(_ -> @zeros(ni...), Val(3))           # ρg[1] is the buoyancy force in the x direction, ρg[2] is the buoyancy force in the y direction

    # MPI ------------------------------------------------
    # global array
    nx_v         = (nx - 2) * igg.dims[1]
    ny_v         = (ny - 2) * igg.dims[2]
    nz_v         = (nz - 2) * igg.dims[3]
    # center
    P_v          = zeros(nx_v, ny_v, nz_v)
    τII_v        = zeros(nx_v, ny_v, nz_v)
    η_v          = zeros(nx_v, ny_v, nz_v)
    η_vep_v      = zeros(nx_v, ny_v, nz_v)
    εII_v        = zeros(nx_v, ny_v, nz_v)
    εII_pl_v     = zeros(nx_v, ny_v, nz_v)
    EII_pl_v     = zeros(nx_v, ny_v, nz_v)
    ϕ_v          = zeros(nx_v, ny_v, nz_v)
    ρg_v         = zeros(nx_v, ny_v, nz_v)
    phases_c_v   = zeros(nx_v, ny_v, nz_v)
    #center nohalo
    P_nohalo     = zeros(nx-2, ny-2, nz-2)
    τII_nohalo   = zeros(nx-2, ny-2, nz-2)
    η_nohalo     = zeros(nx-2, ny-2, nz-2)
    η_vep_nohalo = zeros(nx-2, ny-2, nz-2)
    εII_nohalo   = zeros(nx-2, ny-2, nz-2)
    εII_pl_nohalo= zeros(nx-2, ny-2, nz-2)
    EII_pl_nohalo= zeros(nx-2, ny-2, nz-2)
    ϕ_nohalo     = zeros(nx-2, ny-2, nz-2)
    ρg_nohalo    = zeros(nx-2, ny-2, nz-2)
    phases_c_nohalo = zeros(nx-2, ny-2, nz-2)
    #vertex
    Vxv_v        = zeros(nx_v, ny_v, nz_v)
    Vyv_v        = zeros(nx_v, ny_v, nz_v)
    Vzv_v        = zeros(nx_v, ny_v, nz_v)
    T_v          = zeros(nx_v, ny_v, nz_v)
    #vertex nohalo
    Vxv_nohalo   = zeros(nx-2, ny-2, nz-2)
    Vyv_nohalo   = zeros(nx-2, ny-2, nz-2)
    Vzv_nohalo   = zeros(nx-2, ny-2, nz-2)
    T_nohalo     = zeros(nx-2, ny-2, nz-2)

    xci_v        = LinRange(minimum(x_global).*1e3, maximum(x_global).*1e3, nx_v), LinRange(minimum(y_global).*1e3, maximum(y_global).*1e3, ny_v), LinRange(minimum(z_global).*1e3, maximum(z_global).*1e3, nz_v)
    # -----------------------------------------------------

    args = (; ϕ=ϕ_m, T=thermal.Tc, P=stokes.P, dt=dt, ΔTc=thermal.ΔTc, perturbation_C = perturbation_C)

    for _ in 1:5
        compute_ρg!(ρg, phase_ratios, rheology, (T=thermal.Tc, P=stokes.P))
        @parallel init_P!(stokes.P, ρg[3], xci[3])
    end

    compute_melt_fraction!(
        ϕ_m, phase_ratios, rheology, (T=thermal.T, P=stokes.P)
    )
    compute_viscosity!(stokes, phase_ratios, args, rheology, air_phase, cutoff_visc)

    @copy stokes.P0 stokes.P
    @copy thermal.Told thermal.T

    t, it      = 0.0, 0
    interval   = 1.0
    local Vx, Vy, Vz
    if do_vtk
        Vx = @zeros(ni...)
        Vy = @zeros(ni...)
        Vz = @zeros(ni...)
    end

    grid2particle!(pT, xvi, thermal.T, particles)
    dt₀         = similar(stokes.P)
    pT0.data    .= pT.data;

    ## Do a incompressible stokes solve, to get a better initial guess for the compressible stokes
    ## solver. And after that apply the extension/compression BCs. This is done to avoid the
    ## incompressible stokes solver to blow up.
    # println("Starting incompressible stokes solve")
    solve!(
        stokes,
        pt_stokes,
        di,
        flow_bcs,
        ρg,
        phase_ratios,
        rheology_incomp,
        args,
        dt,
        igg;
        kwargs = (;
            iterMax          = 150e3,#250e3,
            free_surface     = false,
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
    #     args,
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
        BC_velo!(@velocity(stokes)..., εbg, xvi,lx,lz, phases_dev)
        flow_bcs = VelocityBoundaryConditions(;
            free_slip    = (left = true , right = true , top = true , bot = true , front = true , back = true ),
            no_slip      = (left = false, right = false, top = false, bot = false, front = false, back = false),
        )
        flow_bcs!(stokes, flow_bcs) # apply boundary conditions
        update_halo!(@velocity(stokes)...) # update halo cells
    end
    println("Starting main loop")

    while it < 1500 #nt

        args = (; ϕ=ϕ_m, T=thermal.Tc, P=stokes.P, dt=dt, ΔTc=thermal.ΔTc, perturbation_C = perturbation_C)
        ## Stokes solver -----------------------------------
        solve!(
            stokes,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            phase_ratios,
            rheology,
            args,
            dt,
            igg;
            kwargs = (;
                iterMax          = 150e3,#250e3,
                free_surface     = false,
                nout             = 2e3,#5e3,
                viscosity_cutoff = cutoff_visc,
            )
        )

        dt = compute_dt(stokes, di, dt_diff, igg) #/ 9.81

        tensor_invariant!(stokes.ε)
        tensor_invariant!(stokes.ε_pl)

        ## Save the checkpoint file before a possible thermal solver blow up
        checkpointing_jld2(joinpath(checkpoint, "thermal"), stokes, thermal, t, dt, igg)

        # ------------------------------
        compute_shear_heating!(
            thermal,
            stokes,
            phase_ratios,
            rheology, # needs to be a tuple
            dt,
        )
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
                iterMax = 150e3,
                nout    = 1e3,
                verbose = true,
            )
        )


        subgrid_characteristic_time!(
            subgrid_arrays, particles, dt₀, phase_ratios, rheology, thermal, stokes, xci, di
        )
        centroid2particle!(subgrid_arrays.dt₀, xci, dt₀, particles)
        subgrid_diffusion!(
            pT, thermal.T, thermal.ΔT, subgrid_arrays, particles, xvi,  di, dt
        )
        compute_melt_fraction!(
            ϕ_m, phase_ratios, rheology, (T=thermal.Tc, P=stokes.P)
        )

        # Advection --------------------
        # advect particles in space
        # advection!(particles, RungeKutta2(), @velocity(stokes), (grid_vx, grid_vy), dt)
        # advection_LinP!(particles, RungeKutta2(), @velocity(stokes), grid_vxi, dt)
        advection_MQS!(particles, RungeKutta2(), @velocity(stokes), grid_vxi, dt)

        # update halos
        update_cell_halo!(particles.coords..., particle_args...);
        update_cell_halo!(particles.index)

        # advect particles in memory
        move_particles!(particles, xvi, particle_args)

        # check if we need to inject particles
        inject_particles_phase!(particles, pPhases, (pT, ), (thermal.T,), xvi)

        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
        vertex2center!(thermal.ΔTc, thermal.ΔT)

        if igg.me == 0
            @show it += 1
            t        += dt
        end

        #MPI gathering
        phase_center = [argmax(p) for p in Array(phase_ratios.center)]
        #centers
        @views P_nohalo     .= Array(stokes.P[2:end-1, 2:end-1, 2:end-1]) # Copy data to CPU removing the halo
        @views τII_nohalo   .= Array(stokes.τ.II[2:end-1, 2:end-1, 2:end-1]) # Copy data to CPU removing the halo
        @views η_nohalo     .= Array(stokes.viscosity.η[2:end-1, 2:end-1, 2:end-1]) # Copy data to CPU removing the halo
        @views η_vep_nohalo .= Array(stokes.viscosity.η_vep[2:end-1, 2:end-1, 2:end-1])       # Copy data to CPU removing the halo
        @views εII_nohalo   .= Array(stokes.ε.II[2:end-1, 2:end-1, 2:end-1]) # Copy data to CPU removing the halo
        @views εII_pl_nohalo .= Array(stokes.ε_pl.II[2:end-1, 2:end-1, 2:end-1]) # Copy data to CPU removing the halo
        @views EII_pl_nohalo.= Array(stokes.EII_pl[2:end-1, 2:end-1, 2:end-1]) # Copy data to CPU removing the halo
        @views ϕ_nohalo     .= Array(ϕ[2:end-1, 2:end-1, 2:end-1]) # Copy data to CPU removing the halo
        @views ρg_nohalo    .= Array(ρg[end][2:end-1, 2:end-1, 2:end-1]) # Copy data to CPU removing the halo
        @views phases_c_nohalo   .= Array(phase_center[2:end-1, 2:end-1, 2:end-1])
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
        if do_vtk
            velocity2center!(Vx, Vy, Vz, @velocity(stokes)...)
            @views Vxv_nohalo   .= Array(Vx[2:end-1, 2:end-1, 2:end-1]) # Copy data to CPU removing the halo
            @views Vyv_nohalo   .= Array(Vy[2:end-1, 2:end-1, 2:end-1]) # Copy data to CPU removing the halo
            @views Vzv_nohalo   .= Array(Vz[2:end-1, 2:end-1, 2:end-1]) # Copy data to CPU removing the halo
            gather!(Vxv_nohalo, Vxv_v)
            gather!(Vyv_nohalo, Vyv_v)
            gather!(Vzv_nohalo, Vzv_v)
        end
        @views T_nohalo     .= Array(thermal.Tc[2:end-1, 2:end-1, 2:end-1]) # Copy data to CPU removing the halo
        gather!(T_nohalo, T_v)

        ## Plotting -------------------------------------------------------
        if it == 1 || rem(it, 1) == 0
            if igg.me == 0 && it == 1
                metadata(pwd(), checkpoint, basename(@__FILE__), "TanzaniaModelSetup.jl", "TanzaniaRheology.jl")
            end
            checkpointing_jld2(checkpoint, stokes, thermal, t, dt, igg)
            ## Somehow fails to open with load("particles.jld2")
            mktempdir() do tmpdir
                # Save the checkpoint file in the temporary directory
                tmpfname = joinpath(tmpdir, basename(joinpath(checkpoint, "particles.jld2")))
                jldsave(
                    tmpfname;
                    particles = JustPIC._3D.Array(particles),
                    Phases = JustPIC._3D.Array(pPhases),
                    phase_ratios = JustPIC._3D.Array(phase_ratios),
                    time=t,
                    timestep=dt,
                )
                # Move the checkpoint file from the temporary directory to the destination directory
                mv(tmpfname, joinpath(checkpoint, "particles.jld2"); force=true)
            end

            t_yrs = dimensionalize(t, yr, CharDim)
            t_Kyrs = t_yrs / 1e3
            t_Myrs = t_Kyrs / 1e3

            xci_dim = ntuple(Val(3)) do i
                ustrip.(dimensionalize(xci_v[i], km, CharDim))
            end


            if igg.me == 0 && do_vtk
                data_c = (;
                    T = ustrip.(dimensionalize(T_v, C, CharDim)),
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
                    Vz = ustrip.(dimensionalize(Array(Vzv_v), cm / yr, CharDim)),
                    phases = phases_c_v
                )
                velocity = (
                    ustrip.(dimensionalize(Array(Vxv_v), cm / yr, CharDim)),
                    ustrip.(dimensionalize(Array(Vyv_v), cm / yr, CharDim)),
                    ustrip.(dimensionalize(Array(Vzv_v), cm / yr, CharDim)),
                )
                save_vtk(
                    joinpath(vtk_dir, "vtk_" * lpad("$(it)_$(igg.me)", 6, "0")),
                    xci_dim,
                    data_c,
                    velocity,
                    t=(round.(ustrip.(t_Kyrs); digits=3))
                )
            end
        end
    end
end

do_vtk = true
ar = 1 # aspect ratio
n = 128
nx = n * ar
ny = n * ar
nz = n
igg = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, nz; init_MPI=true)...)
else
    igg
end

# GLOBAL Physical domain -----------------------------------------------------
topo_tanzania, Topo_cartesian, Lx, Ly = Tanzania_Topo(nx,ny,nz)
sticky_air = 5.0

x_global = range(Topo_cartesian.x.val[1, 1, 1], Topo_cartesian.x.val[end, 1, 1], nx_g())
y_global = range(Topo_cartesian.y.val[1, 1, 1], Topo_cartesian.y.val[1, end, 1], ny_g())
z_global = range(-40, sticky_air, nz_g())
origin = (x_global[1], y_global[1], z_global[1])
li = (abs(last(x_global)-first(x_global)),
      abs(last(y_global)-first(y_global)),
      abs(last(z_global)-first(z_global)))
ni           = nx, ny, nz           # number of cells
di           = @. li / (nx_g(), ny_g(), nz_g())           # grid steps
grid_global  = Geometry(ni, li; origin = origin)

li_GMG, origin_GMG, phases_GMG, T_GMG = Tanzania_setup3D(grid_global.xvi,igg,nx+1,ny+1,nz+1;)

DisplacementFormulation = false             #specify if you want to use the displacement formulation

shear = true                                #specify if you want to use pure shear boundary conditions
εbg_dim   = 5e-13 / s * shear                                 #specify the background strain rate

figname = "$(today())_Tanzania_$(nx_g())x$(ny_g())x$(nz_g())"

Tanzania_3D(li_GMG, origin_GMG, phases_GMG, T_GMG, εbg_dim, igg; x_global, y_global, z_global, figname=figname, nx=nx, ny=ny, nz=nz, do_vtk=do_vtk)
