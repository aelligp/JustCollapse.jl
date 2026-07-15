const isCUDA = false
# const isCUDA = true

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
using GeoParams, CairoMakie, CellArrays, Statistics, Dates, JLD2, Printf

# Load file with all the rheology configurations
include("SillModelSetup_MPI.jl")
include("SillRheology.jl")


## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------

import ParallelStencil.INDICES
const idx_j = INDICES[2]
macro all_j(A)
    esc(:($A[$idx_j]))
end

# Initial pressure profile - not accurate
@parallel function init_P!(P, ρg, z)
    @all(P) = abs(@all(ρg) * @all_j(z)) * <(@all_j(z), 0.0)
    return nothing
end

function compute_Re!(Re_c, Vx, Vy, ρ, sill_size, η)
    # Characteristic velocity magnitude at center
    # Characteristic length (use mean grid spacing)
    L = mean(sill_size)
    # Reynolds number: Re = ρ * V * L / η
    V = @. sqrt(Vx^2 + Vy^2)
    @. Re_c = (ρ * V * L) / η
    return nothing
end

function compute_Ra!(Ra, ΔT, ρ, α, g, L, κ, η)
    # Rayleigh number: Ra = (ρ * α * g * ΔT * L^3) / (η * κ)
    # where:           Ra = (kg/m3 * 1/K * m/s2 * K * m^3) / (Pa s * m2/s)
    @. Ra = (ρ * α * g * ΔT * L^3) / (η * κ)
    return nothing
end

function d18O_anomaly!(
    d18O, z, phase_ratios;
    crust_gradient::Bool = true,
    crust_min::Float64 = -5.0,
    crust_max::Float64 = 3.0,
    crust_const::Float64 = 0.0,
    sill_value::Float64 = 5.5,
    crust_phase::Int = 1,
    sill_phase::Int = 2
)
    ni = size(phase_ratios.center)

    @parallel_indices (i, j) function _d18O_anomaly!(d18O, z, center_ratio)
        crust_ratio = @index center_ratio[crust_phase, i, j]
        sill_ratio = @index center_ratio[sill_phase, i, j]

        if sill_ratio > 0.5
            d18O[i, j] = sill_value
        elseif crust_ratio > 0.5
            if crust_gradient
                # Linear gradient from crust_min at shallowest to crust_max at deepest
                zmin = z[1]
                zmax = z[end]
                d18O[i, j] = crust_min + (crust_max - crust_min) * (z[j] - zmin) / (zmax - zmin)
            else
                d18O[i, j] = crust_const
            end
        end
        return nothing
    end

    @parallel (@idx ni) _d18O_anomaly!(d18O, z, phase_ratios.center)

    return nothing
end


function init_sill!(
    phases,
    dimensions::NTuple{2, Float64},
    sill_size,
    grid;
    perturbation_amplitude::Float64 = 0.0,
    wavelength::Float64 = 100.0,
    bottom_pertubation = false
    )

    @parallel_indices (i, j) function _init_sill!(
        phases, dimensions, sill_size, x, z, perturbation_amplitude, wavelength, bottom_pertubation
        )
        x_coord = x[i]
        depth = -z[j]

        # Add sinusoidal perturbation to sill top and bottom
        perturbation = perturbation_amplitude * sin.(2π * x_coord / wavelength)
        perturbation_bot = perturbation * bottom_pertubation

        sill_bottom = (dimensions[2] - (dimensions[2] - sill_size) / 2) + perturbation_bot
        sill_top = (dimensions[2] - sill_size) / 2 + perturbation
        if depth <= sill_bottom && depth >= sill_top
            phases[i, j] = 2
        else
            phases[i, j] = 1
        end
        return nothing
    end

    @parallel (@idx size(phases)) _init_sill!(
        phases, dimensions, sill_size, grid..., perturbation_amplitude, wavelength, bottom_pertubation
    )
    return nothing
end

function init_T!(
    T,
    host_rock_temp::Float64,
    sill_temp::Float64,
    dimensions::NTuple{2, Float64},
    sill_size::Float64,
    grid;
    perturbation_amplitude::Float64 = 0.0,
    wavelength::Float64 = 100.0,
    bottom_pertubation = false
    )

    @parallel_indices (i, j) function _init_T!(T, host_rock_temp, sill_temp,  dimensions, sill_size, x, z, perturbation_amplitude, wavelength, bottom_pertubation)

        x_coord = x[i]
        depth = -z[j]

        # Add sinusoidal perturbation to sill top and bottom
        perturbation = perturbation_amplitude * sin.(2π * x_coord / wavelength)
        perturbation_bot = perturbation * bottom_pertubation

        sill_bottom = (dimensions[2] - (dimensions[2] - sill_size) / 2) + perturbation_bot
        sill_top = (dimensions[2] - sill_size) / 2 + perturbation
        if depth ≤ sill_bottom && depth ≥ sill_top
            T[i + 1, j + 1] = sill_temp
        else
            T[i + 1, j + 1] = host_rock_temp
        end
        return nothing
    end

    ni = size(T) .- 2
    @parallel (@idx ni) _init_T!(T, host_rock_temp, sill_temp, dimensions, sill_size, grid..., perturbation_amplitude, wavelength, bottom_pertubation)

end

## END OF HELPER FUNCTION ------------------------------------------------------------

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main(li, origin, phases_GMG, T_GMG, igg, x_global, z_global; nx = 64, ny =64, figdir="SillConvection2D", do_vtk = false, cutoff_visc = (-Inf, Inf), plotting = true, sill_temp = 1000, host_rock_temp = 500, sill_size = 0.1, depth = 5e3)

    # -----------------------------------------------------
    # Set up the JustRelax model
    # -----------------------------------------------------
    ni = nx, ny           # number of cells
    di = @. li / ni       # grid steps
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid             # nodes at the center and vertices of the cells

    # ---------------------------------------------------

    # Physical properties using GeoParams ----------------
                     # (SiO2   TiO2  Al2O3  FeO   MgO   CaO   Na2O  K2O   H2O)
    oxd_wt_sill      = (70.78, 0.55, 15.86, 3.93, 1.11, 1.20, 2.54, 3.84, 2.0)
    oxd_wt_host_rock = (75.75, 0.28, 12.48, 2.14, 0.09, 0.48, 3.53, 5.19, 2.0)
    rheology = init_rheologies(oxd_wt_sill, oxd_wt_host_rock; scaling = 1e5, magma = true)
    dt_time = 1.0 * 3600 * 24 * 365
    κ            = (4 / (1050 * rheology[1].Density[1].ρ))
    dt_diff = 0.5 * min(di...)^2 / κ / 2.01
    dt = min(dt_time, dt_diff)
    # ----------------------------------------------------
    # Weno model -----------------------------------------
    weno = WENO5(backend, Val(2), ni) # T and d18O live at cell centers

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 100, 150, 75
    particles = init_particles(
        backend_JP, nxcell, max_xcell, min_xcell, grid.xi_vel...
    )

    # temperature
    pT, pPhases      = init_cell_arrays(particles, Val(2));
    # particle fields for the stress rotation
    pτ = StressParticles(particles)
    particle_args = (pT, pPhases, unwrap(pτ)...)
    particle_args_reduced = (pT, unwrap(pτ)...)

    # Assign material phases --------------------------
    phases_dev   = PTArray(backend)(phases_GMG)
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni);
    init_phases!(pPhases, phases_dev, particles, xvi)
    update_phase_ratios!(phase_ratios, particles, pPhases)

    update_halo!(particles.coords...);
    update_halo!(particle_args...);
    update_halo!(particles.index)
    update_halo!(phase_ratios.center)
    update_halo!(phase_ratios.vertex)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes          = StokesArrays(backend, ni)
    pt_stokes       = PTStokesCoeffs(li, di; Re = 14.9, ϵ_rel=1e-5, ϵ_abs=1e-7, CFL=0.9 / √2.1) #ϵ=1e-4,  CFL=1 / √2.1 CFL=0.27 / √2.1
    # ----------------------------------------------------

    thermal         = ThermalArrays(backend, ni) # T lives at cell centers with one ghost node on every boundary
    vertex2center!(thermal.T, PTArray(backend)(T_GMG); ghost_x = true, ghost_y = true)
    thermal_bc      = TemperatureBoundaryConditions(;
        no_flux     = (left = true, right = true, top = false, bot = false),
    )
    thermal_bcs!(thermal, thermal_bc)

    args = (; T=thermal.T, P=stokes.P, dt=dt)

    pt_thermal = PTThermalCoeffs(
        backend, rheology, phase_ratios, args, dt, ni, di, li; ϵ=1e-5, CFL=0.98 / √2.1
    )

    # Melt Fraction
    ϕ = @zeros(ni...)

    # Buoyancy force
    ρg = @zeros(ni...), @zeros(ni...)                      # ρg[1] is the buoyancy force in the x direction, ρg[2] is the buoyancy force in the y direction

    for _ in 1:5
        compute_ρg!(ρg[end], phase_ratios, rheology, (T=thermal.T, P=stokes.P))
        @parallel init_P!(stokes.P, ρg[2], xci[2])
        pressure_offset = ρg[end] .* depth  # in Pascals
        # Add pressure offset to simulate pressure at reference depth
        stokes.P .+= pressure_offset
    end
    # Rheology
    compute_melt_fraction!(
        ϕ, phase_ratios, rheology, (T=thermal.T, P=stokes.P)
    )
    compute_viscosity!(stokes, phase_ratios, args, rheology, cutoff_visc)

    # Track isotope ratios
    d18O = @zeros(ni...)
    d18O_anomaly!(d18O, xci[2], phase_ratios; crust_gradient = false, crust_const = -3.0)

    @copy stokes.P0 stokes.P
    @copy thermal.Told thermal.T

    # Boundary conditions
    flow_bcs         = VelocityBoundaryConditions(;
        free_slip    = (left = true, right=true, top=true, bot=true),
    )
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    # IO -------------------------------------------------
    # if it does not exist, make folder where figures are stored
    if plotting
        take(figdir)
        if do_vtk
            vtk_dir = joinpath(figdir, "vtk")
            take(vtk_dir)
        end
        checkpoint = joinpath(figdir, "checkpoint")
    end
    # ----------------------------------------------------
    # Plot initial T and η profiles
    let
        Y   = [y for x in xci[1], y in xci[2]][:]
        fig = Figure(size = (1200, 900))
        ax1 = Axis(fig[1,1], aspect = 2/3, title = "T")
        ax2 = Axis(fig[1,2], aspect = 2/3, title = "log10(η)")
        scatter!(
            ax1,
            Array(thermal.T[2:(end - 1), 2:(end - 1)][:].-273),
            Y,)
        scatter!(
            ax2,
            log10.(Array(stokes.viscosity.η[:])),
            Y,)
        hideydecorations!(ax2)
        # save(joinpath(figdir, "initial_profile.png"), fig)
        fig
    end
    let
        compo = [oxd_wt_sill[1] (oxd_wt_sill[7]+oxd_wt_sill[8]);
                oxd_wt_host_rock[1] (oxd_wt_host_rock[7]+oxd_wt_host_rock[8])]
        fig=Plot_TAS_diagram(compo; sz=(1000, 1000))
        save(joinpath(figdir, "TAS_diagram.png"), fig)
    end
    #MPI
    # global array
    nx_v = (nx - 2) * igg.dims[1]
    ny_v = (ny - 2) * igg.dims[2]
    # center
    P_v = zeros(nx_v, ny_v)
    τII_v = zeros(nx_v, ny_v)
    η_vep_v = zeros(nx_v, ny_v)
    η_v = zeros(nx_v, ny_v)
    εII_v = zeros(nx_v, ny_v)
    phases_c_v = zeros(nx_v, ny_v)
    d18O_c_v = zeros(nx_v, ny_v)    # d18O
    ϕ_v = zeros(nx_v, ny_v)         # Melt fraction
    ρg_v = zeros(nx_v, ny_v)     # Buoyancy force
    Re_v = zeros(nx_v, ny_v)         # Reynolds number
    Ra_v = zeros(nx_v, ny_v)         # Rayleigh number
    #center nohalo
    P_nohalo = zeros(nx - 2, ny - 2)
    τII_nohalo = zeros(nx - 2, ny - 2)
    η_vep_nohalo = zeros(nx - 2, ny - 2)
    η_nohalo = zeros(nx - 2, ny - 2)
    εII_nohalo = zeros(nx - 2, ny - 2)
    phases_c_nohalo = zeros(nx - 2, ny - 2)
    d18O_c_nohalo = zeros(nx - 2, ny - 2)
    ϕ_nohalo = zeros(nx - 2, ny - 2)
    ρg_nohalo = zeros(nx - 2, ny - 2)
    Re_nohalo = zeros(nx - 2, ny - 2)
    Ra_nohalo = zeros(nx - 2, ny - 2)
    #vertex
    Vxv_v = zeros(nx_v, ny_v)
    Vyv_v = zeros(nx_v, ny_v)
    T_v = zeros(nx_v, ny_v)
    #vertex nohalo
    Vxv_nohalo = zeros(nx - 2, ny - 2)
    Vyv_nohalo = zeros(nx - 2, ny - 2)
    T_nohalo = zeros(nx - 2, ny - 2)

    xci_v = LinRange(minimum(x_global) .* 1.0e3, maximum(x_global) .* 1.0e3, nx_v),
        LinRange(minimum(z_global) .* 1.0e3, maximum(z_global) .* 1.0e3, ny_v)


    # WENO arrays
    T_WENO  = @zeros(ni...)
    Vx_v = @zeros(ni.+1...)
    Vy_v = @zeros(ni.+1...)
    Vx_c = @zeros(ni...)
    Vy_c = @zeros(ni...)
    Vx = @zeros(ni...)
    Vy = @zeros(ni...)
    τxx_v = @zeros(ni .+ 1...)
    τyy_v = @zeros(ni .+ 1...)

    # Time loop
    t, it = 0.0, 0

    Re = @zeros(ni...)
    Ra = @zeros(ni...)

    while it < 3000
    # while it < 50

        args = (; ϕ= ϕ,T = thermal.T, P = stokes.P, dt = dt, ΔT = thermal.ΔT)
        compute_ρg!(ρg[end], phase_ratios, rheology, (T = thermal.T, P = stokes.P))
        compute_viscosity!(
            stokes, phase_ratios, args, rheology, cutoff_visc
        )

        stress2grid!(stokes, pτ, particles)
        # ------------------------------

        # Stokes solver ----------------
        solve!(
            stokes,
            pt_stokes,
            grid,
            flow_bcs,
            ρg,
            phase_ratios,
            rheology,
            args,
            Inf,
            igg;
            kwargs = (;
                iterMax = 150.0e3,
                nout = 1.0e3,
                viscosity_cutoff = cutoff_visc,
            )
        )
        # rotate stresses
        rotate_stress!(pτ, stokes, particles, dt)

        tensor_invariant!(stokes.ε)
        tensor_invariant!(stokes.ε_pl)
        ## Save the checkpoint file before a possible thermal solver blow up
        checkpointing_jld2(joinpath(checkpoint, "thermal"), stokes, thermal, t, dt, igg)

        dt   = compute_dt(stokes, di, dt_diff)
        println("dt = $(dt/(3600*24))")
        # Thermal solver ---------------
        heatdiffusion_PT!(
            thermal,
            pt_thermal,
            thermal_bc,
            rheology,
            args,
            dt,
            grid;
            kwargs =(;
                igg     = igg,
                phase   = phase_ratios,
                iterMax = 150e3,
                nout    = 1e3,
                verbose = true,
            )
        )
        # ------------------------------

        T_WENO .= @views thermal.T[2:end-1, 2:end-1]
        velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
        ## for Re and Ra number:
        velocity2center!(Vx_c, Vy_c, @velocity(stokes)...)

        WENO_advection!(T_WENO, (Vx_c, Vy_c), weno, di, dt)
        @views thermal.T[2:(end - 1), 2:(end - 1)] .= T_WENO

        # Isotope advection
        WENO_advection!(d18O, (Vx_c, Vy_c), weno, di, dt)

        thermal.ΔT .= thermal.T .- thermal.Told

        compute_Re!(Re, Vx_c, Vy_c, ρg[end] ./9.81,  sill_size *1e3, stokes.viscosity.η);
        compute_Ra!(Ra, sill_temp - host_rock_temp, ρg[end] ./ 9.81, 1050, rheology[1].Gravity[1].g.val, li, κ, stokes.viscosity.η);

        @show extrema(thermal.T)
        any(isnan.(thermal.T)) && break

        # Advection --------------------
        # advect particles in space
        advection_MQS!(particles, RungeKutta4(), @velocity(stokes), dt)
        update_halo!(particles.coords...);
        update_halo!(particle_args...);
        update_halo!(particles.index)
        update_halo!(phase_ratios.center)
        update_halo!(phase_ratios.vertex)
        # advect particles in memory
        move_particles!(particles, particle_args)
        center2vertex!(τxx_v, stokes.τ.xx)
        center2vertex!(τyy_v, stokes.τ.yy)

        # check if we need to inject particles
        inject_particles_phase!(
            particles,
            pPhases,
            particle_args_reduced,
            (T_WENO, τxx_v, τyy_v, stokes.τ.xy, stokes.ω.xy)
        )
        compute_melt_fraction!(ϕ, phase_ratios, rheology, (T=thermal.T, P=stokes.P))
        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, pPhases)
        tensor_invariant!(stokes.τ)

        if igg.me == 0
            @show it += 1
            t += dt
        end

        #MPI gathering
        phase_center = [argmax(p) for p in Array(phase_ratios.center)]
        #centers
        @views P_nohalo .= Array(stokes.P[2:(end - 1), 2:(end - 1)]) # Copy data to CPU removing the halo
        @views τII_nohalo .= Array(stokes.τ.II[2:(end - 1), 2:(end - 1)]) # Copy data to CPU removing the halo
        @views η_vep_nohalo .= Array(stokes.viscosity.η_vep[2:(end - 1), 2:(end - 1)])       # Copy data to CPU removing the halo
        @views η_nohalo .= Array(stokes.viscosity.η[2:(end - 1), 2:(end - 1)]) # Copy data to CPU removing the halo
        @views εII_nohalo .= Array(stokes.ε.II[2:(end - 1), 2:(end - 1)]) # Copy data to CPU removing the halo
        @views phases_c_nohalo .= Array(phase_center[2:(end - 1), 2:(end - 1)])
        @views d18O_c_nohalo .= Array(d18O[2:(end - 1), 2:(end - 1)]) # Copy data to CPU removing the halo
        @views ϕ_nohalo .= Array(ϕ[2:(end - 1), 2:(end - 1)]) # Copy data to CPU removing the halo
        @views ρg_nohalo .= Array(ρg[end][2:(end - 1), 2:(end - 1)]) # Copy data to CPU removing the halo
        @views Re_nohalo .= Array(Re[2:(end - 1), 2:(end - 1)]) # Copy data to CPU removing the halo
        @views Ra_nohalo .= Array(Ra[2:(end - 1), 2:(end - 1)]) # Copy data to CPU removing the halo

        gather!(P_nohalo, P_v)
        gather!(τII_nohalo, τII_v)
        gather!(η_vep_nohalo, η_vep_v)
        gather!(η_nohalo, η_v)
        gather!(εII_nohalo, εII_v)
        gather!(phases_c_nohalo, phases_c_v)
        gather!(d18O_c_nohalo, d18O_c_v)
        gather!(ϕ_nohalo, ϕ_v)
        gather!(ρg_nohalo, ρg_v)
        gather!(Re_nohalo, Re_v)
        gather!(Ra_nohalo, Ra_v)
        #vertices
        if do_vtk
            velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
            vertex2center!(Vx, Vx_v)
            vertex2center!(Vy, Vy_v)
            @views Vxv_nohalo .= Array(Vx[2:(end - 1), 2:(end - 1)]) # Copy data to CPU removing the halo
            @views Vyv_nohalo .= Array(Vy[2:(end - 1), 2:(end - 1)]) # Copy data to CPU removing the halo
            gather!(Vxv_nohalo, Vxv_v)
            gather!(Vyv_nohalo, Vyv_v)
        end
        @views T_nohalo .= Array(T_WENO[2:(end - 1), 2:(end - 1)]) # Copy data to CPU removing the halo
        gather!(T_nohalo, T_v)

        # Data I/O and plotting ---------------------
        if it == 1 || rem(it, 25) == 0
            if igg.me == 0 && it == 1
                metadata(pwd(), checkpoint, joinpath(@__DIR__, "SillConvection_MPI.jl"), joinpath(@__DIR__, "SillModelSetup_MPI.jl"), joinpath(@__DIR__, "SillRheology.jl"))
            end
            checkpointing_jld2(checkpoint, stokes, thermal, t, dt, igg)
            checkpointing_particles(checkpoint, particles; phases = pPhases, phase_ratios = phase_ratios, particle_args = particle_args, t = t, dt = dt)

            if do_vtk

                data_c = (;
                    T = Array(T_v),
                    P = Array(P_v),
                    viscosity_vep = Array(η_vep_v),
                    viscosity = Array(η_v),
                    phases = phases_c_v,
                    d18O = Array(d18O_c_v),
                    Melt_fraction = Array(ϕ_v),
                    stress_II = Array(τII_v),
                    strain_rate_II = Array(εII_v,),
                    density = Array(ρg_v ./ 9.81),
                )
                velocity_v = (
                    Array(Vxv_v),
                    Array(Vyv_v),
                )
                save_vtk(
                    joinpath(vtk_dir, "vtk_" * lpad("$it", 6, "0")),
                    xci_v ./ 1.0e3,
                    data_c,
                    velocity_v;
                    t = round(t / (1.0e3 * 3600 * 24 * 365.25); digits = 3)
                )
                save_particles(particles, pPhases; conversion = 1.0e3, fname = joinpath(vtk_dir, "particles_" * lpad("$it", 6, "0")))
            end

            # Make Makie figure
            fig = Figure(size = (2000, 1000), title = "t = $t", )
            ar = DataAspect()
            if t < 1e3
                TimeScale = 1
                TimeUnits = "s"
            elseif t >= 1e3 && t < 24*3600
                TimeScale = 3600
                TimeUnits = "hr"
            elseif t >= 24*3600 && t < 365*3600*24
                TimeScale = 3600*24
                TimeUnits = "days"
            elseif t >= 365*3600*24 && t < 1e3*3600*24*365
                TimeScale = 3600*24*365
                TimeUnits = "yr"
            else
                TimeScale = 1e3*3600*24*365
                TimeUnits = "kyr"
            end
            ax0 = Axis(
                fig[1, 1:2];
                aspect=ar,
                title = "T [C]  (time = $(round(t/TimeScale, digits=2)) $TimeUnits)",
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

            ax1 = Axis( fig[2, 1][1, 1], aspect = DataAspect(), title = L"T \;[\mathrm{C}]",  titlesize=40,
            yticklabelsize=25,
            xticklabelsize=25,
            xlabelsize=25,)
            ax2 = Axis(fig[2, 2][1, 1], aspect = DataAspect(), title = L"Density \;[\mathrm{kg/m}^{3}]", titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,)

            ax3 = Axis(fig[3, 1][1, 1], aspect = DataAspect(), title = L"Vy \;[\mathrm{m/s}]", titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,)

            ax4 = Axis(fig[3, 2][1, 1], aspect = DataAspect(), title = L"\phi", titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,)

            # Plot temperature
            h1  = heatmap!(
                ax1,
                xci_v...,
                Array(T_v.-273); colormap=:lipari, colorrange=(host_rock_temp, sill_temp))

            h2  = heatmap!(
                ax2,
                xci_v...,
                Array(ρg_v./9.81);
                colormap=:batlowW)

            # Plot vy velocity
            h3  = heatmap!(
                ax3,
                xci_v...,
                Array(Vyv_v); colormap=:batlow)

            # Plot melt fraction
            h4  = heatmap!(ax4,
                xci_v...,
                Array(ϕ_v);
                colormap=:lipari)

            hidexdecorations!(ax1)
            hidexdecorations!(ax2)
            hideydecorations!(ax2)
            hideydecorations!(ax4)
            Colorbar(fig[2, 1][1, 2], h1, height = Relative(4/4), ticklabelsize=25, ticksize=15)
            Colorbar(fig[2, 2][1, 2], h2, height = Relative(4/4), ticklabelsize=25, ticksize=15)
            Colorbar(fig[3, 1][1, 2], h3, height = Relative(4/4), ticklabelsize=25, ticksize=15)
            Colorbar(fig[3, 2][1, 2], h4, height = Relative(4/4), ticklabelsize=25, ticksize=15)
            linkaxes!(ax1, ax2, ax3, ax4)
            fig
            figsave = joinpath(figdir, @sprintf("%06d.png", it))
            save(figsave, fig)

            let
                fig = Figure(size = (2000, 1000), title = "t = $t")
                # Determine time scale and units for plotting
                if t < 1e3
                    TimeScale = 1
                    TimeUnits = "s"
                elseif t >= 1e3 && t < 24*3600
                    TimeScale = 3600
                    TimeUnits = "hr"
                elseif t >= 24*3600 && t < 365*3600*24
                    TimeScale = 3600*24
                    TimeUnits = "days"
                elseif t >= 365*3600*24 && t < 1e3*3600*24*365
                    TimeScale = 3600*24*365
                    TimeUnits = "yr"
                else
                    TimeScale = 1e3*3600*24*365
                    TimeUnits = "kyr"
                end
                ax1 = Axis(
                    fig[1,1],
                    aspect = DataAspect(),
                    title = "T [C]  (time = $(round(t/TimeScale, digits=2)) $TimeUnits)",
                    titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,
                )
                 # Plot temperature
                h1  = heatmap!(
                    ax1,
                    xci_v...,
                    (Array(T_v.-273));
                    colormap=:lipari, colorrange=(host_rock_temp, sill_temp))
                Colorbar(fig[1,2], h1, height = Relative(4/4), ticklabelsize=25, ticksize=15)
                save(joinpath(figdir, "Temperature_$(it).png"), fig)
                fig

                fig1 = Figure(size = (1600, 1000), title = "t = $t")

                ax1 = Axis(
                    fig1[1,1],
                    aspect = DataAspect(),
                    title = "Re (log10)  (time = $(round(t/TimeScale, digits=2)) $TimeUnits)",
                    titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,
                )
                # Plot Reynolds number
                h1  = heatmap!(
                    ax1,
                    xci_v...,
                    Array(log10.(Re_v));
                    colormap=:lapaz)

                Colorbar(fig1[1,2], h1, height = Relative(4/4), ticklabelsize=25, ticksize=15)
                ax2 = Axis(
                    fig1[2,1],
                    aspect = DataAspect(),
                    title = "δ18O",
                    titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,
                )
                # # Plot isotopes
                h2  = heatmap!(
                    ax2,
                    xci_v...,
                    Array(d18O_c_v);
                    colormap=:lapaz)
                Colorbar(fig1[2,2], h2, height = Relative(4/4), ticklabelsize=25, ticksize=15)
                linkaxes!(ax1, ax2)
                ax3 = Axis(
                    fig1[1,3],
                    aspect = DataAspect(),
                    title = "Ra(log10)",
                    titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,
                )
                # Plot Rayleigh number
                h3 = heatmap!(
                    ax3,
                    xci_v...,
                    Array(log10.(Ra_v));
                    colormap=:lapaz)
                Colorbar(fig1[1,4], h3, height = Relative(4/4), ticklabelsize=25, ticksize=15)
                save(joinpath(figdir, "Re_$(it).png"), fig1)
                fig1

            end

        end
        # ------------------------------

    end

    return nothing
end
## END OF MAIN SCRIPT ----------------------------------------------------------------
const plotting = true
do_vtk = true

# (Path)/folder where output data and figures are stored
figdir   = "SillConvection2D_$(today())"
n = 64
nx, ny = n, n *2  #>> 1

igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end
##### USER INPUTS ##########################################################
dimensions = (0.3, 0.25)
sill_temp = 1000.0 # in C
host_rock_temp = 500.0 # in C
sill_size = 0.1 # in km
depth = 5e3 # in m
############################################################################

x_global = range(0, dimensions[1], nx_g())
z_global = range(-dimensions[2], 0.0, ny_g())
origin   = (x_global[1], z_global[1]) # origin of the grid
li = (abs(last(x_global) - first(x_global)), abs(last(z_global) - first(z_global)))

ni = nx, ny           # number of cells
di = @. li / (nx_g(), ny_g())           # grid steps
grid_global = Geometry(ni, li; origin = origin)


li, origin, phases_GMG, T_GMG, _ = SillSetup_MPI(
    nx + 1,
    ny + 1,
    grid_global.xvi;
    dimensions = dimensions,
    sill_temp = sill_temp,
    host_rock_temp = host_rock_temp,
    sill_size = sill_size
)

# run main script
main(li, origin, phases_GMG, T_GMG, igg, x_global, z_global; nx = nx, ny = ny, figdir = figdir, do_vtk = do_vtk, cutoff_visc = (1e1, 1.0e18), plotting = plotting, sill_temp = sill_temp, host_rock_temp = host_rock_temp, sill_size = sill_size, depth = depth);
