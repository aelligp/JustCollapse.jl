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
# include("SillModelSetup.jl")
include("SillRheology.jl")


## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------

function copyinn_x!(A, B)

    @parallel function f_x(A, B)
        @all(A) = @inn_x(B)
        return nothing
    end

    @parallel f_x(A, B)
end

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
    @. Ra = (ρ * α * g * ΔT * L^3) / (η * κ)  # Ensure all arrays have the same shape
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
    ni = size(phase_ratios.vertex)

    @parallel_indices (i, j) function _d18O_anomaly!(d18O, z, vertex_ratio)
        crust_ratio = @index vertex_ratio[crust_phase, i, j]
        sill_ratio = @index vertex_ratio[sill_phase, i, j]

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

    @parallel (@idx ni) _d18O_anomaly!(d18O, z, phase_ratios.vertex)

    return nothing
end


function init_sill!(
    phases,
    dimensions::NTuple{2, Float64},
    sill_size,
    z
    )

    @parallel_indices (i, j) function _init_sill!(
        phases, dimensions, sill_size, z
        )

        depth = -z[j]
        sill_top = (dimensions[2] - (dimensions[2] - sill_size) / 2)
        sill_bottom = (dimensions[2] - sill_size) / 2
        if depth <= sill_top && depth >= sill_bottom
            phases[i, j] = 2
        else
            phases[i, j] = 1
        end
        return nothing
    end

    @parallel (@idx size(phases)) _init_sill!(
        phases, dimensions, sill_size, z
    )
    return nothing
end

@parallel_indices (i, j) function init_T!(T, host_rock_temp, sill_temp,  dimensions, sill_size, z)
    depth = -z[j]
    sill_top = (dimensions[2] - (dimensions[2] - sill_size) / 2)
    sill_bottom = (dimensions[2] - sill_size) / 2
    if depth <= sill_top && depth >= sill_bottom
        T[i + 1, j] = sill_temp
    else
        T[i + 1, j] = host_rock_temp
    end
    return nothing
end

## END OF HELPER FUNCTION ------------------------------------------------------------

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main(li, origin, igg; nx = 64, ny =64, figdir="SillConvection2D", do_vtk = false, cutoff_visc = (-Inf, Inf), plotting = true, sill_temp = 1000, host_rock_temp = 500, sill_size = 0.1, depth = 5e3)

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
    rheology = init_rheologies(oxd_wt_sill, oxd_wt_host_rock; scaling = 1e3, magma = true)
    # rheology     = init_rheologies(;)
    dt_time = 1.0 * 3600 * 24 * 365
    κ            = (4 / (1050 * rheology[1].Density[1].ρ))
    # κ = (4 / (rheology[1].HeatCapacity[1].Cp.val * rheology[1].Density[1].ρ0.val)) # thermal diffusivity                                 # thermal diffusivity
    dt_diff = 0.5 * min(di...)^2 / κ / 2.01
    dt = min(dt_time, dt_diff)
    # ----------------------------------------------------
    # Weno model -----------------------------------------
    weno = WENO5(backend, Val(2), ni.+1) # ni.+1 for Temp
    weno_c = WENO5(backend, Val(2), ni) # ni.+1 for Temp

    # Initialize particles -------------------------------
    # nxcell, max_xcell, min_xcell = 100, 150, 75
    # particles = init_particles(
    #     backend_JP, nxcell, max_xcell, min_xcell, xvi...
    # )
    # subgrid_arrays = SubgridDiffusionCellArrays(particles)
    # velocity grids
    # grid_vxi = velocity_grids(xci, xvi, di)

    # temperature
    # pT, pPhases, pδ18O      = init_cell_arrays(particles, Val(3));
    # particle fields for the stress rotation
    # pτ = StressParticles(particles)
    # particle_args = (pT, pδ18O, pPhases, unwrap(pτ)...)
    # particle_args_reduced = (pT, unwrap(pτ)...)

    # Assign material phases --------------------------
    phases_dev   = @zeros(ni...)
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni);
    init_sill!(phases_dev, li, sill_size, xci[2])

    phase_sill = @zeros(ni...) # initialize phase_sill array
    phase_host = @zeros(ni...) # initialize phase_host array
    @views phase_sill[phases_dev .== 1.0] .= 0.0 # set the blob to 1.0 where phases == 2.0
    @views phase_sill[phases_dev .== 2.0] .= 1.0 # set the blob to 1.0 where phases == 2.0
    @views phase_host[phases_dev .== 1.0] .= 1.0 # set the blob to 1.0 where phases == 2.0
    @views phase_host[phases_dev .== 2.0] .= 0.0 # set the blob to 1.0 where phases == 2.0
    clamp!(phase_sill, 0.0, 1.0) # clamp
    clamp!(phase_host, 0.0, 1.0) # clamp phase_host to 0.0 and 1.0
    update_phase_ratios_2D!(phase_ratios, (phase_host, phase_sill), xci, xvi)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes          = StokesArrays(backend, ni)
    pt_stokes       = PTStokesCoeffs(li, di; Re = 14.9, ϵ_rel=1e-5, ϵ_abs=1e-5, CFL=0.9 / √2.1) #ϵ=1e-4,  CFL=1 / √2.1 CFL=0.27 / √2.1
    # ----------------------------------------------------

    thermal         = ThermalArrays(backend, ni)
    # @views thermal.T[2:end-1, :] .= PTArray(backend)(T_GMG)
    @parallel (@idx ni .+ 1) init_T!(thermal.T, host_rock_temp, sill_temp, li, sill_size, xvi[2])
    thermal_bc      = TemperatureBoundaryConditions(;
        no_flux     = (left = true, right = true, top = false, bot = false),
    )
    thermal_bcs!(thermal, thermal_bc)
    temperature2center!(thermal)

    args = (; T=thermal.Tc, P=stokes.P, dt=dt)

    pt_thermal = PTThermalCoeffs(
        backend, rheology, phase_ratios, args, dt, ni, di, li; ϵ=1e-5, CFL=0.98 / √2.1
    )

    # Melt Fraction
    ϕ = @zeros(ni...)

    # Buoyancy force
    ρg = @zeros(ni...), @zeros(ni...)                      # ρg[1] is the buoyancy force in the x direction, ρg[2] is the buoyancy force in the y direction

    for _ in 1:5
        compute_ρg!(ρg[end], phase_ratios, rheology, (T=thermal.Tc, P=stokes.P))
        @parallel init_P!(stokes.P, ρg[2], xci[2])
        pressure_offset = ρg[end] .* depth  # in Pascals
        # Add pressure offset to simulate pressure at reference depth
        stokes.P .+= pressure_offset
    end
    # Rheology
    compute_melt_fraction!(
        ϕ, phase_ratios, rheology, (T=thermal.Tc, P=stokes.P)
    )
    # Track isotope ratios
    d18O = @zeros(ni.+ 1...)
    d18O_anomaly!(d18O, xvi[2], phase_ratios; crust_gradient = false, crust_const = -3.0)

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
        Yv  = [y for x in xvi[1], y in xvi[2]][:]
        Y   = [y for x in xci[1], y in xci[2]][:]
        fig = Figure(size = (1200, 900))
        ax1 = Axis(fig[1,1], aspect = 2/3, title = "T")
        ax2 = Axis(fig[1,2], aspect = 2/3, title = "log10(η)")
        scatter!(
            ax1,
            Array(thermal.T[2:(end - 1), :][:].-273),
            Yv,)
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
    # WENO arrays
    T_WENO  = @zeros(ni.+1)
    Vx_v = @zeros(ni.+1...)
    Vy_v = @zeros(ni.+1...)
    Vx_c = @zeros(ni...)
    Vy_c = @zeros(ni...)
    τxx_v = @zeros(ni .+ 1...)
    τyy_v = @zeros(ni .+ 1...)

    local Vx_v, Vy_v, T_WENO
    # Time loop
    t, it = 0.0, 0

    Re = @zeros(ni...)
    Ra = @zeros(ni...)
    time_vec = Float64[0.0]
    d18O_evo = Float64[5.5]
    melt_fraction_evo = Float64[1.0]

    while it < 100e3 && round(maximum(ϕ), digits=2) > 0.3 && t < (60 * 3600 * 24 * 365)
    # while it < 50

        args = (; ϕ= ϕ,T = thermal.Tc, P = stokes.P, dt = dt, ΔTc = thermal.ΔTc)
        compute_ρg!(ρg[end], phase_ratios, rheology, (T = thermal.Tc, P = stokes.P))
        # ------------------------------

        # Stokes solver ----------------
        solve!(
            stokes,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            phase_ratios,
            rheology,
            args,
            Inf,
            igg;
            kwargs = (;
                iterMax = 100.0e3,
                nout = 2.0e3,
                viscosity_cutoff = cutoff_visc,
            )
        )
        # rotate stresses
        # rotate_stress!(pτ, stokes, particles, xci, xvi, dt)

        tensor_invariant!(stokes.ε)
        tensor_invariant!(stokes.ε_pl)
        tensor_invariant!(stokes.τ)
        ## Save the checkpoint file before a possible thermal solver blow up
        # checkpointing_jld2(joinpath(checkpoint, "thermal"), stokes, thermal, t, dt, igg)

        dt   = compute_dt(stokes, di, dt_diff)
        println("dt = $(dt/(3600*24)) days")
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
                iterMax = 100e3,
                nout    = 1e3,
                verbose = true,
            )
        )
        # ------------------------------

        T_WENO .= @views thermal.T[2:end-1, :]
        velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
        velocity2center!(Vx_c, Vy_c, @velocity(stokes)...)

        # Ensure Vx_v and Vy_v have the same shape as T_WENO for WENO_advection!
        WENO_advection!(T_WENO, (Vx_v, Vy_v), weno, di, dt)
        @views thermal.T[2:(end - 1), :] .= T_WENO

        # Advect phases
        WENO_advection!(phase_sill, (Vx_c, Vy_c), weno_c, di, dt)
        WENO_advection!(phase_host, (Vx_c, Vy_c), weno_c, di, dt)

        # Isotope advection
        WENO_advection!(d18O, (Vx_v, Vy_v), weno, di, dt)

        thermal.ΔT .= thermal.T .- thermal.Told
        vertex2center!(thermal.ΔTc, thermal.ΔT[2:(end - 1), :])

        compute_Re!(Re, Vx_c, Vy_c, ρg[end] ./9.81,  li[2], stokes.viscosity.η);
        compute_Ra!(Ra, sill_temp - host_rock_temp, ρg[end] ./ 9.81, 3e-5, rheology[1].Gravity[1].g.val, li[2], κ, stokes.viscosity.η);

        @show extrema(thermal.T)
        any(isnan.(thermal.T)) && break

        # # Advection --------------------
        # # advect particles in space
        # advection_MQS!(particles, RungeKutta4(), @velocity(stokes), grid_vxi, dt)
        # # advect particles in memory
        # move_particles!(particles, xvi, particle_args)
        # center2vertex!(τxx_v, stokes.τ.xx)
        # center2vertex!(τyy_v, stokes.τ.yy)

        # # check if we need to inject particles
        # inject_particles_phase!(
        #     particles,
        #     pPhases,
        #     particle_args_reduced,
        #     (T_WENO, τxx_v, τyy_v, stokes.τ.xy, stokes.ω.xy),
        #     xvi
        # )
        # compute_viscosity!(stokes, phase_ratios, args, rheology, cutoff_visc)
        # update phase ratios
        # update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
        update_phase_ratios_2D!(phase_ratios, (phase_host, phase_sill), xci, xvi)

        compute_melt_fraction!(ϕ, phase_ratios, rheology, (T=thermal.Tc, P=stokes.P))

        @show it += 1
        t        += dt
        push!(time_vec, t)
        push!(d18O_evo, round(mean(d18O[1:end-1, 1:end-1][ϕ .> 0.2]), digits=2))
        push!(melt_fraction_evo, round(maximum(ϕ), digits=2))

        # Data I/O and plotting ---------------------
        if it == 1 || rem(it, 15) == 0
            if igg.me == 0 && it == 1
                metadata(pwd(), checkpoint, joinpath(@__DIR__, "SillConvection.jl"), joinpath(@__DIR__, "SillModelSetup.jl"), joinpath(@__DIR__, "SillRheology.jl"))
            end
            checkpointing_jld2(checkpoint, stokes, thermal, t, dt, igg)
            # checkpointing_particles(checkpoint, particles; phases = pPhases, phase_ratios = phase_ratios, particle_args = particle_args, t = t, dt = dt)

            η_eff = @. stokes.τ.II / (2 * stokes.ε.II)
            (; η_vep, η) = stokes.viscosity


            if do_vtk
                velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
                data_v = (;
                    T = Array(T_WENO),
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
                    Melt_fraction = Array(ϕ),
                    EII_pl = Array(stokes.EII_pl),
                    stress_II = Array(stokes.τ.II),
                    strain_rate_II = Array(stokes.ε.II),
                    # plastic_strain_rate_II = Array(stokes.ε_pl.II),
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
            #ax2 = Axis(fig[2,1], aspect = DataAspect(), title = "Phase")
            ax2 = Axis(fig[2, 2][1, 1], aspect = DataAspect(), title = L"Density \;[\mathrm{kg/m}^{3}]", titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,)


            #ax3 = Axis(fig[1,3], aspect = DataAspect(), title = "log10(εII)")
            ax3 = Axis(fig[3, 1][1, 1], aspect = DataAspect(), title = L"Vy \;[\mathrm{m/s}]", titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,)

            #ax4 = Axis(fig[2,3], aspect = DataAspect(), title = "log10(η)")
            ax4 = Axis(fig[3, 2][1, 1], aspect = DataAspect(), title = L"\phi", titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,)
            # ax5 = Axis(fig[4, 1][1, 1], aspect = DataAspect(), title = L"log10(\eta)", titlesize=40,
            #         yticklabelsize=25,
            #         xticklabelsize=25,
            #         xlabelsize=25,)

            # Plot temperature
            h1  = heatmap!(
                ax1,
                xvi...,
                Array(thermal.T[2:(end - 1), :].-273); colormap=:lipari, colorrange=(host_rock_temp .-273.15, sill_temp.-273.15))

            h2  = heatmap!(
                ax2,
                xci...,
                Array(ρg[end]./9.81);
                colormap=:batlowW)

            # Plot 2nd invariant of strain rate
            #h3  = heatmap!(ax3, xci[1], xci[2], Array(log10.(stokes.ε.II)) , colormap=:batlow)

            # Plot vy velocity
            h3  = heatmap!(
                ax3,
                xvi...,
                Array(stokes.V.Vy); colormap=:batlow)

            # Plot effective viscosity
            #h4  = heatmap!(ax4, xci[1], xci[2], Array(log10.(η_vep)) , colormap=:batlow)

            # Plot melt fraction
            h4  = heatmap!(ax4,
                xci...,
                Array(ϕ);
                colormap=:lipari,
                colorrange=(0.0, 1.0))

            # h5  = heatmap!(ax5,
            #     xci...,
            #     log10.(Array(stokes.viscosity.η));
            #     colormap=:lipari)


            hidexdecorations!(ax1)
            hidexdecorations!(ax2)
            hideydecorations!(ax2)
            hideydecorations!(ax4)
            Colorbar(fig[2, 1][1, 2], h1, height = Relative(4/4), ticklabelsize=25, ticksize=15)
            Colorbar(fig[2, 2][1, 2], h2, height = Relative(4/4), ticklabelsize=25, ticksize=15)
            Colorbar(fig[3, 1][1, 2], h3, height = Relative(4/4), ticklabelsize=25, ticksize=15)
            Colorbar(fig[3, 2][1, 2], h4, height = Relative(4/4), ticklabelsize=25, ticksize=15)
            # Colorbar(fig[4, 1][1, 2], h5, height = Relative(4/4), ticklabelsize=25, ticksize=15)
            linkaxes!(ax1, ax2, ax3, ax4)
            fig
            figsave = joinpath(figdir, @sprintf("%06d.png", it))
            save(figsave, fig)

            # Plot time evolution of mean melt fraction with flow regime backgrounds
            let
                fig = Figure(size = (2000, 1000), title = "t = $t")

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
                    xvi[1],
                    xvi[2],
                    (Array(thermal.T[2:(end - 1), :].-273));
                    colormap=:lipari, colorrange=(host_rock_temp-273.15, sill_temp-273.15))
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
                    xci[1],
                    xci[2],
                    Array(log10.(Re));
                    colormap=:lapaz)

                Colorbar(fig1[1,2], h1, height = Relative(4/4), ticklabelsize=25, ticksize=15)
                ax2 = Axis(
                    fig1[2,1],
                    aspect = DataAspect(),
                    title = "δ18O",
                    # title = "Rayleigh number (log10)  (time = $(round(t/TimeScale, digits=2)) $TimeUnits)",
                    titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,
                )
                # # Plot isotopes
                h2  = heatmap!(
                    ax2,
                    xci[1],
                    xci[2],
                    # Array(log10.(Ra));
                    Array(d18O);
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
                    xci[1],
                    xci[2],
                    Array(log10.(Ra));
                    colormap=:lapaz, colorrange= (0, maximum(Array(log10.(Ra)))))
                Colorbar(fig1[1,4], h3, height = Relative(4/4), ticklabelsize=25, ticksize=15)

                ax4 = Axis(
                    fig1[2,3],
                    title = "δ18O evolution - mean(d18O[ϕ .> 0.2])",
                    titlesize=30,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,
                    ylabelsize=25,
                    xlabel = "Time ($TimeUnits)",
                    ylabel = "δ18O mean(ϕ > 0.2)"
                    )
                    l4 = lines!(
                        ax4,
                        time_vec ./ TimeScale,
                        d18O_evo;
                        color = :black,
                        linewidth = 2,
                    )
                save(joinpath(figdir, "Re_$(it).png"), fig1)
                fig1

                fig2 = Figure(size = (1600, 800), title = "Melt Fraction Evolution")

                ax = Axis(
                    fig2[1,1],
                    title = "Mean Melt Fraction Evolution",
                    titlesize=30,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,
                    ylabelsize=25,
                    xlabel = "Time ($TimeUnits)",
                    ylabel = "mean(ϕ)"
                )

                # Define regime boundaries
                porous_max = 0.08
                mush_max = 0.45
                suspension_max = 1.0

                # Plot colored backgrounds for regimes
                # Define x range for bands (span entire time axis)
                x_min = 0.0
                x_max = maximum(time_vec ./ TimeScale)
                x_band = [x_min, x_max]

                band!(
                    x_band,
                    [0.0, 0.0],
                    [porous_max, porous_max],
                    color = (:blue, 0.2),
                    # label = "Porous flow"
                )
                text!(
                    ax,
                    x_min + 0.01 * (x_max - x_min),  # near left edge
                    porous_max / 2,
                    text = "Porous flow",
                    align = (:left, :center),
                    fontsize = 40,
                    color = :blue,
                    font = "bold"
                )
                band!(
                    x_band,
                    [porous_max, porous_max],
                    [mush_max, mush_max],
                    color = (:orange, 0.5),
                    # label = "Mushy flow"
                )
                text!(
                    ax,
                    x_min + 0.01 * (x_max - x_min),
                    (porous_max + mush_max) / 2,
                    text = "Mushy flow",
                    align = (:left, :center),
                    fontsize = 40,
                    color = :orange,
                    font = "bold"
                )
                band!(
                    x_band,
                    [mush_max, mush_max],
                    [suspension_max, suspension_max],
                    color = (:red, 0.2),
                    # label = "Suspension flow"
                )
                text!(
                    ax,
                    x_min + 0.01 * (x_max - x_min),
                    (mush_max + suspension_max) / 2,
                    text = "Suspension flow",
                    align = (:left, :center),
                    fontsize = 40,
                    color = :red,
                    font = "bold"
                )

                lines!(
                    ax,
                    time_vec ./ TimeScale,
                    melt_fraction_evo;
                    color = :black,
                    linewidth = 2,
                    label = "mean(ϕ)"
                )

                # axislegend(ax, position=:lb, framevisible=true, fontsize=30)
                save(joinpath(figdir, "FlowRegime_diagram_$(it).png"), fig2)
                fig2
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
figdir   = "WENO5_SillConvection2D_$(today())"
n = 128
nx, ny = n, n #>> 1

sill_temp = 1273.15 # in K
host_rock_temp = 500.0 + 273.15 # in C
sill_size = 100 # in m
depth = 5e3 # in m
li = dimensions = (300.0, 200.0) # in m

igg = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 1; init_MPI=true)...)
else
    igg
end

# run main script
main(li, origin, igg; nx = nx, ny = ny, figdir = figdir, do_vtk = do_vtk, cutoff_visc = (1e1, 1.0e18), plotting = plotting, sill_temp = sill_temp, host_rock_temp = host_rock_temp, sill_size = sill_size, depth = depth);


## move away from GMG and initialize the phase distribution manually
