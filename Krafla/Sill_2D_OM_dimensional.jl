const isCUDA = true

@static if isCUDA
    using CUDA
    # CUDA.allowscalar(false)
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
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
const backend = @static if isCUDA
    CUDABackend        # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

using Printf, Statistics, LinearAlgebra, GeoParams, CairoMakie
import GeoParams.Dislocation
using GeophysicalModelGenerator
using WriteVTK, JLD2, Dates

# Load file with all the rheology configurations
include("SillRheology.jl")
include("SillModelSetup.jl")


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

## END OF HELPER FUNCTION ------------------------------------------------------------

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main2D(igg; figname="SillConvection2D", nx=64, ny=64, nz=64, do_vtk =false)

    #-----------------------------------------------------
    # USER INPUTS
    #-----------------------------------------------------

    # IO --------------------------------------------------
    # if it does not exist, make folder where figures are stored
    figdir = "./fig2D/$figname/"
    checkpoint = joinpath(figdir, "checkpoint")
    if do_vtk
        vtk_dir      = joinpath(figdir,"vtk")
        take(vtk_dir)
    end
    take(figdir)
    # ----------------------------------------------------
    # Set up the grid
    # ----------------------------------------------------
    li_GMG, origin_GMG, phases_GMG, T_GMG, Grid = SillSetup(nx+1,ny+1,nz+1)
    # -----------------------------------------------------
    # Set up the JustRelax model
    # -----------------------------------------------------
    lx              = li_GMG[1]              #  domain length in x-direction
    lz              = li_GMG[end]            #  domain length in y-direction
    li              = (lx, lz)                                              # domain length in x- and y-direction
    ni              = (nx, nz)                                              # number of grid points in x- and y-direction
    di              = @. li / ni                                            # grid spacing in x- and y-direction
    origin          = ntuple(Val(2)) do i
       origin_GMG[i]                       # origin coordinates of the domain
    end
    grid         = Geometry(ni, li; origin=origin)
    (; xci, xvi) = grid                                                     # nodes at the center and vertices of the cells

    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology     = init_rheologies(;)
    cutoff_visc  = (3e3,6.0e14)
    # κ            = (4 / (compute_heatcapacity(rheology[1].HeatCapacity[1].Cp) * 2900.0))
    # κ            = (4 / (compute_heatcapacity(rheology[1].HeatCapacity[1].Cp) * rheology[1].Density[1].ρ))
    κ            = (4 / (1050 * rheology[1].Density[1].ρ))
    # κ            = (4 / (rheology[1].HeatCapacity[1].Cp * rheology[1].Density[1].ρ))
    dt = dt_diff = (0.5 * min(di...)^2 / κ / 2.01) # diffusive CFL timestep limiter
    # dt = dt_diff = 0.5 * min(di...)^2 / κ / 2.01 / 1000# diffusive CFL timestep limiter
   # dt = dt_diff/10
    @show dt
    # ----------------------------------------------------
    # Weno model -----------------------------------------
    weno = WENO5(backend_JR, Val(2), ni.+1) # ni.+1 for Temp

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 30, 40, 20
    particles        = init_particles(backend, nxcell, max_xcell, min_xcell, xvi...);

    # subgrid_arrays = SubgridDiffusionCellArrays(particles)

    # velocity grids
    grid_vx, grid_vy = velocity_grids(xci, xvi, di)
    # temperature
    pT, pPhases      = init_cell_arrays(particles, Val(2));
    particle_args    = (pT, pPhases);

    # Assign material phases --------------------------
    phases_dev   = PTArray(backend_JR)(phases_GMG)
    phase_ratios = PhaseRatios(backend, length(rheology), ni);
    init_phases!(pPhases, phases_dev, particles, xvi)
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)

    thermal         = ThermalArrays(backend_JR, ni)
    @views thermal.T[2:end-1, :] .= PTArray(backend_JR)(T_GMG)
    thermal_bc      = TemperatureBoundaryConditions(;
        no_flux     = (left = true, right = true, top = false, bot = false),
    )
    thermal_bcs!(thermal, thermal_bc)
    temperature2center!(thermal)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes          = StokesArrays(backend_JR, ni)
    pt_stokes       = PTStokesCoeffs(li, di; Re=3π, ϵ=1e-4, CFL=0.9 / √2.1) #ϵ=1e-4,  CFL=1 / √2.1 CFL=0.27 / √2.1
    # ----------------------------------------------------

    args = (; T=thermal.Tc, P=stokes.P, dt=dt)

    pt_thermal = PTThermalCoeffs(
        backend_JR, rheology, phase_ratios, args, dt, ni, di, li; ϵ=1e-5, CFL=0.98 / √2.1
    )

    # Melt Fraction
    ϕ = @zeros(ni...)

    # Buoyancy force
    ρg = @zeros(ni...), @zeros(ni...)                      # ρg[1] is the buoyancy force in the x direction, ρg[2] is the buoyancy force in the y direction

    for _ in 1:5
        compute_ρg!(ρg[end], phase_ratios, rheology, (T=thermal.Tc, P=stokes.P))
        @parallel init_P!(stokes.P, ρg[2], xci[2])
    end
    # Rheology
    compute_melt_fraction!(
        ϕ, phase_ratios, rheology, (T=thermal.Tc, P=stokes.P)
    )
    compute_viscosity!(stokes, phase_ratios, args, rheology, cutoff_visc)

    @copy stokes.P0 stokes.P
    @copy thermal.Told thermal.T

    # Boundary conditions
    flow_bcs         = VelocityBoundaryConditions(;
        free_slip    = (left = true, right=true, top=true, bot=true),
    )
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

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
        save(joinpath(figdir, "initial_profile.png"), fig)
        fig
    end

    # WENO arrays
    T_WENO  = @zeros(ni.+1)
    Vx_v = @zeros(ni.+1...)
    Vy_v = @zeros(ni.+1...)

    local Vx_v, Vy_v, T_WENO
    # Time loop
    t, it = 0.0, 0


    # while it < 30e3
    while it < 100e4
        # Update buoyancy and viscosity -
        args = (; T = thermal.Tc, P = stokes.P,  dt=dt, ϕ= ϕ)
        compute_melt_fraction!(ϕ, phase_ratios, rheology, (T=thermal.Tc, P=stokes.P))
        compute_viscosity!(stokes, phase_ratios, args, rheology, cutoff_visc)
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
            dt,
            igg;
            kwargs = (;
                iterMax          = 250e3,#250e3,
                free_surface     = false,
                nout             = 2e3,#5e3,
                viscosity_cutoff = cutoff_visc,
                viscosity_relaxation=1e-3,
            )
        )
        tensor_invariant!(stokes.ε)
        ## Save the checkpoint file before a possible thermal solver blow up
        checkpointing_jld2(joinpath(checkpoint, "thermal"), stokes, thermal, t, dt, igg)

        dt   = compute_dt(stokes, di, dt_diff)

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
        # ------------------------------

        T_WENO .= @views thermal.T[2:end-1, :]
        velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
        WENO_advection!(T_WENO, (Vx_v, Vy_v), weno, di, dt)
        thermal.T[2:end-1, :] .= T_WENO

                # Advection --------------------
        # advect particles in space
        advection_MQS!(particles, RungeKutta2(), @velocity(stokes), (grid_vx, grid_vy), dt)
        # advect particles in memory
        move_particles!(particles, xvi, particle_args)
        # check if we need to inject particles
        inject_particles_phase!(particles, pPhases, (pT, ), (T_WENO,), xvi)
        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
        @show it += 1
        t        += dt
        @show extrema(thermal.T)
        any(isnan.(thermal.T)) && break

        # Data I/O and plotting ---------------------
        if it == 1 || rem(it, 10) == 0
            if igg.me == 0 && it == 1
                metadata(pwd(), checkpoint, basename(@__FILE__), "SillModelSetup.jl", "SillRheology.jl")
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

            velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)

            if do_vtk
                data_v = (;
                    T   = Array(thermal.T[2:(end - 1), :].-273),
                    τxy = Array(stokes.τ.xy),
                    εxy = Array(stokes.ε.xy),
                    Vx  = Array(Vx_v),
                    Vy  = Array(Vy_v),
                )
                data_c = (;
                    P   = Array(stokes.P),
                    τxx = Array(stokes.τ.xx),
                    τyy = Array(stokes.τ.yy),
                    τII = Array(stokes.τ.II),
                    εxx = Array(stokes.ε.xx),
                    εyy = Array(stokes.ε.yy),
                    εII = Array(stokes.ε.II),
                    η   = Array(stokes.viscosity.η),
                    ρ   = Array(ρg[end]./9.81),
                )
                velocity_v = (
                    Array(Vx_v),
                    Array(Vy_v),
                )
                save_vtk(
                    joinpath(vtk_dir, "vtk_" * lpad("$it", 6, "0")),
                    xvi,
                    xci,
                    data_v,
                    data_c,
                    velocity_v
                )
            end

            # Make particles plottable
            p        = particles.coords
            ppx, ppy = p
            pxv      = ppx.data[:]./1e3
            pyv      = ppy.data[:]./1e3
            clr      = pPhases.data[:]
            idxv     = particles.index.data[:];

            # Make Makie figure
            fig = Figure(size = (2000, 1000), title = "t = $t", )
            ar = DataAspect()
            ax0 = Axis(
                fig[1, 1:2];
                aspect=ar,
                title="t=$(round((t/(3600*24*365)))) years",
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
                Array(thermal.T[2:(end - 1), :].-273); colormap=:lipari, colorrange=(400, 1200))

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
                colormap=:lipari)

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

            let
                fig = Figure(size = (2000, 1000), title = "t = $t")
                ax1 = Axis(fig[1,1], aspect = DataAspect(), title = "T [C]  (t=$(round((t/(3600)))) hours)",  titlesize=40,
                yticklabelsize=25,
                xticklabelsize=25,
                xlabelsize=25,)
                 # Plot temperature
                h1  = heatmap!(
                    ax1,
                    xvi[1],
                    xvi[2],
                    (Array(thermal.T[2:(end - 1), :].-273));
                    colormap=:lipari, colorrange=(400, 1200))
                Colorbar(fig[1,2], h1, height = Relative(4/4), ticklabelsize=25, ticksize=15)
                save(joinpath(figdir, "Temperature_$(it).png"), fig)
                fig
            end

        end
        # ------------------------------

    end

    return nothing
end
## END OF MAIN SCRIPT ----------------------------------------------------------------


# (Path)/folder where output data and figures are stored
figname   = "$(today())_Krafla_Sill_Geometry_new_rheology"
do_vtk = true
ar = 1 # aspect ratio
n = 256
nx = n * ar #608
ny = n #512
nz = n #512
igg = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 1; init_MPI=true)...)
else
    igg
end

# run main script
main2D(igg; figname=figname, nx=nx, ny=ny, nz=nz, do_vtk=do_vtk);
