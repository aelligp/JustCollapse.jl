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
using GeoParams, CairoMakie, CellArrays, JLD2
using Dates

# Load file with all the rheology configurations
include("Caldera_setup.jl")
include("Caldera_rheology.jl")

## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------

import ParallelStencil.INDICES
const idx_k = INDICES[2]
macro all_k(A)
    esc(:($A[$idx_k]))
end

function copyinn_x!(A, B)
    @parallel function f_x(A, B)
        @all(A) = @inn_x(B)
        return nothing
    end

    @parallel f_x(A, B)
end

# Initial pressure profile - not accurate
@parallel function init_P!(P, ρg, z)
    @all(P) = abs(@all(ρg) * @all_k(z)) * <(@all_k(z), 0.0)
    return nothing
end

function apply_pure_shear(Vx,Vy, εbg, xvi, lx, ly)
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
    @parallel_indices (I...) function _phase_change!(phases, px, py , index, cx, cy, air_phase, init)

        @inbounds for ip in cellaxes(phases)
            # quick escape
            @index(index[ip, I...]) == 0 && continue

            x = @index px[ip, I...]
            y = @index py[ip, I...]
            chain_x = @index cx[ip, I[1]]
            chain_y = @index cy[ip, I[1]]


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

    @parallel (@idx ni) _phase_change!(phases, particles.coords..., particles.index, chain.coords..., air_phase, init)
end

function extract_topo_from_GMG_phase(phases_GMG, xvi, air_phase)
    topo_idx = [findfirst(x->x==air_phase, row) - 1 for row in eachrow(phases_GMG)]
    yv = xvi[2]
    topo_y = yv[topo_idx]
    return topo_y
end

# air_phase = 5
# topo_y =  extract_topo_from_GMG_phase(phases_GMG, xvi, air_phase)
# lines(topo_y)

## END OF HELPER FUNCTION ------------------------------------------------------------

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main(li, origin, phases_GMG, T_GMG, igg; nx=16, ny=16, figname="figs2D", do_vtk =false)

    # IO --------------------------------------------------
    # if it does not exist, make folder where figures are stored
    figdir = "./fig2D/$figname/"
    checkpoint = joinpath(figdir, "checkpoint")
    take(figdir)
    if do_vtk
        vtk_dir      = joinpath(figdir,"vtk")
        take(vtk_dir)
    end
    # ----------------------------------------------------

    # Physical domain ------------------------------------
    ni                  = nx, ny           # number of cells
    di                  = @. li / ni       # grid steps
    grid                = Geometry(ni, li; origin = origin)
    (; xci, xvi)        = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology            = init_rheologies()
    dt                  = 5e2 * 3600 * 24 * 365 # diffusive CFL timestep limiter
    # dt                  = Inf # diffusive CFL timestep limiter
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell              = 40
    max_xcell           = 60
    min_xcell           = 20
    particles           = init_particles(
        backend_JP, nxcell, max_xcell, min_xcell, xvi, di, ni
    )
    subgrid_arrays      = SubgridDiffusionCellArrays(particles)
    # velocity grids
    grid_vxi            = velocity_grids(xci, xvi, di)
    # material phase & temperature
    pPhases, pT         = init_cell_arrays(particles, Val(2))

    # particle fields for the stress rotation
    pτ  = pτxx, pτyy, pτxy        = init_cell_arrays(particles, Val(3)) # stress
    # pτ_o = pτxx_o, pτyy_o, pτxy_o = init_cell_arrays(particles, Val(3)) # old stress
    pω   = pωxy,                  = init_cell_arrays(particles, Val(1)) # vorticity
    particle_args                 = (pT, pPhases, pτ..., pω...)
    particle_args_reduced         = (pT, pτ..., pω...)

    # Assign particles phases anomaly
    phases_device    = PTArray(backend)(phases_GMG)
    phase_ratios     = phase_ratios = PhaseRatios(backend_JP, length(rheology), ni);
    init_phases!(pPhases, phases_device, particles, xvi)
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)

    # rock ratios for variational stokes
    # RockRatios
    air_phase   = 5
    ϕ           = RockRatio(backend, ni)
    update_rock_ratio!(ϕ, phase_ratios, air_phase)

    # Initialize marker chain-------------------------------
    nxcell, max_xcell, min_xcell = 100, 150, 75
    initial_elevation            = 0e0
    chain                        = init_markerchain(backend_JP, nxcell, min_xcell, max_xcell, xvi[1], initial_elevation);
    topo_y                       = extract_topo_from_GMG_phases(phases_GMG, xvi, air_phase)
    for _ in 1:3
        hn                      = 0.5 .* (topo_y[1:end-1] .+ topo_y[2:end])
        @views topo_y[2:end-1] .= 0.5 .* (hn[1:end-1] .+ hn[2:end])
        fill_chain_from_vertices!(chain, topo_y)
        update_phases_given_markerchain!(pPhases, chain, particles, origin, di, air_phase)
    end
    # update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
    # update_cell_halo!(particles.coords..., particle_args...);
    # update_cell_halo!(particles.index)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes           = StokesArrays(backend, ni)
    pt_stokes        = PTStokesCoeffs(li, di; ϵ=1e-4, Re = 3e0, r=0.7, CFL = 0.98 / √2.1) # Re=3π, r=0.7
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal          = ThermalArrays(backend, ni)
    @views thermal.T[2:end-1, :] .= PTArray(backend)(T_GMG)
    thermal_bc       = TemperatureBoundaryConditions(;
        no_flux      = (left = true, right = true, top = false, bot = false),
    )
    thermal_bcs!(thermal, thermal_bc)
    temperature2center!(thermal)
    Ttop             = thermal.T[2:end-1, end]
    Tbot             = thermal.T[2:end-1, 1]
    # ----------------------------------------------------

    # Buoyancy forces
    ρg               = ntuple(_ -> @zeros(ni...), Val(2))
    compute_ρg!(ρg[2], phase_ratios, rheology, (T=thermal.Tc, P=stokes.P))
    stokes.P        .= PTArray(backend)(reverse(cumsum(reverse((ρg[2]).* di[2], dims=2), dims=2), dims=2))

    # Melt fraction
    ϕ_m = @zeros(ni...)

    compute_melt_fraction!(
        ϕ_m, phase_ratios, rheology, (T=thermal.Tc, P=stokes.P)
    )

    # Rheology
    args0            = (T=thermal.Tc, P=stokes.P, dt = Inf)
    viscosity_cutoff = (1e18, 1e23)
    compute_viscosity!(stokes, phase_ratios, args0, rheology, air_phase, viscosity_cutoff)

    # PT coefficients for thermal diffusion
    pt_thermal       = PTThermalCoeffs(
        backend, rheology, phase_ratios, args0, dt, ni, di, li; ϵ=1e-8, CFL=0.95 / √2
    )

    # Boundary conditions
    # flow_bcs         = DisplacementBoundaryConditions(;
    flow_bcs         = VelocityBoundaryConditions(;
        free_slip    = (left = true , right = true , top = true , bot = true),
        free_surface = false,
    )

    # U            = 0.02
    # stokes.U.Ux .= PTArray(backend)([(x - li[1] * 0.5) * U / dt for x in xvi[1], _ in 1:ny+2])
    # stokes.U.Uy .= PTArray(backend)([-y * U / dt for _ in 1:nx+2, y in xvi[2]])
    # flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    # displacement2velocity!(stokes, dt)

    εbg          = 1e-14 * 1
    apply_pure_shear(@velocity(stokes)..., εbg, xvi, li...)

    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

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

    local Vx_v, Vy_v, Vx, Vy
    if do_vtk
        Vx_v = @zeros(ni.+1...)
        Vx_v = @zeros(ni.+1...)
        Vx   = @zeros(ni...)
        Vy   = @zeros(ni...)
    end

    # ----------------------------------------------------
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
        ax1 = Axis(fig[1, 1]; aspect=2 / 3, title="T [K]")
        ax2 = Axis(fig[1, 2]; aspect=2 / 3, title="Pressure [MPa]")
        scatter!(
            ax1,
            Array(thermal.T[2:(end - 1), :][:]),
            Yv ./1e3,
        )
        lines!(
            ax2,
            Array(stokes.P[:] ./ 1e6),
            Y./1e3,
        )
        hideydecorations!(ax2)
        # save(joinpath(figdir, "initial_profile.png"), fig)
        fig
    end

    τxx_v = @zeros(ni.+1...)
    τyy_v = @zeros(ni.+1...)

    # Time loop
    t, it = 0.0, 0
    thermal.Told .= thermal.T

    while it < 100 #000 # run only for 5 Myrs

        # interpolate fields from particle to grid vertices
        particle2grid!(T_buffer, pT, xvi, particles)
        @views T_buffer[:, end]      .= Ttop
        @views T_buffer[:, 1]        .= Tbot
        @views thermal.T[2:end-1, :] .= T_buffer
        thermal_bcs!(thermal, thermal_bc)
        temperature2center!(thermal)

        args = (; T=thermal.Tc, P=stokes.P, dt=Inf, ΔTc=thermal.ΔTc)
        # args = (; T=thermal.Tc, P=stokes.P, dt=Inf)

        t_stokes = @elapsed solve_VariationalStokes!(
            stokes,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            phase_ratios,
            ϕ,
            rheology,
            args,
            dt,
            igg;
            kwargs = (;
                iterMax          = 100e3,
                nout             = 2e3,
                viscosity_cutoff = viscosity_cutoff,
            )
        )

        center2vertex!(τxx_v, stokes.τ.xx)
        center2vertex!(τyy_v, stokes.τ.yy)
        centroid2particle!(pτxx , xci, stokes.τ.xx, particles)
        centroid2particle!(pτyy , xci, stokes.τ.yy, particles)
        grid2particle!(pτxy, xvi, stokes.τ.xy, particles)
        grid2particle!(pωxy, xvi, stokes.ω.xy, particles)
        rotate_stress_particles!(pτ, pω, particles, dt)

        println("Stokes solver time             ")
        println("   Total time:      $t_stokes s")
        tensor_invariant!(stokes.ε)
        tensor_invariant!(stokes.ε_pl)
        dtmax = 8e2 * 3600 * 24 * 365.25
        dt    = compute_dt(stokes, di, dtmax)

        println("dt = $(dt/(3600 * 24 *365.25)) years")
        # ------------------------------

        # Thermal solver ---------------
        heatdiffusion_PT!(
            thermal,
            pt_thermal,
            thermal_bc,
            rheology,
            args,
            dt,
            di;
            kwargs = (
                igg     = igg,
                phase   = phase_ratios,
                iterMax = 50e3,
                nout    = 1e2,
                verbose = true,
            )
        )
        thermal.ΔT .= thermal.T .- thermal.Told
        vertex2center!(thermal.ΔTc, thermal.ΔT[2:end-1, :])

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

        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection!(particles, RungeKutta2(), @velocity(stokes), grid_vxi, dt)
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

        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
        update_rock_ratio!(ϕ, phase_ratios, air_phase)

        particle2centroid!(stokes.τ.xx, pτxx, xci, particles)
        particle2centroid!(stokes.τ.yy, pτyy, xci, particles)
        particle2grid!(stokes.τ.xy, pτxy, xvi, particles)

        compute_melt_fraction!(
            ϕ_m, phase_ratios, rheology, (T=thermal.Tc, P=stokes.P)
        )

        advect_markerchain!(chain, RungeKutta2(), @velocity(stokes), grid_vxi, dt)
        update_phases_given_markerchain!(pPhases, chain, particles, origin, di, air_phase)

        @show it += 1
        t        += dt

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
        @views ϕ_nohalo     .= Array(ϕ_m[2:end-1, 2:end-1]) # Copy data to CPU removing the halo
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

        # Data I/O and plotting ---------------------
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

            (; η_vep, η) = stokes.viscosity
            if do_vtk
                velocity2vertex!(Vx, Vy, @velocity(stokes)...)
                data_c = (;
                    T = T_v,
                    P = P_v,
                    τII = τII_v,
                    εII = εII_v,
                    εII_pl = εII_pl_v,
                    EII_pl = EII_pl_v,
                    ϕ = ϕ_v,
                    ρ = ρg_v./ 9.81,
                    η = η_v,
                    η_vep   = η_vep_v,
                    Vx = Array(Vxv_v),
                    Vy = Array(Vyv_v),
                    phases = phases_c_v
                )
                velocity = (
                    Array(Vxv_v),
                    Array(Vyv_v),
                )
                save_vtk(
                    joinpath(vtk_dir, "vtk_" * lpad("$(it)_$(igg.me)", 6, "0")),
                    xci_v,
                    data_c,
                    velocity,
                    t=(round.(t/(3600 * 24 *365.25 * 1e3)); digits=3)
                )
            end

            # Make particles plottable
            p        = particles.coords
            ppx, ppy = p
            pxv      = ppx.data[:]
            pyv      = ppy.data[:]
            clr      = pPhases.data[:]
            # clr      = pT.data[:]
            idxv     = particles.index.data[:];
            # Make Makie figure
            ar  = 2
            fig = Figure(size = (1200, 900), title = "t = $t")
            ax1 = Axis(fig[1,1], aspect = ar, title = "T [K]  (t=$(round.(t/(3600 * 24 *365.25 * 1e3); digits=3)) Kyrs)")
            ax2 = Axis(fig[2,1], aspect = ar, title = "Vy")
            # ax2 = Axis(fig[2,1], aspect = ar, title = "Phase")
            ax3 = Axis(fig[1,3], aspect = ar, title = "τII [MPa]")
            # ax4 = Axis(fig[2,3], aspect = ar, title = "log10(εII)")
            ax4 = Axis(fig[2,3], aspect = ar, title = "log10(η)")
            # Plot temperature
            h1  = heatmap!(ax1, xvi[1], xvi[2], Array(thermal.T[2:end-1,:]) , colormap=:batlow)
            # Plot particles phase
            h2  = scatter!(ax2, Array(pxv[idxv]), Array(pyv[idxv]), color=Array(clr[idxv]), markersize = 1)
            # h2  = heatmap!(ax2, xvi[1], xvi[2], Array(stokes.V.Vy) , colormap=:batlow)
            # Plot 2nd invariant of strain rate
            # h3  = heatmap!(ax3, xci[1], xci[2], Array(log10.(stokes.ε_pl.II)) , colormap=:batlow)
            h3  = heatmap!(ax3, xci[1], xci[2], Array(stokes.τ.II)./1e6 , colormap=:batlow)
            # Plot effective viscosity
            # h4  = heatmap!(ax4, xci[1], xci[2], Array(log10.(stokes.ε.II)) , colormap=:lipari)
            h4  = heatmap!(ax4, xci[1], xci[2], Array(log10.(stokes.viscosity.η_vep)), colorrange=log10.(viscosity_cutoff), colormap=:batlow)
            hidexdecorations!(ax1)
            hidexdecorations!(ax2)
            hidexdecorations!(ax3)
            Colorbar(fig[1,2], h1)
            Colorbar(fig[2,2], h2)
            Colorbar(fig[1,4], h3)
            Colorbar(fig[2,4], h4)
            linkaxes!(ax1, ax2, ax3, ax4)
            fig
            save(joinpath(figdir, "$(it).png"), fig)

            ## Plot initial T and P profile
            fig = let
                Yv = [y for x in xvi[1], y in xvi[2]][:]
                Y = [y for x in xci[1], y in xci[2]][:]
                fig = Figure(; size=(1200, 900))
                ax1 = Axis(fig[1, 1]; aspect=2 / 3, title="T")
                ax2 = Axis(fig[1, 2]; aspect=2 / 3, title="Pressure")
                scatter!(
                    ax1,
                    Array(thermal.T[2:(end - 1), :][:]),
                    Yv,
                )
                lines!(
                    ax2,
                    Array(stokes.P[:] ./ 1e6),
                    Y,
                )
                hideydecorations!(ax2)
                save(joinpath(figdir, "thermal_profile_$it.png"), fig)
                fig
            end

            fig1 = let
                fig = Figure(size = (900, 900), title = "t = $t")
                ax  = Axis(fig[1,1], aspect = 1, title = " t=$(round.(t/(1e3 * 3600 * 24 *365.25); digits=3)) Kyrs")

                # Make particles plottable
                p        = particles.coords
                ppx, ppy = p
                pxv      = ppx.data[:]./1e3
                pyv      = ppy.data[:]./1e3
                clr      = pPhases.data[:]
                idxv     = particles.index.data[:];

                chain_x = chain.coords[1].data[:]./1e3
                chain_y = chain.coords[2].data[:]./1e3

                scatter!(ax, Array(pxv[idxv]), Array(pyv[idxv]), color=Array(clr[idxv]), markersize = 5)
                # arrows!(
                #     ax,
                #     xvi[1][1:nt:end-1]./1e3, xvi[2][1:nt:end-1]./1e3, Array.((Vx_v[1:nt:end-1, 1:nt:end-1], Vy_v[1:nt:end-1, 1:nt:end-1]))...,
                #     lengthscale = 25 / max(maximum(Vx_v),  maximum(Vy_v)),
                #     color = :gray,
                # )
                scatter!(ax, Array(chain_x), Array(chain_y), color=:red, markersize = 5)
                scatter!(ax, xci[1]./1e3, topo./1e3, color=:green, markersize=5)

                fig
            end
        end
        # ------------------------------

    end

    return nothing
end

## END OF MAIN SCRIPT ----------------------------------------------------------------
do_vtk   = true # set to true to generate VTK files for ParaView
# figdir   = "Caldera2D_noPguess"
figname   = "$(today())_Caldera2D_1e-14"
n        = 480# * 2
nx, ny   = n, n #>>> 1

igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI= true)...)
else
    igg
end

sticky_air = 5.0
x_global = range(0.0, 50, nx_g());
z_global      = range(-25e0, sticky_air, ny_g());
origin = (x_global[1], z_global[1])
li = (abs(last(x_global)-first(x_global)), abs(last(z_global)-first(z_global)))

ni           = nx, ny           # number of cells
di           = @. li / (nx_g(), ny_g())           # grid steps
grid_global  = Geometry(ni, li; origin = origin)

# Calculate magma volume of ellipsoid
li, origin, phases_GMG, T_GMG = setup2D(
    grid_global.xvi,
    nx+1, ny+1, igg;
    flat           = false,
    chimney        = true,
    volcano_size   = (3e0, 5e0),
    chamber_T      = 927e0, # in Celsius
    chamber_depth  = 5e0,
    chamber_radius = 1.0,
    aspect_x       = 5,
)
V = 4/3 * π * (1.0*5) * 1.75 * 1.0
println("Magma volume of the initial chamber: $(round(V; digits=3)) km³")
heatmap(phases_GMG)
# heatmap(T_GMG)



main(li, origin, phases_GMG, T_GMG, igg; figname = figname, nx = nx, ny = ny, do_vtk = do_vtk);


# p = particles.coords
# # pp = [argmax(p) for p in phase_ratios.center] #if you want to plot it in a heatmap rather than scatter
# ppx, ppy = p


# pxv = ppx.data[:]
# pyv = ppy.data[:]
# # clr = pPhases.data[:]
# # clrT = pT.data[:]
# px = chain.coords[1].data[:];
# py = chain.coords[2].data[:];
# idxv = particles.index.data[:]
# f,ax,h=scatter(Array(pxv[idxv]), Array(pyv[idxv]), color=Array(clr[idxv]), colormap=:roma, markersize=1)
# # scatter!(ax,xci[1], topo, color=:black, markersize=3)
# scatter!(px, py, color=:black)#, markersize=1)
# Colorbar(f[1,2], h)
# f
