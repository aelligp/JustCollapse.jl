const isCUDA = false

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
import JustPIC._2D.cellaxes#, JustPIC._2D.phase_ratios_center!
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
    @parallel (1:nx - 2, 1:ny-2, 1:nz) pure_shear_y!(Vz, εbg, lz)

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

# function phase_change!(phases, particles)
#     ni = size(phases)
#     @parallel_indices (I...) function _phase_change!(phases, px, py, index)

#         @inbounds for ip in cellaxes(phases)
#             #quick escape
#             @index(index[ip, I...]) == 0 && continue

#             x = @index px[ip,I...]
#             y = (@index py[ip,I...])
#             phase_ij = @index phases[ip, I...]
#             if y > 0.0 && (phase_ij  == 2.0 || phase_ij  == 3.0)
#                 @index phases[ip, I...] = 4.0
#             end
#         end
#         return nothing
#     end

#     @parallel (@idx ni) _phase_change!( phases, particles.coords..., particles.index)
# end

# function phase_change!(phases, EII_pl, threshold, particles)
#     ni = size(phases)
#     @parallel_indices (I...) function _phase_change!(phases, EII_pl, threshold, px, py, index)

#         @inbounds for ip in cellaxes(phases)
#             #quick escape
#             @index(index[ip, I...]) == 0 && continue

#             x = @index px[ip,I...]
#             y = (@index py[ip,I...])
#             phase_ij = @index phases[ip, I...]
#             EII_pl_ij = @index EII_pl[ip, I...]
#             if EII_pl_ij > threshold && (phase_ij < 4.0)
#                 @index phases[ip, I...] = 2.0
#             end
#         end
#         return nothing
#     end

#     @parallel (@idx ni) _phase_change!(phases, EII_pl, threshold, particles.coords..., particles.index)
# end

# function phase_change!(phases, melt_fraction, threshold, sticky_air_phase, particles)
#     ni = size(phases)
#     @parallel_indices (I...) function _phase_change!(phases, melt_fraction, threshold, sticky_air_phase, px, py, index)

#         @inbounds for ip in cellaxes(phases)
#             #quick escape
#             @index(index[ip, I...]) == 0 && continue

#             x = @index px[ip,I...]
#             y = (@index py[ip,I...])
#             phase_ij = @index phases[ip, I...]
#             melt_fraction_ij = @index melt_fraction[ip, I...]
#             if melt_fraction_ij < threshold && (phase_ij < sticky_air_phase)
#                 @index phases[ip, I...] = 1.0
#             end
#         end
#         return nothing
#     end

#     @parallel (@idx ni) _phase_change!(phases, melt_fraction, threshold, sticky_air_phase, particles.coords..., particles.index)
# end

# function circular_perturbation!(T, δT, max_temperature, xc_anomaly, yc_anomaly, r_anomaly, xvi)

#     @parallel_indices (i, j) function _circular_perturbation!(T, δT, max_temperature, xc_anomaly, yc_anomaly, r_anomaly, x, y)
#         @inbounds if  ((x[i] - xc_anomaly)^2 + (y[j] - yc_anomaly)^2 ≤ r_anomaly^2)
#             new_temperature = T[i+1, j] * (δT / 100 + 1)
#             T[i+1, j] = new_temperature > max_temperature ? max_temperature : new_temperature
#         end
#         return nothing
#     end

#     nx, ny = size(T)

#     @parallel (1:nx-2, 1:ny) _circular_perturbation!(T, δT, max_temperature, xc_anomaly, yc_anomaly, r_anomaly, xvi...)
# end

# function new_thermal_anomaly!(phases, particles, xc_anomaly, yc_anomaly, r_anomaly)
#     ni = size(phases)

#     @parallel_indices (I...) function new_anomlay_particles(phases, px, py, index, xc_anomaly, yc_anomaly, r_anomaly)
#         @inbounds for ip in cellaxes(phases)
#             @index(index[ip, I...]) == 0 && continue

#             x = @index px[ip, I...]
#             y = @index py[ip, I...]

#             # thermal anomaly - circular
#             if ((x - xc_anomaly)^2 + (y - yc_anomaly)^2 ≤ r_anomaly^2)
#                 @index phases[ip, I...] = 3.0
#             end
#         end
#         return nothing
#     end
#     @parallel (@idx ni) new_anomlay_particles(phases, particles.coords..., particles.index, xc_anomaly, yc_anomaly, r_anomaly)
# end

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
    cutoff_visc  = nondimensionalize((1e15Pa*s, 1e24Pa*s),CharDim)
    κ            = (4 / (rheology[1].HeatCapacity[1].Cp.Cp.val * rheology[1].Density[1].ρsolid.ρ0.val))                                 # thermal diffusivity
    # κ            = (4 / (rheology[1].HeatCapacity[1].Cp.val * rheology[1].Density[1].ρ0.val))                                 # thermal diffusivity
    # κ            = (4 / (rheology[2].HeatCapacity[1].Cp.Cp.val * rheology[2].Density[1].ρ0.val))                                 # thermal diffusivity
    dt           = dt_diff = 0.5 * min(di...)^2 / κ / 2.01

    # Initialize particles ----------------------------------
    nxcell, max_xcell, min_xcell = 150, 175, 125
    particles        = init_particles(backend, nxcell, max_xcell, min_xcell, xvi, di, ni);
    subgrid_arrays   = SubgridDiffusionCellArrays(particles);
    # velocity grids
    grid_vx, grid_vy = velocity_grids(xci, xvi, di);
    # temperature
    pT, pT0, pPhases = init_cell_arrays(particles, Val(3));
    particle_args       = (pT, pT0, pPhases);

    # Assign material phases --------------------------
    phases_dev   = PTArray(backend_JR)(phases_GMG)
    phase_ratios = PhaseRatios(backend, length(rheology), ni);
    init_phases!(pPhases, phases_dev, particles, xvi)
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)

    update_cell_halo!(particles.coords..., particle_args...);
    update_cell_halo!(particles.index)

    thermal         = ThermalArrays(backend_JR, ni)
    thermal.T       .= PTArray(backend_JR)(nondimensionalize(T_GMG.*C, CharDim))
    thermal_bc      = TemperatureBoundaryConditions(;
        no_flux     = (left = true , right = true , top = false, bot = false, front = true , back = true),
    )
    thermal_bcs!(thermal, thermal_bc)
    temperature2center!(thermal)


    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes          = StokesArrays(backend_JR, ni)
    pt_stokes       = PTStokesCoeffs(li, di; ϵ=1e-5,  CFL = 0.99 / √3.1) #ϵ=1e-4,  CFL=1 / √2.1 CFL=0.27 / √2.1
    # -----------------------------------------------------
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
    ϕ = @zeros(ni...)

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

    args = (; ϕ=ϕ, T=thermal.Tc, P=stokes.P, dt=dt, ΔTc=thermal.ΔTc, perturbation_C = perturbation_C)

    for _ in 1:5
        compute_ρg!(ρg[end], phase_ratios, rheology, (T=thermal.Tc, P=stokes.P))
        @parallel init_P!(stokes.P, ρg[3], xci[3])
    end

    compute_melt_fraction!(
        ϕ, phase_ratios, rheology, (T=thermal.T, P=stokes.P)
    )
    compute_viscosity!(stokes, phase_ratios, args, rheology, cutoff_visc)

    @copy stokes.P0 stokes.P
    @copy thermal.Told thermal.T

    t, it      = 0.0, 0
    interval   = 1.0
    dt_new     = dt
    local Vx, Vy, Vz
    if do_vtk
        Vx = @zeros(ni...)
        Vy = @zeros(ni...)
        Vz = @zeros(ni...)
    end

    pT0.data    .= pT.data

    ## Do a incompressible stokes solve, to get a better initial guess for the compressible stokes
    ## solver. And after that apply the extension/compression BCs. This is done to avoid the
    ## incompressible stokes solver to blow up.
    println("Starting incompressible stokes solve")
    solve!(
        stokes,
        pt_stokes,
        di,
        flow_bcs,
        ρg,
        phase_ratios,
        rheology_incomp,
        args,
        dt*0.1,
        igg;
        kwargs = (;
            iterMax          = 100e3,#250e3,
            free_surface     = true,
            nout             = 2e3,#5e3,
            viscosity_cutoff = cutoff_visc,
            verbose          = false,
        )
    )
    tensor_invariant!(stokes.ε)
    heatdiffusion_PT!(
        thermal,
        pt_thermal,
        thermal_bc,
        rheology_incomp,
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

    # if shear == true && DisplacementFormulation == true
    #     BC_displ!(@displacement(stokes)..., εbg, xvi,lx,lz,dt)
    #     flow_bcs = DisplacementBoundaryConditions(;
    #         free_slip    = (left = true , right = true , top = true , bot = true , front = true , back = true ),
    #         no_slip      = (left = false, right = false, top = false, bot = false, front = false, back = false),
    #     )
    #     flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    #     displacement2velocity!(stokes, dt) # convert displacement to velocity
    #     update_halo!(@velocity(stokes)...) # update halo cells
    # elseif shear == true && DisplacementFormulation == false
    if shear == true
        BC_velo!(@velocity(stokes)..., εbg, xvi,lx,lz, phases_dev)
        flow_bcs = VelocityBoundaryConditions(;
            free_slip    = (left = true , right = true , top = true , bot = true , front = true , back = true ),
            no_slip      = (left = false, right = false, top = false, bot = false, front = false, back = false),
        )
        flow_bcs!(stokes, flow_bcs) # apply boundary conditions
        update_halo!(@velocity(stokes)...) # update halo cells
    else
        flow_bcs = VelocityBoundaryConditions(;
            free_slip    = (left = true , right = true , top = true , bot = true , front = true , back = true ),
            no_slip      = (left = false, right = false, top = false, bot = false, front = false, back = false),
        )
        flow_bcs!(stokes, flow_bcs) # apply boundary conditions
        update_halo!(@velocity(stokes)...) # update halo cells
    end
    println("Starting main loop")

    while it < 1500 #nt

        # if DisplacementFormulation == true
        #     BC_displ!(@displacement(stokes)..., εbg, xvi,lx,lz,dt)
        #     flow_bcs!(stokes, flow_bcs) # apply boundary conditions
        # end

        # if it > 1 && ustrip(dimensionalize(t,yr,CharDim)) >= (ustrip.(1.5e3yr)*interval)
        #     # add_thermal_anomaly!(pPhases, particles, interval, lx, CharDim, thermal, T_buffer, Told_buffer, Tsurf, xvi, phase_ratios, grid, pT)
        #     new_thermal_anomaly!(pPhases, particles, lx*0.5, nondimensionalize(-5km, CharDim), nondimensionalize(0.5km, CharDim))
        #     ## rhyolite
        #     circular_perturbation!(thermal.T, 30.0, nondimensionalize(1150C, CharDim), lx*0.5, nondimensionalize(-5km, CharDim), nondimensionalize(0.5km, CharDim), xvi)
        #     # circular_perturbation!(thermal.T, 30.0, nondimensionalize(1250C, CharDim), lx*0.5, nondimensionalize(-5km, CharDim), nondimensionalize(0.5km, CharDim), xvi)
        #     for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
        #         copyinn_x!(dst, src)
        #     end
        #     @views T_buffer[:,end] .= Tsurf
        #     @views thermal.T[2:end-1, :] .= T_buffer
        #     temperature2center!(thermal)
        #     # grid2particle_flip!(pT, xvi, T_buffer, Told_buffer, particles)
        #     grid2particle!(pT, xvi, T_buffer, particles)
        #     update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
        #     interval += 1.0
        # end

        args = (; ϕ=ϕ, T=thermal.Tc, P=stokes.P, dt=dt, ΔTc=thermal.ΔTc, perturbation_C = perturbation_C)
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
            ϕ, phase_ratios.center, rheology, (T=thermal.Tc, P=stokes.P)
        )

        # Advection --------------------
        # advect particles in space
        # advection!(particles, RungeKutta2(), @velocity(stokes), (grid_vx, grid_vy), dt)
        advection_LinP!(particles, RungeKutta2(), @velocity(stokes), (grid_vx, grid_vy), dt)

        # update halos
        update_cell_halo!(particles.coords..., particle_args...);
        update_cell_halo!(particles.index)
        # advection_MQS!(particles, RungeKutta2(), @velocity(stokes), (grid_vx, grid_vy), dt)
        # advect particles in memory
        move_particles!(particles, xvi, particle_args)

        # check if we need to inject particles
        inject_particles_phase!(particles, pPhases, (pT, ), (thermal.T,), xvi)

        # phase change for particles
        # phase_change!(pPhases, pϕ, 0.05, 4.0, particles)
        # phase_change!(pPhases, pEII, 1e-2, particles)
        # phase_change!(pPhases, particles)

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
        @views εII_pl_nohalo.= Array(stokes.ε_pl[2:end-1, 2:end-1, 2:end-1]) # Copy data to CPU removing the halo
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

            p = particles.coords
            # pp = [argmax(p) for p in phase_ratios.center] #if you want to plot it in a heatmap rather than scatter
            ppx, ppy = p
            pxv = ustrip.(dimensionalize(ppx.data[:], km, CharDim))
            pyv = ustrip.(dimensionalize(ppy.data[:], km, CharDim))
            clr = pPhases.data[:]
            clrT = pT.data[:]
            idxv = particles.index.data[:]

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
                    ustrip.(dimensionalize(xci_v, km, CharDim)),
                    ustrip.(dimensionalize(xvi, km, CharDim)),
                    data_v,
                    data_c,
                    velocity,
                    (round.(ustrip.(t_Kyrs); digits=3))
                )
            end
        end
    end
end

do_vtk = true
ar = 1 # aspect ratio
n = 32
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
li = (abs(last(x_global)-first(x_global)), abs(last(y_global)-first(y_global)), abs(last(z_global)-first(z_global)))
ni           = nx, ny, nz           # number of cells
di           = @. li / (nx_g(), ny_g(), nz_g())           # grid steps
grid_global  = Geometry(ni, li; origin = origin)

li_GMG, origin_GMG, phases_GMG, T_GMG = Tanzania_setup3D(grid_global.xvi,igg,nx+1,ny+1,nz+1;)

DisplacementFormulation = false             #specify if you want to use the displacement formulation

shear = true                                #specify if you want to use pure shear boundary conditions
εbg_dim   = 5e-13 / s * shear                                 #specify the background strain rate

figname = "$(today())_Tanzania_$(nx_g())x$(ny_g())x$(nz_g())"

Tanzania_3D(li_GMG, origin_GMG, phases_GMG, T_GMG, εbg_dim, igg; x_global, y_global, z_global, figname=figname, nx=nx, ny=ny, nz=nz, do_vtk=do_vtk)
