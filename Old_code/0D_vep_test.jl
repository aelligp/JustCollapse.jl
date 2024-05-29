using CUDA
# CUDA.allowscalar(false) # for safety
CUDA.allowscalar(true) # for safety
using JustRelax, JustRelax.DataIO, JustPIC
import JustRelax.@cell

## NOTE: need to run one of the lines below if one wishes to switch from one backend to another
# set_backend("Threads_Float64_2D")
# set_backend("CUDA_Float64_2D")

const USE_GPU = false;
const GPU_ID = 1;

model = if USE_GPU
    # select if CUDA or AMDGPU
    PS_Setup(:CUDA, Float64, 2)            # initialize parallel stencil in 2D
else
    PS_Setup(:Threads, Float64, 2)            # initialize parallel stencil in 2D
end
environment!(model)

using Printf, Statistics, LinearAlgebra, GeoParams, CairoMakie, SpecialFunctions, CellArrays
using ParallelStencil.FiniteDifferences2D   #specify Dimension
using ImplicitGlobalGrid
using MPI: MPI
using GeophysicalModelGenerator, StaticArrays
# using Plots 
using WriteVTK

# using MagmaThermoKinematics
# using BenchmarkTools
# using TimerOutputs
using JLD2

# -----------------------------------------------------------------------------------------
# Viscosity calculation functions stored in this script
include("./src/Particles_Helperfunc.jl");
# include("./src/Helperfunc_old.jl");
include("./src/LoadModel.jl");
println("Loaded helper functions")
#------------------------------------------------------------------------------------------
LonLat = load("./Data/ExportToba_2.jld2", "TobaTopography_LATLON");

proj = ProjectionPoint(; Lat=2.19, Lon=98.91);
Topo = Convert2CartData(LonLat, proj);

println("Done loading Model... starting Dike Injection 2D routine")
#------------------------------------------------------------------------------------------

function init_phases!(phases, particles, phases_topo)
    ni = size(phases)

    @parallel_indices (i, j) function init_phases!(
        phases, px, py, index, phases_topo)
        @inbounds for ip in JustRelax.JustRelax.cellaxes(phases)
            # quick escape
            JustRelax.@cell(index[ip, i, j]) == 0 && continue

            x = JustRelax.@cell px[ip, i, j]
            y = -(JustRelax.@cell py[ip, i, j])
            Phase = (phases_topo[i, j])
            
            # # topography
            if Phase == 1
                @cell phases[ip, i, j] = 1.0
                
            elseif Phase == 2
                @cell phases[ip, i, j] = 2.0
                
            elseif Phase == 3
                @cell phases[ip, i, j] = 3.0
            
            elseif Phase == 4
                @cell phases[ip, i, j] = 4.0
            end
        end
        return nothing
    end

    @parallel (JustRelax.@idx ni) init_phases!(
        phases, particles.coords..., particles.index, phases_topo)
end


@parallel_indices (i, j) function init_T!(temperature, depth, geotherm,tempoffset)

        temperature[i + 1, j] = tempoffset + geotherm * @all(depth)

    return nothing
end

@parallel_indices (i, j) function init_P!(pressure, ρg, depth)

        @all(pressure) = 0.0 + @all(ρg) * @all(depth)

    return nothing
end

function dirichlet_velocities_pureshear!(Vx, Vy, v_extension, xvi)
    lx = abs(reduce(-, extrema(xvi[1])))
    ly = abs(reduce(-, extrema(xvi[2])))
    xv, yv = xvi
    # v_extension /= lx / 2

    @parallel_indices (i, j) function pure_shear_x!(Vx)
        xi = xv[i]
        Vx[i, j + 1] = v_extension * (xi - lx * 0.5) / (lx/2)/2
        # Vx[i, j + 1] = v_extension * (xi - lx * 0.5) / 2
        return nothing
    end

    @parallel_indices (i, j) function pure_shear_y!(Vy)
        yi = abs(yv[j])
        Vy[i + 1, j] = v_extension * yi /ly
        return nothing
    end

    nx, ny = size(Vx)
    @parallel (1:nx, 1:(ny - 2)) pure_shear_x!(Vx)
    nx, ny = size(Vy)
    @parallel (1:(nx - 2), 1:ny) pure_shear_y!(Vy)

    return nothing
end

@parallel_indices (i, j) function compute_melt_fraction!(ϕ, rheology, args)
ϕ[i, j] = compute_meltfraction(rheology, ntuple_idx(args, i, j))
return nothing
end

#Multiplies parameter with the fraction of a phase
@generated function compute_param_times_frac(
    fn::F,
    PhaseRatios::Union{NTuple{N,T},SVector{N,T}},
    MatParam::NTuple{N,AbstractMaterialParamsStruct},
    argsi,
    ) where {F,N,T}
    # # Unrolled dot product
    quote
        val = zero($T)
        Base.Cartesian.@nexprs $N i ->
        val += @inbounds PhaseRatios[i] * fn(MatParam[i], argsi)
        return val
    end
end

function compute_meltfraction_ratio(args::Vararg{Any,N}) where {N}
    return compute_param_times_frac(compute_meltfraction, args...)
end

@parallel_indices (I...) function compute_melt_fraction!(ϕ, phase_ratios, rheology, args)
args_ijk = ntuple_idx(args, I...)
ϕ[I...] = compute_melt_frac(rheology, args_ijk, phase_ratios[I...])
return nothing
end

@inline function compute_melt_frac(rheology, args, phase_ratios)
    return compute_meltfraction_ratio(phase_ratios, rheology, args)
end

function copy_arrays_GPU2CPU!(
    T_CPU::AbstractArray, ϕ_CPU::AbstractArray, T_GPU::AbstractArray, ϕ_GPU::AbstractArray
    )
    T_CPU .= Array(T_GPU)
    ϕ_CPU .= Array(ϕ_GPU)
    
    return nothing
end


solution(ε, t, G, η) = 2 * ε * η * (1 - exp(-G * t / η))
solution_η_vep(τII, εII) = 0.5 .* τII ./ εII

function pureshear!(stokes, εbg, xvi)
    stokes.V.Vx .= PTArray([ x*εbg for x in xvi[1], _ in 1:ny+2])
    stokes.V.Vy .= PTArray([-y*εbg for _ in 1:nx+2, y in xvi[2]])
    return nothing
end

# function DikeInjection_2D(igg; figname=figname, nx=nx, ny=ny, )

    # IO ------------------------------------------------
    figdir = "./fig2D/$figname/"
    !isdir(figdir) && mkpath(figdir)

    # Standard MTK Toba routine--------------------------------------------------------------

    # nondimensionalize with CharDim 
    CharDim = GEO_units(; length=40km, viscosity=1e20Pa * s)
    ni = (nx, ny)   #grid spacing for calculation (without ghost cells)

    #-------rheology parameters--------------------------------------------------------------
    # plasticity setup
    do_DP = false               # do_DP=false: Von Mises, do_DP=true: Drucker-Prager (friction angle)
    η_reg = 1.0e18Pas           # regularisation "viscosity" for Drucker-Prager
    τ_y = 15MPa              # yield stress. If do_DP=true, τ_y stand for the cohesion: c*cos(ϕ)
    ϕ = 15.0 * do_DP         # friction angle
    G0 = 1e10Pa        # elastic shear modulus
    Coh = τ_y         # cohesion
    εbg = 1e-15 / s             # background strain rate
    εbg = nondimensionalize(εbg, CharDim) # background strain rate
    pl = DruckerPrager_regularised(; C=Coh, ϕ=ϕ, η_vp=η_reg, Ψ=0)        # plasticity
    el = SetConstantElasticity(; G=G0, ν=0.5)                            # elastic spring
    disl_upper_crust = DislocationCreep(;
        A=5.07e-18, n=2.3, E=154e3, V=6e-6, r=0.0, R=8.3145
    )
    η_uppercrust = 1e21             #viscosity of the upper crust
    creep_rock = LinearViscous(; η=η_uppercrust * Pa * s)
    cutoff_visc = (
        nondimensionalize(1e14Pa * s, CharDim), nondimensionalize(1e24Pa * s, CharDim)
    )
    β_rock = inv(get_Kb(el))
    Kb = get_Kb(el)

    #-------JustRelax parameters-------------------------------------------------------------
    # Domain setup for JustRelax 
    lx, lz                  = nondimensionalize(1km, CharDim), nondimensionalize(1km, CharDim) # nondim if CharDim=CharDim
    li                      = lx, lz
    b_width                 = (4, 4, 0) #boundary width
    origin                  = 0.0, -lz
    igg                     = igg
    di                      = @. li / (nx_g(), ny_g()) # grid step in x- and y-direction
    xci, xvi                = lazy_grid(di, li, ni; origin=origin) #non-dim nodes at the center and the vertices of the cell (staggered grid)
    #---------------------------------------------------------------------------------------

    # Set material parameters                                       
    MatParam = (
        #Name="UpperCrust"
        SetMaterialParams(; 
            Phase   = 1, 
            Density  = PT_Density(ρ0=2700kg/m^3, β=β_rock/Pa),
            HeatCapacity = ConstantHeatCapacity(cp=1050J/kg/K),
            Conductivity = ConstantConductivity(k=3.0Watt/K/m),       
            LatentHeat = ConstantLatentHeat(Q_L=350e3J/kg),
            CompositeRheology = CompositeRheology((creep_rock,)),
            #   CompositeRheology = CompositeRheology((el, creep_rock, pl, )),
            Melting = MeltingParam_Caricchi(),
            #  Elasticity = ConstantElasticity(; G=Inf*Pa, Kb=Inf*Pa),
            # Elasticity = el,
            CharDim  = CharDim,),)
    #----------------------------------------------------------------------------------

    #update depth with topography !!!CPU ONLY!!!
    phases_topo_v = ones(nx + 1, ny + 1)
    phases_topo_c = ones(nx, ny)

    phases_topo_v = (PTArray(phases_topo_v))
    phases_topo_c = (PTArray(phases_topo_c))
    @parallel vertex2center!(phases_topo_c, phases_topo_v)
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 20, 40, 5 # nxcell = initial number of particles per cell; max_cell = maximum particles accepted in a cell; min_xcell = minimum number of particles in a cell
    particles = init_particles_cellarrays(
        nxcell, max_xcell, min_xcell, xvi..., di..., ni...
    )
    # velocity grids
    grid_vx, grid_vy = velocity_grids(xci, xvi, di)
    # temperature
    pT, pPhases = init_particle_fields_cellarrays(particles, Val(3))
    particle_args = (pT, pPhases)
    init_phases!(pPhases, particles, phases_topo_v)

    phase_ratios = PhaseRatio(ni, length(MatParam))
    @parallel (@idx ni) phase_ratios_center(phase_ratios.center, pPhases)

    #not yet nice
    p = particles.coords
    pp = [argmax(p) for p in phase_ratios.center]; #if you want to plot it in a heatmap rather than scatter

    # Physical Parameters 

    geotherm = GeoUnit(30K / km) 
    geotherm_nd = ustrip(Value(nondimensionalize(geotherm, CharDim)))
    tempoffset = nondimensionalize(0C, CharDim)
    ρ0 = MatParam[1].Density[1].ρ0.val                   # reference Density
    Cp0 = MatParam[1].HeatCapacity[1].cp.val              # heat capacity     
    κ = nondimensionalize(1.5Watt / K / m, CharDim) / (ρ0 * Cp0)                                   # thermal diffusivity
    dt = dt_diff = 0.5 * min(di...)^2 / κ / 2.01           # diffusive CFL timestep limiter

    v_extension = nondimensionalize(0.50cm / yr, CharDim)   # extension velocity for pure shear boundary conditions
    
    # Initialisation 
    thermal = ThermalArrays(ni)                                # initialise thermal arrays and boundary conditions
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux=(left=true, right=true, top=false, bot=false),
        periodicity=(left=false, right=false, top=false, bot=false),
    )

    xyv = PTArray([y for x in xvi[1], y in xvi[2]])

    @parallel (@idx ni .+ 1) init_T!(thermal.T, abs.(xyv), geotherm_nd,tempoffset
    )
  
    thermal_bcs!(thermal.T, thermal_bc)

    @parallel (JustRelax.@idx size(thermal.Tc)...) temperature2center!(
        thermal.Tc, thermal.T
    )

    stokes = StokesArrays(ni, ViscoElastic)                         # initialise stokes arrays with the defined regime
    pt_stokes = PTStokesCoeffs(li, di; ϵ=1e-5, CFL=0.8 / √2.1) #ϵ=1e-4,  CFL=1 / √2.1 CFL=0.27 / √2.1

    args = (; T=thermal.Tc, P=stokes.P, dt=dt)
    pt_thermal = PTThermalCoeffs(
        MatParam, phase_ratios, args, dt, ni, di, li; ϵ=1e-5, CFL=1e-3 / √2.1
    )
    # Boundary conditions of the flow

    # dirichlet_velocities_pureshear!(@velocity(stokes)..., v_extension, xvi)
    pureshear!(stokes, εbg, xvi)

    flow_bcs = FlowBoundaryConditions(;
        free_slip=(left=true, right=true, top=true, bot=true)
    )
    flow_bcs!(stokes, flow_bcs)

    η = @ones(ni...)                                     # initialise viscosity       
    η_vep = deepcopy(η)                                       # initialise viscosity for the VEP
    ϕ = similar(η)                                        # melt fraction center

    # Buoyancy force
    ρg = @zeros(ni...), @zeros(ni...)                      # ρg[1] is the buoyancy force in the x direction, ρg[2] is the buoyancy force in the y direction                                   

    @copy thermal.Told thermal.T

    args = (; ϕ=ϕ, T=thermal.Tc, P=stokes.P, dt=dt)#, S=S, mfac=mfac, η_f=η_f, η_s=η_s) # this is used for various functions, remember to update it within loops.

    for _ in 1:2
        @parallel (JustRelax.@idx ni) compute_ρg!(
            ρg[2], phase_ratios.center, MatParam, (T=thermal.Tc, P=stokes.P)
        )
        @parallel (@idx ni) init_P!(stokes.P, ρg[2],abs.(xyv))
    end

    @parallel (@idx ni) compute_viscosity!(
        η, 1.0, phase_ratios.center, @strain(stokes)..., args, MatParam, cutoff_visc
    )
    η_vep = copy(η)

    # make sure they are the same 
    thermal.Told .= thermal.T
    @parallel (JustRelax.@idx size(thermal.Tc)...) temperature2center!(
        thermal.Tc, thermal.T
    )    
    # ----------------------------------------------------  
    @parallel (@idx ni) compute_melt_fraction!(
        ϕ, phase_ratios.center, MatParam, (T=thermal.Tc, P=stokes.P)
    )

    # Time loop
    t, it = 0.0, 0
    interval = 1.0
    InjectVol = 0.0
    evo_t = Array{Float64}[]
    evo_InjVol = Array{Float64}[]
    evo_τxx = Array{Float64}[]
    evo_η_vep = Array{Float64}[]
    sol = Array{Float64}[]
    sol_vep = Array{Float64}[]
    local iters

    T_buffer = @zeros(ni .+ 1)
    Told_buffer = similar(T_buffer)
    for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
        copyinn_x!(dst, src)
    end
   
    JustPIC.grid2particle!(pT, xvi, T_buffer, particles.coords)

    args = (; ϕ=ϕ, T=thermal.Tc, P=stokes.P, dt=dt)


    while it < 2 #nt
        println("Check 2, before particlegrid2: ", extrema((pT.data[:])[particles.index.data[:]]))

        args = (; ϕ=ϕ, T=thermal.Tc, P=stokes.P, dt=dt)
        @views T_buffer[:, end] .= nondimensionalize(0.0C, CharDim)
        @views T_buffer[args.sticky_air.==4.0] .= nondimensionalize(0.0C, CharDim)
        # @views T_buffer[:, 1] .= maximum(thermal.T)
        @views thermal.T[2:end-1, :] .= T_buffer
        temperature2center!(thermal)
        args = (; ϕ=ϕ, T=thermal.Tc, P=stokes.P, dt=dt, pressure_top=pressure_top,sticky_air=mask_sticky_air)
        @parallel (@idx ni) compute_viscosity!(
            η, 1.0, phase_ratios.center, @strain(stokes)..., args, MatParam, cutoff_visc
        )

        @parallel (@idx ni) compute_ρg!(
            ρg[2], phase_ratios.center, MatParam, (T=thermal.Tc, P=stokes.P)
        )
        @copy stokes.P0 stokes.P
        args = (; ϕ=ϕ, T=thermal.Tc, P=stokes.P, dt=dt)
        # Stokes solver -----------------
        # dt    = t < 10 * kyr ? 0.05 * kyr : 1.0 * kyr
        # dt = t < nondimensionalize(10000yr, CharDim) ? nondimensionalize(50yr, CharDim) : nondimensionalize(1000yr, CharDim)

        solve!(
            stokes,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            η,
            η_vep,
            phase_ratios,
            MatParam,
            args,
            dt,
            igg;
            iterMax=100e3,
            nout=1e3,
            b_width,
            viscosity_cutoff=cutoff_visc,
        )
        @parallel (JustRelax.@idx ni) tensor_invariant!(stokes.ε.II, @strain(stokes)...)

        @parallel (@idx ni .+ 1) multi_copy!(@tensor(stokes.τ_o), @tensor(stokes.τ))
        @parallel (@idx ni) multi_copy!(
            @tensor_center(stokes.τ_o), @tensor_center(stokes.τ)
        )
        dt = compute_dt(stokes, di, dt_diff, igg)
        # ------------------------------
        @show dt
        @show extrema(stokes.V.Vy)
        @show extrema(stokes.V.Vx)
        particle2grid!(T_buffer, pT, xvi, particles.coords)
        @views T_buffer[:, end] .= nondimensionalize(0.0C, CharDim)
        @views thermal.T[2:end-1, :] .= T_buffer
        temperature2center!(thermal)


        # Thermal solver ---------------
        println("Check 4, before thermal solver: ", extrema(thermal.T))
        heatdiffusion_PT!(
            thermal,
            pt_thermal,
            thermal_bc,
            MatParam,
            args,
            dt,
            di;
            igg=igg,
            phase=phase_ratios,
            iterMax=50e3,
            nout=1e2,
            verbose=true,
        )
        for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
            copyinn_x!(dst, src)
        end
        # copyinn_x!(Told_buffer, thermal.T)
        println("Check 5, after thermal solver: ", extrema(thermal.T))
        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection_RK!(particles, @velocity(stokes), grid_vx, grid_vy, dt, 2 / 3)
        # advect particles in memory
        shuffle_particles!(particles, xvi, particle_args)   
        JustPIC.clean_particles!(particles, xvi, particle_args)
        grid2particle_flip!(pT, xvi, T_buffer, Told_buffer, particles.coords)
        println("Check 1, after grid2particle_flip: ", extrema((pT.data[:])[particles.index.data[:]]))
        # check if we need to inject particles
        inject = check_injection(particles)
        inject && inject_particles_phase!(particles, pPhases, (pT, ), (T_buffer,), xvi)

        @parallel (@idx ni) phase_ratios_center(phase_ratios.center, pPhases)
        
        @parallel (@idx ni) compute_melt_fraction!(
            ϕ, phase_ratios.center, MatParam, (T=thermal.Tc, P=stokes.P)
        )

        @show it += 1
        t += dt
        if it == 1
            # @show Ra
        end

        push!(evo_t, ustrip(dimensionalize(t, s, CharDim)))
        push!(evo_τxx, ustrip(dimensionalize(maximum(stokes.τ.II), Pa, CharDim)))
        push!(sol, solution(ustrip(dimensionalize(εbg,s^-1.0,CharDim)), ustrip(dimensionalize(t,s,CharDim)), ustrip(G0), ustrip(1e21Pa*s)))
        push!(evo_η_vep, (ustrip(dimensionalize(maximum(η_vep), Pa*s, CharDim))))
        push!(sol_vep, maximum(solution_η_vep(ustrip(dimensionalize(maximum(stokes.τ.II), Pa, CharDim)), (ustrip(dimensionalize(maximum(stokes.ε.II), s^-1.0, CharDim))))))

        fig2 = Figure(resolution = (2000, 2000), createmissing = true, fontsize = 40.0)
        ax10 = Axis(fig2[1,1], xlabel="Time [Yrs]", ylabel="Stress [Pa]", title="0D_Plot")
        ax20 = Axis(fig2[2,1], xlabel="Time [Yrs]", ylabel="Viscosity", title="0D_Plot")
        lines!(ax10, (evo_t)./(365.25*24*3600), evo_τxx,color=:black)
        lines!(ax10, (evo_t)./(365.25*24*3600), sol, color=:red)
        lines!(ax20, (evo_t)./(365.25*24*3600), log10.(evo_η_vep), color=:black)
        lines!(ax20, (evo_t)./(365.25*24*3600), log10.(sol_vep), color=:red)
        save(joinpath(figdir, "0D_plot.png"), fig2)
        display(fig2)

             
    end
    # finalize_global_grid(; finalize_MPI=true)
    # finalize_global_grid()

    # print_timer() 
    return thermal, stokes
end

# figdir = "figs2D"
# ni, xci, li, di, figdir ,thermal = DikeInjection_2D();
function run()
    figname = "0D_test_pureshear_LV"
    ar = 1 # aspect ratio
    n = 32
    nx = n * ar - 2
    ny = n - 2
    nz = n - 2
   global igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
        IGG(init_global_grid(nx, ny, 0; init_MPI=true)...)
        # IGG(init_global_grid(nx, ny, nz; init_MPI= true)...)
    else
        igg
    end
    DikeInjection_2D(igg; figname=figname, nx=nx, ny=ny)

    return figname, nx, ny, igg
end

# @time 
run()
