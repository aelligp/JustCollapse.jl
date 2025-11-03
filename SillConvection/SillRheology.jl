

function init_rheologies(oxd_wt_sill, oxd_wt_host_rock; scaling = 1e0, magma = true, CharDim = nothing)
    # Define parameters
    sill = magma ? ViscosityPartialMelt_Costa_etal_2009(η = GiordanoMeltViscosity(oxd_wt = oxd_wt_sill, η0 = scaling)) : LinearViscous(η = 1.0e8Pa*s)
    host_rock = magma ? ViscosityPartialMelt_Costa_etal_2009(η = GiordanoMeltViscosity(oxd_wt = oxd_wt_host_rock, η0 = scaling)) : LinearViscous(η = 1.0e13Pa*s)


    # Define rheolgy struct
    rheology = (
        # Name              = "host_rock",
        SetMaterialParams(;
            Phase             = 1,
            Density           = MeltDependent_Density(ρsolid=T_Density(; ρ0 = 2700kg/m^3,α=3e-5/K), ρmelt=Melt_DensityX(oxd_wt = oxd_wt_host_rock)),
            HeatCapacity      = Latent_HeatCapacity(Cp=T_HeatCapacity_Whittington(), Q_L=350e3J/kg),
            Conductivity      = ConstantConductivity(; k = 3.0),
            CompositeRheology = CompositeRheology((host_rock,)),
            Melting           = SmoothMelting(p=MeltingParam_Quadratic(T_s=(625+273)K,T_l=(875+273)K), k_liq=0.21/K),
            Gravity           = ConstantGravity(; g = 9.81),
            CharDim           = CharDim,
        ),
        # Name              = "Sill",
        SetMaterialParams(;
            Phase             = 2,
            Density           = MeltDependent_Density(ρsolid=T_Density(; ρ0 = 2700kg/m^3,α=3e-5/K), ρmelt=Melt_DensityX(oxd_wt = oxd_wt_sill)),
            HeatCapacity      = Latent_HeatCapacity(Cp=T_HeatCapacity_Whittington(), Q_L=350e3J/kg),
            Conductivity      = ConstantConductivity(),
            CompositeRheology = CompositeRheology((sill,)),
            # Melting           = MeltingParam_Smooth3rdOrder(a=3043.0,b=-10552.0,c=12204.9,d=-4709.0),
            Melting           = SmoothMelting(p=MeltingParam_Quadratic(T_s=(675+273)K,T_l=(1125+273)K), k_liq=0.21/K),
            CharDim           = CharDim,
        ),
    )
end

function init_phases!(phases, phase_grid, particles, xvi)
    ni = size(phases)
    return @parallel (@idx ni) _init_phases!(phases, phase_grid, particles.coords, particles.index, xvi)
end

@parallel_indices (I...) function _init_phases!(phases, phase_grid, pcoords::NTuple{N, T}, index, xvi) where {N, T}

    ni = size(phases)

    for ip in cellaxes(phases)
        # quick escape
        @index(index[ip, I...]) == 0 && continue

        pᵢ = ntuple(Val(N)) do i
            @index pcoords[i][ip, I...]
        end

        d = Inf # distance to the nearest particle
        particle_phase = -1
        for offi in 0:1, offj in 0:1
            ii = I[1] + offi
            jj = I[2] + offj

            !(ii ≤ ni[1]) && continue
            !(jj ≤ ni[2]) && continue

            xvᵢ = (
                xvi[1][ii],
                xvi[2][jj],
            )
            d_ijk = √(sum((pᵢ[i] - xvᵢ[i])^2 for i in 1:N))
            if d_ijk < d
                d = d_ijk
                particle_phase = phase_grid[ii, jj]
            end
        end
        @index phases[ip, I...] = Float64(particle_phase)
    end

    return nothing
end
