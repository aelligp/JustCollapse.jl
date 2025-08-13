using GeoParams.Dislocation
using Random

function init_rheologies(layers, oxd_wt, fric_angle; linear = false, incompressible = true, plastic = true, magma = false, softening_C=true, softening_ϕ=false)

    η_reg = 1.0e15
    C = plastic ? 10.0e6 : Inf
    ϕ = fric_angle
    Ψ = 0.0
    # soft_C = NonLinearSoftening(; ξ₀ = C, Δ = C / 2, σ = 0.001)       # nonlinear softening law
    # soft_ϕ = NonLinearSoftening(; ξ₀ = ϕ, Δ = ϕ / 2, σ = 0.001)       # nonlinear softening law
    soft_C = softening_C ? LinearSoftening(C/2, C, 0.0, 0.5) : NoSoftening()       # nonlinear softening law
    soft_ϕ = softening_ϕ ? LinearSoftening(fric_angle/2, fric_angle, 0.0, 0.5) : NoSoftening()       # nonlinear softening law
    pl = DruckerPrager_regularised(; C = C, ϕ = ϕ, η_vp = (η_reg), Ψ = Ψ, softening_C = soft_C, softening_ϕ = soft_ϕ)
    pl_bot = DruckerPrager_regularised(; C = C, ϕ = ϕ, η_vp = (η_reg), Ψ = Ψ)

    pl_cone_1 = DruckerPrager_regularised(; C = ((C / 2) * 0.3), ϕ = ϕ , η_vp = (η_reg), Ψ = Ψ, softening_C = soft_C, softening_ϕ = soft_ϕ)
    pl_cone_2 = DruckerPrager_regularised(; C = ((C / 2) * 0.4), ϕ = ϕ , η_vp = (η_reg), Ψ = Ψ, softening_C = soft_C, softening_ϕ = soft_ϕ)
    pl_cone_3 = DruckerPrager_regularised(; C = ((C / 2) * 0.5), ϕ = ϕ , η_vp = (η_reg), Ψ = Ψ, softening_C = soft_C, softening_ϕ = soft_ϕ)
    pl_cone_4 = DruckerPrager_regularised(; C = ((C / 2) * 0.6), ϕ = ϕ , η_vp = (η_reg), Ψ = Ψ, softening_C = soft_C, softening_ϕ = soft_ϕ)
    pl_cone_5 = DruckerPrager_regularised(; C = ((C / 2) * 0.7), ϕ = ϕ , η_vp = (η_reg), Ψ = Ψ, softening_C = soft_C, softening_ϕ = soft_ϕ)
    G0 = 25.0e9Pa        # elastic shear modulus
    G_magma = 6.0e9Pa        # elastic shear modulus magma

    el = incompressible ? ConstantElasticity(; G = G0, ν = 0.45) : ConstantElasticity(; G = G0, ν = 0.25)
    el_magma = incompressible ? ConstantElasticity(; G = G_magma, ν = 0.45) : ConstantElasticity(; G = G_magma, ν = 0.25)
    β = 1 / el.Kb.val
    β_magma = 1 / el_magma.Kb.val
    Cp = 1050.0

    # magma_visc = magma ? ViscosityPartialMelt_Costa_etal_2009(η=LinearViscous(η=1e15)) : LinearViscous(η=1e15)
    # magma_visc = magma ? GiordanoMeltViscosity(oxd_wt = oxd_wt, η0 = 1.0e12Pas) : LinearViscous(η = 1.0e15)
    magma_visc = magma ? ViscosityPartialMelt_Costa_etal_2009(η = GiordanoMeltViscosity(oxd_wt = oxd_wt, η0 = 1.0Pas)) : LinearViscous(η = 1.0e15)
    # conduit_visc = magma  ? ViscosityPartialMelt_Costa_etal_2009(η=LinearViscous(η=1e15)) : LinearViscous(η=1e15)

    #dislocation laws
    # disl_top  = linear ? LinearViscous(η=1e23) : DislocationCreep(; A=1.67e-24, n=3.5, E=1.87e5, V=6e-6, r=0.0, R=8.3145)
    # disl_top  = linear ? LinearViscous(η=1e23) : SetDislocationCreep(Dislocation.dry_olivine_Karato_2003)
    disl_top = linear ? LinearViscous(η = 1.0e23) : SetDislocationCreep(Dislocation.granite_Carter_1987)

    disl_bot = linear ? LinearViscous(η = 1.0e21) : SetDislocationCreep(Dislocation.granite_Carter_1987)
    # disl_bot = linear ? LinearViscous(η = 1.0e21) : SetDislocationCreep(Dislocation.strong_diabase_Mackwell_1998)
    # disl_bot = linear ? LinearViscous(η = 1.0e21) : SetDislocationCreep(Dislocation.wet_quartzite_Hirth_2001)


    # Define the Volcano cone rheology
    layer_rheology(::Val{1}) =
        SetMaterialParams(;
        Phase = 5,
        Density = PT_Density(; ρ0 = 2.7e3, T0 = 273.15, β = β),
        # Density = Melt_DensityX(oxd_wt = oxd_wt; β=β_magma),
        HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
        Conductivity = ConstantConductivity(; k = 3.0),
        CompositeRheology = CompositeRheology((magma_visc, disl_top, el, pl_cone_1)),
        # CompositeRheology = CompositeRheology( (LinearViscous(; η=1e23), el, pl)),
        Melting = MeltingParam_Smooth3rdOrder(a = 3043.0, b = -10552.0, c = 12204.9, d = -4709.0), #felsic melting curve
        Gravity = ConstantGravity(; g = 9.81),
    )

    layer_rheology(::Val{2}) =
        SetMaterialParams(;
        Phase = 6,
        Density = PT_Density(; ρ0 = 2.7e3, T0 = 273.15, β = β),
        # Density = Melt_DensityX(oxd_wt = oxd_wt; β=β_magma),
        HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
        Conductivity = ConstantConductivity(; k = 3.0),
        CompositeRheology = CompositeRheology((magma_visc, disl_top, el, pl_cone_2)),
        # CompositeRheology = CompositeRheology( (LinearViscous(; η=1e23), el, pl_cone)),
        Melting = MeltingParam_Smooth3rdOrder(a = 3043.0, b = -10552.0, c = 12204.9, d = -4709.0), #felsic melting curve
        Gravity = ConstantGravity(; g = 9.81),
    )

    layer_rheology(::Val{3}) =
        SetMaterialParams(;
        Phase = 7,
        Density = PT_Density(; ρ0 = 2.7e3, T0 = 273.15, β = β),
        # Density = Melt_DensityX(oxd_wt = oxd_wt; β=β_magma),
        HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
        Conductivity = ConstantConductivity(; k = 3.0),
        CompositeRheology = CompositeRheology((magma_visc, disl_top, el, pl_cone_3)),
        # CompositeRheology = CompositeRheology( (LinearViscous(; η=1e23), el, pl_cone)),
        Melting = MeltingParam_Smooth3rdOrder(a = 3043.0, b = -10552.0, c = 12204.9, d = -4709.0), #felsic melting curve
        Gravity = ConstantGravity(; g = 9.81),
    )

    layer_rheology(::Val{4}) =
        SetMaterialParams(;
        Phase = 8,
        Density = PT_Density(; ρ0 = 2.7e3, T0 = 273.15, β = β),
        # Density = Melt_DensityX(oxd_wt = oxd_wt; β=β_magma),
        HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
        Conductivity = ConstantConductivity(; k = 3.0),
        CompositeRheology = CompositeRheology((magma_visc, disl_top, el, pl_cone_4)),
        # CompositeRheology = CompositeRheology( (LinearViscous(; η=1e23), el, pl_cone)),
        Melting = MeltingParam_Smooth3rdOrder(a = 3043.0, b = -10552.0, c = 12204.9, d = -4709.0), #felsic melting curve
        Gravity = ConstantGravity(; g = 9.81),
    )

    layer_rheology(::Val{5}) =
        SetMaterialParams(;
        Phase = 9,
        Density = PT_Density(; ρ0 = 2.7e3, T0 = 273.15, β = β),
        # Density = Melt_DensityX(oxd_wt = oxd_wt; β=β_magma),
        HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
        Conductivity = ConstantConductivity(; k = 3.0),
        CompositeRheology = CompositeRheology((magma_visc, disl_top, el, pl_cone_5)),
        # CompositeRheology = CompositeRheology( (LinearViscous(; η=1e23), el, pl_cone)),
        Melting = MeltingParam_Smooth3rdOrder(a = 3043.0, b = -10552.0, c = 12204.9, d = -4709.0), #felsic melting curve
        Gravity = ConstantGravity(; g = 9.81),
    )


    rheology_volcano = ntuple(i -> layer_rheology(Val(i)), Val(layers))


    # Define rheolgy struct
    return rheology = (
        # Name = "Upper crust",
        SetMaterialParams(;
            Phase = 1,
            Density = PT_Density(; ρ0 = 2.7e3, T0 = 273.15, β = β),
            # Density = Melt_DensityX(oxd_wt = oxd_wt; β=β_magma),
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
            Conductivity = ConstantConductivity(; k = 3.0),
            CompositeRheology = CompositeRheology((magma_visc, disl_top, el, pl)),
            # CompositeRheology = CompositeRheology( (LinearViscous(; η=1e23), el, pl)),
            # Melting = MeltingParam_Smooth3rdOrder(a = 3043.0, b = -10552.0, c = 12204.9, d = -4709.0), #felsic melting curve
            Melting = MeltingParam_Assimilation(T_s = (725+273)K, T_l= (1140+273)K),
            Gravity = ConstantGravity(; g = 9.81),
        ),
        # Name = "Lower crust",
        SetMaterialParams(;
            Phase = 2,
            Density = PT_Density(; ρ0 = 2.7e3, T0 = 273.15, β = β),
            # Density = Melt_DensityX(oxd_wt = oxd_wt; β=β_magma),
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
            Conductivity = ConstantConductivity(; k = 3.0),
            CompositeRheology = CompositeRheology((magma_visc, disl_bot, el, pl_bot)),
            # CompositeRheology = CompositeRheology( (LinearViscous(; η=1e21), el, pl)),
            # Melting = MeltingParam_Smooth3rdOrder(a = 3043.0, b = -10552.0, c = 12204.9, d = -4709.0), #felsic melting curve
            Melting = MeltingParam_Assimilation(T_s = (725+273)K, T_l= (1140+273)K),
            Gravity = ConstantGravity(; g = 9.81),
        ),

        # Name              = "magma chamber",
        SetMaterialParams(;
            Phase = 3,
            # Density           = MeltDependent_Density(ρsolid=PT_Density(ρ0=2.65e3, T0=273.15, β=β_magma), ρmelt=T_Density(ρ0=2.4e3, T0=273.15)),
            # Density           = ConstantDensity(ρ=2.7e3),
            Density = BubbleFlow_Density(ρgas = ConstantDensity(ρ = 10.0), ρmelt = Melt_DensityX(oxd_wt = oxd_wt), c0 = 3.0e-2),
            # Density           = PT_Density(; ρ0=2.4e3, T0=273.15, β=β_magma),
            Conductivity = ConstantConductivity(; k = 3.0),
            # HeatCapacity      = Latent_HeatCapacity(Cp=ConstantHeatCapacity()),
            HeatCapacity = Latent_HeatCapacity(Cp = ConstantHeatCapacity(), Q_L = 350.0e3J / kg),
            LatentHeat = ConstantLatentHeat(Q_L = 350.0e3J / kg),
            CompositeRheology = CompositeRheology((magma_visc, el_magma)),
            Melting = MeltingParam_Smooth3rdOrder(a = 3043.0, b = -10552.0, c = 12204.9, d = -4709.0), #felsic melting curve
        ),
        # Name              = "magma chamber - hot anomaly",
        SetMaterialParams(;
            Phase = 4,
            # Density           = T_Density(; ρ0=2.2e3, T0=273.15),
            Density = BubbleFlow_Density(ρgas = ConstantDensity(ρ = 10.0), ρmelt = Melt_DensityX(oxd_wt = oxd_wt), c0 = 3.0e-2),
            # Density           = BubbleFlow_Density(ρgas=ConstantDensity(ρ=10.0), ρmelt=MeltDependent_Density(ρsolid=T_Density(ρ0=2.65e3, T0=273.15), ρmelt= ConstantDensity(ρ=2.4e3)), c0=4e-2),
            # Density           = ConstantDensity(ρ=2.7e3),
            # Density           = Melt_DensityX(oxd_wt= oxd_wt),
            Conductivity = ConstantConductivity(; k = 3.0),
            # HeatCapacity      = Latent_HeatCapacity(Cp=ConstantHeatCapacity()),
            HeatCapacity = Latent_HeatCapacity(Cp = ConstantHeatCapacity(), Q_L = 350.0e3J / kg),
            LatentHeat = ConstantLatentHeat(Q_L = 350.0e3J / kg),
            CompositeRheology = CompositeRheology((magma_visc, el_magma)),
            Melting = MeltingParam_Smooth3rdOrder(a = 3043.0, b = -10552.0, c = 12204.9, d = -4709.0), #felsic melting curve
        ),

        rheology_volcano...,

        # Name              = "Layers",
        SetMaterialParams(;
        Phase = Int64(layers + 5),
            Density = PT_Density(; ρ0 = 2.7e3, T0 = 273.15, β = β),
            # Density = Melt_DensityX(oxd_wt = oxd_wt; β=β_magma),
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
            Conductivity = ConstantConductivity(; k = 3.0),
            CompositeRheology = CompositeRheology((magma_visc, disl_top, el, pl)),
            # CompositeRheology = CompositeRheology( (LinearViscous(; η=1e23), el, pl)),
            # Melting = MeltingParam_Smooth3rdOrder(a = 3043.0, b = -10552.0, c = 12204.9, d = -4709.0), #felsic melting curve
            Melting = MeltingParam_Assimilation(T_s = (725+273)K, T_l= (1140+273)K),
            Gravity = ConstantGravity(; g = 9.81),
        ),

        # # Name              = "Conduit",
        # SetMaterialParams(;
        #     Phase = Int64(layers + 6),
        #     Density = PT_Density(; ρ0 = 2.7e3, T0 = 273.15, β = β),
        #     # Density = Melt_DensityX(oxd_wt = oxd_wt; β=β_magma),
        #     HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
        #     Conductivity = ConstantConductivity(; k = 3.0),
        #     CompositeRheology = CompositeRheology((magma_visc, disl_top, el, pl)),
        #     # CompositeRheology = CompositeRheology( (LinearViscous(; η=1e23), el, pl)),
        #     Melting = MeltingParam_Smooth3rdOrder(a = 3043.0, b = -10552.0, c = 12204.9, d = -4709.0), #felsic melting curve
        #     Gravity = ConstantGravity(; g = 9.81),
        # ),
        # Name              = "StickyAir",
        SetMaterialParams(;
            Phase = Int64(layers + 6),
            Density = ConstantDensity(; ρ = 1.0e0),
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
            Conductivity = ConstantConductivity(; k = 3.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e22), el, pl)),
            Gravity = ConstantGravity(; g = 9.81),
            # Melting           = MeltingParam_Smooth3rdOrder(a=3043.0,b=-10552.0,c=12204.9,d=-4709.0), #felsic melting curve
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


## old conduit rheology:
# # Name              = "Conduit",
# SetMaterialParams(;
#     Phase             = Int64(layers+5),
#     # Density           = BubbleFlow_Density(ρgas=ConstantDensity(ρ=10.0), ρmelt=MeltDependent_Density(ρsolid=T_Density(ρ0=2.65e3, T0=273.15), ρmelt=T_Density(ρ0=2.4e3, T0=273.15)), c0=4e-2),
#     # # Density           = BubbleFlow_Density(ρgas=ConstantDensity(ρ=10.0), ρmelt=ConstantDensity(ρ=2.4e3), c0=4e-2),
#     # # Density           = T_Density(; ρ0=1.5e3, T0=273.15),
#     # Conductivity      = ConstantConductivity(; k  = 3.0),
#     # # HeatCapacity      = Latent_HeatCapacity(Cp=ConstantHeatCapacity()),
#     # HeatCapacity      = Latent_HeatCapacity(Cp=ConstantHeatCapacity(), Q_L=350e3J/kg),
#     # LatentHeat        = ConstantLatentHeat(Q_L=350e3J/kg),
#     # CompositeRheology = CompositeRheology((conduit_visc, el_magma,)),
#     # Melting           = MeltingParam_Smooth3rdOrder(a=3043.0,b=-10552.0,c=12204.9,d=-4709.0), #felsic melting curve
#     Density           = PT_Density(; ρ0=2.7e3, T0=273.15, β=β),
#     HeatCapacity      = ConstantHeatCapacity(; Cp = Cp),
#     Conductivity      = ConstantConductivity(; k  = 3.0),
#     CompositeRheology = CompositeRheology( (disl_top, el, pl)),
#     # CompositeRheology = CompositeRheology( (LinearViscous(; η=1e23), el, pl)),
# Melting = MeltingParam_Smooth3rdOrder(a = 3043.0, b = -10552.0, c = 12204.9, d = -4709.0), #felsic melting curve
#     ),
