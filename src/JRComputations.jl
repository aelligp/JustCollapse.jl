function stress(stokes::StokesArrays{ViscoElastic,A,B,C,D,nDim}) where {A,B,C,D,nDim}
    return stress(stokes.τ), stress(stokes.τ_o)
end

## DIMENSION AGNOSTIC ELASTIC KERNELS

@parallel function elastic_iter_params!(
    dτ_Rho::AbstractArray,
    Gdτ::AbstractArray,
    ητ::AbstractArray,
    Vpdτ::T,
    G::T,
    dt::M,
    Re::T,
    r::T,
    max_li::T,
) where {T,M}
    @all(dτ_Rho) = Vpdτ * max_li / Re / (one(T) / (one(T) / @all(ητ) + one(T) / (G * dt)))
    @all(Gdτ) = Vpdτ^2 / @all(dτ_Rho) / (r + T(2.0))
    return nothing
end

@parallel function elastic_iter_params!(
    dτ_Rho::AbstractArray,
    Gdτ::AbstractArray,
    ητ::AbstractArray,
    Vpdτ::T,
    G::AbstractArray,
    dt::M,
    Re::T,
    r::T,
    max_li::T,
) where {T,M}
    @all(dτ_Rho) =
        Vpdτ * max_li / Re / (one(T) / (one(T) / @all(ητ) + one(T) / (@all(G) * dt)))
    @all(Gdτ) = Vpdτ^2 / @all(dτ_Rho) / (r + T(2.0))
    return nothing
end


function update_τ_o!(stokes::StokesArrays{ViscoElastic,A,B,C,D,2}) where {A,B,C,D}
    τxx, τyy, τxy, τxy_c = stokes.τ.xx, stokes.τ.yy, stokes.τ.xy, stokes.τ.xy_c
    τxx_o, τyy_o, τxy_o, τxy_o_c = stokes.τ_o.xx,
    stokes.τ_o.yy, stokes.τ_o.xy,
    stokes.τ_o.xy_c
    @parallel update_τ_o!(τxx_o, τyy_o, τxy_o, τxy_o_c, τxx, τyy, τxy, τxy_c)
    return nothing
end

@parallel function update_τ_o!(τxx_o, τyy_o, τxy_o, τxy_o_c, τxx, τyy, τxy, τxy_c)
    @all(τxx_o) = @all(τxx)
    @all(τyy_o) = @all(τyy)
    @all(τxy_o) = @all(τxy)
    @all(τxy_o_c) = @all(τxy_c)
    return nothing
end

@parallel function compute_∇V!(∇V, Vx, Vy, _dx, _dy)
    @all(∇V) = @d_xi(Vx) * _dx + @d_yi(Vy) * _dy
    return nothing
end

@parallel function compute_strain_rate!(εxx, εyy, εxy, ∇V, Vx, Vy, _dx, _dy)
    @all(εxx) = @d_xi(Vx) * _dx - @all(∇V) / 3.0
    @all(εyy) = @d_yi(Vy) * _dy - @all(∇V) / 3.0
    @all(εxy) = 0.5 * (@d_ya(Vx) * _dy + @d_xa(Vy) * _dx)
    return nothing
end

# Continuity equation

## Incompressible 
@parallel function compute_P!(P, RP, ∇V, η, r, θ_dτ)
    @all(RP) = -@all(∇V)
    @all(P) = @all(P) + @all(RP) * r / θ_dτ * @all(η)
    return nothing
end

## Compressible 
@parallel function compute_P!(P, P_old, RP, ∇V, η, K, dt, r, θ_dτ)
    @all(RP) = -@all(∇V) - (@all(P) - @all(P_old)) / (@all(K) * dt)
    @all(P) = @all(P) + @all(RP) / (1.0 / (r / θ_dτ * @all(η)) + 1.0 / (@all(K) * dt))
    return nothing
end

## Compressible - GeoParams
@parallel function compute_P!(P, P_old, RP, ∇V, η, K::Number, dt, r, θ_dτ)
    @all(RP) = -@all(∇V) - (@all(P) - @all(P_old)) / (K * dt)
    @all(P) = @all(P) + @all(RP) / (1.0 / (r / θ_dτ * @all(η)) + 1.0 / (K * dt))
    return nothing
end

@parallel_indices (i, j) function compute_P!(
    P, P_old, RP, ∇V, η, rheology::NTuple{N,MaterialParams}, phase, dt, r, θ_dτ
) where {N}
    @inbounds begin
        RP[i, j] =
            -∇V[i, j] - (P[i, j] - P_old[i, j]) / (get_Kb(rheology, phase[i, j]) * dt)
        P[i, j] =
            P[i, j] +
            RP[i, j] /
            (1.0 / (r / θ_dτ * η[i, j]) + 1.0 / (get_Kb(rheology, phase[i, j]) * dt))
    end
    return nothing
end

@parallel function compute_V!(Vx, Vy, P, τxx, τyy, τxyv, ηdτ, ρgx, ρgy, ητ, _dx, _dy)
    @inn(Vx) =
        @inn(Vx) +
        (-@d_xa(P) * _dx + @d_xa(τxx) * _dx + @d_yi(τxyv) * _dy - @av_xa(ρgx)) * ηdτ /
        @harm_xa(ητ)
    @inn(Vy) =
        @inn(Vy) +
        (-@d_ya(P) * _dy + @d_ya(τyy) * _dy + @d_xi(τxyv) * _dx - @av_ya(ρgy)) * ηdτ /
        @harm_ya(ητ)
    return nothing
end

@parallel_indices (i, j) function compute_Res!(Rx, Ry, P, τxx, τyy, τxy, ρgx, ρgy, _dx, _dy)
    # Again, indices i, j are captured by the closure
    Base.@propagate_inbounds @inline d_xa(A) = (A[i + 1, j] - A[i, j]) * _dx
    Base.@propagate_inbounds @inline d_ya(A) = (A[i, j + 1] - A[i, j]) * _dy
    Base.@propagate_inbounds @inline d_xi(A) = (A[i + 1, j + 1] - A[i, j + 1]) * _dx
    Base.@propagate_inbounds @inline d_yi(A) = (A[i + 1, j + 1] - A[i + 1, j]) * _dy
    Base.@propagate_inbounds @inline av_xa(A) = (A[i + 1, j] + A[i, j]) * 0.5
    Base.@propagate_inbounds @inline av_ya(A) = (A[i, j + 1] + A[i, j]) * 0.5

    @inbounds begin
        if i ≤ size(Rx, 1) && j ≤ size(Rx, 2)
            Rx[i, j] = d_xa(τxx) + d_yi(τxy) - d_xa(P) - av_xa(ρgx)
        end
        if i ≤ size(Ry, 1) && j ≤ size(Ry, 2)
            Ry[i, j] = d_ya(τyy) + d_xi(τxy) - d_ya(P) - av_ya(ρgy)
        end
    end
    return nothing
end

@parallel_indices (i, j) function compute_V_Res!(
    Vx, Vy, Rx, Ry, P, τxx, τyy, τxy, ρgx, ρgy, ητ, ηdτ, _dx, _dy
)

    # Again, indices i, j are captured by the closure
    Base.@propagate_inbounds @inline d_xa(A) = (A[i + 1, j] - A[i, j]) * _dx
    Base.@propagate_inbounds @inline d_ya(A) = (A[i, j + 1] - A[i, j]) * _dy
    Base.@propagate_inbounds @inline d_xi(A) = (A[i + 1, j + 1] - A[i, j + 1]) * _dx
    Base.@propagate_inbounds @inline d_yi(A) = (A[i + 1, j + 1] - A[i + 1, j]) * _dy
    Base.@propagate_inbounds @inline av_xa(A) = (A[i + 1, j] + A[i, j]) * 0.5
    Base.@propagate_inbounds @inline av_ya(A) = (A[i, j + 1] + A[i, j]) * 0.5

    @inbounds begin
        if all((i, j) .≤ size(Rx))
            R = Rx[i, j] = d_xa(τxx) + d_yi(τxy) - d_xa(P) - av_xa(ρgx)
            Vx[i + 1, j + 1] += R * ηdτ / av_xa(ητ)
        end
        if all((i, j) .≤ size(Ry))
            R = Ry[i, j] = d_ya(τyy) + d_xi(τxy) - d_ya(P) - av_ya(ρgy)
            Vy[i + 1, j + 1] += R * ηdτ / av_ya(ητ)
        end
    end

    return nothing
end

# Stress calculation

# viscous
@parallel function compute_τ!(τxx, τyy, τxy, εxx, εyy, εxy, η, θ_dτ)
    @all(τxx) = @all(τxx) + (-@all(τxx) + 2.0 * @all(η) * @all(εxx)) * 1.0 / (θ_dτ + 1.0)
    @all(τyy) = @all(τyy) + (-@all(τyy) + 2.0 * @all(η) * @all(εyy)) * 1.0 / (θ_dτ + 1.0)
    @inn(τxy) = @inn(τxy) + (-@inn(τxy) + 2.0 * @av(η) * @inn(εxy)) * 1.0 / (θ_dτ + 1.0)
    return nothing
end

# visco-elastic
@parallel function compute_τ!(
    τxx, τyy, τxy, τxx_o, τyy_o, τxy_o, εxx, εyy, εxy, η, G, θ_dτ, dt
)
    @all(τxx) =
        @all(τxx) +
        (
            -(@all(τxx) - @all(τxx_o)) * @all(η) / (@all(G) * dt) - @all(τxx) +
            2.0 * @all(η) * @all(εxx)
        ) * 1.0 / (θ_dτ + @all(η) / (@all(G) * dt) + 1.0)
    @all(τyy) =
        @all(τyy) +
        (
            -(@all(τyy) - @all(τyy_o)) * @all(η) / (@all(G) * dt) - @all(τyy) +
            2.0 * @all(η) * @all(εyy)
        ) * 1.0 / (θ_dτ + @all(η) / (@all(G) * dt) + 1.0)
    @inn(τxy) =
        @inn(τxy) +
        (
            -(@inn(τxy) - @inn(τxy_o)) * @av(η) / (@av(G) * dt) - @inn(τxy) +
            2.0 * @av(η) * @inn(εxy)
        ) * 1.0 / (θ_dτ + @av(η) / (@av(G) * dt) + 1.0)

    return nothing
end

# visco-elasto-plastic with GeoParams
@parallel_indices (i, j) function compute_τ_gp!(
    τxx,
    τyy,
    τxy,
    τII,
    τxx_o,
    τyy_o,
    τxyv_o,
    εxx,
    εyy,
    εxyv,
    η,
    η_vep,
    z,
    T,
    rheology,
    dt,
    θ_dτ,
)
    # convinience closure
    @inline gather(A) = A[i, j], A[i + 1, j], A[i, j + 1], A[i + 1, j + 1]
    @inline av(T) = (T[i, j] + T[i + 1, j] + T[i, j + 1] + T[i + 1, j + 1]) * 0.25

    return nothing
end

# visco-elasto-plastic with GeoParams - with multiple phases
@parallel_indices (i, j) function compute_τ_gp!(
    τxx,
    τyy,
    τxy,
    τII,
    τxx_o,
    τyy_o,
    τxyv_o,
    εxx,
    εyy,
    εxyv,
    η,
    η_vep,
    z,
    T,
    phase_v,
    phase_c,
    args_η,
    rheology,
    dt,
    θ_dτ,
)
    #! format: off
    # convinience closure
    Base.@propagate_inbounds @inline function gather(A)
        A[i, j], A[i + 1, j], A[i, j + 1], A[i + 1, j + 1]
    end
    Base.@propagate_inbounds @inline function av(T)
        (T[i, j] + T[i + 1, j] + T[i, j + 1] + T[i + 1, j + 1]) * 0.25
    end
    #! format: on

    @inbounds begin
        k = keys(args_η)
        v = getindex.(values(args_η), i, j)
        # # numerics
        # dτ_r                = 1.0 / (θ_dτ + η[i, j] / (get_G(rheology[1]) * dt) + 1.0) # original
        dτ_r = 1.0 / (θ_dτ / η[i, j] + 1.0 / η_vep[i, j]) # equivalent to dτ_r = @. 1.0/(θ_dτ + η/(G*dt) + 1.0)
        # # Setup up input for GeoParams.jl
        args = (; zip(k, v)..., dt=dt, T=av(T), τII_old=0.0)
        εij_p = εxx[i, j] + 1e-25, εyy[i, j] + 1e-25, gather(εxyv) .+ 1e-25
        τij_p_o = τxx_o[i, j], τyy_o[i, j], gather(τxyv_o)
        phases = phase_c[i, j], phase_c[i, j], gather(phase_v) # for now hard-coded for a single phase
        # update stress and effective viscosity
        τij, τII[i, j], ηᵢ = compute_τij(rheology, εij_p, args, τij_p_o, phases)
        τxx[i, j] += dτ_r * (-(τxx[i, j]) + τij[1]) / ηᵢ # NOTE: from GP Tij = 2*η_vep * εij
        τyy[i, j] += dτ_r * (-(τyy[i, j]) + τij[2]) / ηᵢ
        τxy[i, j] += dτ_r * (-(τxy[i, j]) + τij[3]) / ηᵢ
        η_vep[i, j] = ηᵢ
    end

    return nothing
end

#update density depending on melt fraction and number of phases
@parallel_indices (i, j) function compute_ρg!(ρg, ϕ, rheology, args)
    i1, j1 = i + 1, j + 1
    i2 = i + 2
    @inline av(T) = 0.25 * (T[i1, j] + T[i2, j] + T[i1, j1] + T[i2, j1])

    ρg[i, j] =
        compute_density_ratio(
            (1 - ϕ[i, j], ϕ[i, j], 0.0), rheology, (; T=av(args.T), P=args.P[i, j])
        ) * compute_gravity(rheology[1])
    return nothing
end
