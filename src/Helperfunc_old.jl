## Helperfunctions for MTK vs JustRelax

using Printf, LinearAlgebra, GeoParams, Parameters, JustRelax, ParallelStencil
import JustRelax: @tuple
import ParallelStencil.INDICES

# to be added to GP# Viscosity with partial melting -----------------------------------------
"""
MeltViscous(η_s = 1e22 * Pa*s,η_f = 1e16 * Pa*s,ϕ = 0.0 * NoUnits,S = 1.0 * NoUnits,mfac = -2.8 * NoUnits)
Defines a effective viscosity of partially molten rock as: 
```math  
\\eta  = \\min(\\eta_f (1-S(1-\\phi))^m_{fac})
```
"""
@with_kw_noshow struct MeltViscous{T,U1,U2} <: AbstractCreepLaw{T}
η_s::GeoUnit{T,U1} = 1e22 * Pa*s # rock's viscosity
η_f::GeoUnit{T,U1} = 1e16 * Pa*s # magma's viscosity 
S::GeoUnit{T,U2} = 1.0 * NoUnits # factors for hexagons
mfac::GeoUnit{T,U2} = -2.8 * NoUnits # factors for hexagons
end
MeltViscous(a...) = MeltViscous(convert.(GeoUnit, a)...)

#Utils (AdM)
function unroll(f::F, args::NTuple{N,T}) where {F,N,T}
    ntuple(Val(N)) do i
        f(args[i])
    end
end

macro unroll(f, args)
    return esc(:(unroll($f, $args)))
end


function copy_arrays_GPU2CPU!(T_CPU::AbstractArray,  ϕ_CPU::AbstractArray, T_GPU::AbstractArray, ϕ_GPU::AbstractArray)

    T_CPU  .= Array(T_GPU)
    ϕ_CPU  .= Array(ϕ_GPU)
    
    return nothing 
end

#stress rotation JR 050723
@parallel_indices (i, j) function rotate_stress!(V, τ::NTuple{3,T}, _di, dt) where {T}
    @inbounds rotate_stress!(V, τ, (i, j), _di, dt)
    return nothing
end

@parallel_indices (i, j, k) function rotate_stress!(V, τ::NTuple{6,T}, _di, dt) where {T}
    @inbounds rotate_stress!(V, τ, (i, j, k), _di, dt)
    return nothing
end

"""
    Jaumann derivative

τij_o += v_k * ∂τij_o/∂x_k - ω_ij * ∂τkj_o + ∂τkj_o * ω_ij

"""
Base.@propagate_inbounds function rotate_stress!(
    V, τ::NTuple{N,T}, idx, _di, dt
) where {N,T}
    ## 1) Advect stress
    Vᵢⱼ = velocity2center(V..., idx...) # averages @ cell center
    τij_adv = advect_stress(τ..., Vᵢⱼ..., idx..., _di...)

    ## 2) Rotate stress
    # average ∂Vx/∂y @ cell center
    ∂V∂x = cross_derivatives(V..., _di..., idx...)
    # compute xy component of the vorticity tensor; normal components = 0.0
    ω = compute_vorticity(∂V∂x)
    # stress tensor in Voigt notation
    τ_voigt = ntuple(Val(N)) do k
        Base.@_inline_meta
        τ[k][idx...]
    end
    # actually rotate stress tensor
    τr_voigt = GeoParams.rotate_elastic_stress2D(ω, τ_voigt, dt)

    ## 3) Update stress
    for k in 1:N
        τ[k][idx...] = muladd(τij_adv[k], dt, τr_voigt[k])
    end
    return nothing
end

# 2D
Base.@propagate_inbounds function advect_stress(τxx, τyy, τxy, Vx, Vy, i, j, _dx, _dy)
    τ = τxx, τyy, τxy
    τ_adv = ntuple(Val(3)) do k
        Base.@_inline_meta
        dx_right, dx_left, dy_up, dy_down = upwind_derivatives(τ[k], i, j)
        advection_term(Vx, Vy, dx_right, dx_left, dy_up, dy_down, _dx, _dy)
    end
    return τ_adv
end

# 3D
Base.@propagate_inbounds function advect_stress(
    τxx, τyy, τzz, τyz, τxz, τxy, Vx, Vy, Vz, i, j, k, _dx, _dy, _dz
)
    τ = τxx, τyy, τzz, τyz, τxz, τxy
    τ_adv = ntuple(Val(6)) do l
        Base.@_inline_meta
        dx_right, dx_left, dy_back, dy_front, dz_up, dz_down = upwind_derivatives(
            τ[l], i, j, k
        )
        advection_term(
            Vx,
            Vy,
            Vz,
            dx_right,
            dx_left,
            dy_back,
            dy_front,
            dz_up,
            dz_down,
            _dx,
            _dy,
            _dz,
        )
    end
    return τ_adv
end

# 2D
Base.@propagate_inbounds function upwind_derivatives(A, i, j)
    nx, ny = size(A)
    center = A[i, j]
    # dx derivatives
    x_left = i - 1 > 1 ? A[i - 1, j] : 0.0
    x_right = i + 1 < nx ? A[i + 1, j] : 0.0
    dx_right = x_right - center
    dx_left = center - x_left
    # dy derivatives
    y_down = j - 1 > 1 ? A[i, j - 1] : 0.0
    y_up = j + 1 < ny ? A[i, j + 1] : 0.0
    dy_up = y_up - center
    dy_down = center - y_down

    return dx_right, dx_left, dy_up, dy_down
end

# 3D
Base.@propagate_inbounds function upwind_derivatives(A, i, j, k)
    nx, ny, nz = size(A)
    center = A[i, j, k]
    x_left = x_right = y_front = y_back = z_down = z_up = 0.0
    # dx derivatives
    i - 1 > 1 && (x_left = A[i - 1, j, k])
    i + 1 < nx && (x_right = A[i + 1, j, k])
    dx_right = x_right - center
    dx_left = center - x_left
    # dy derivatives
    j - 1 > 1 && (y_front = A[i, j - 1, k])
    j + 1 < ny && (y_back = A[i, j + 1, k])
    dy_back = y_back - center
    dy_front = center - y_front
    # dz derivatives
    k - 1 > 1 && (z_down = A[i, j, k - 1])
    k + 1 < nz && (z_up = A[i, j, k + 1])
    dz_up = z_up - center
    dz_down = center - z_down

    return dx_right, dx_left, dy_back, dy_front, dz_up, dz_down
end

# 2D
@inline function advection_term(Vx, Vy, dx_right, dx_left, dy_up, dy_down, _dx, _dy)
    return (Vx > 0 ? dx_right : dx_left) * Vx * _dx + (Vy > 0 ? dy_up : dy_down) * Vy * _dy
end

# 3D
@inline function advection_term(
    Vx, Vy, Vz, dx_right, dx_left, dy_back, dy_front, dz_up, dz_down, _dx, _dy, _dz
)
    return (Vx > 0 ? dx_right : dx_left) * Vx * _dx +
           (Vy > 0 ? dy_back : dy_front) * Vy * _dy +
           (Vz > 0 ? dz_up : dz_down) * Vz * _dz
end

# averages @ cell center 2D
Base.@propagate_inbounds function velocity2center(Vx, Vy, i, j)
    i1, j1 = @add 1 i j
    Vxᵢⱼ = 0.5 * (Vx[i, j1] + Vx[i1, j1])
    Vyᵢⱼ = 0.5 * (Vy[i1, j] + Vy[i1, j1])
    return Vxᵢⱼ, Vyᵢⱼ
end

# averages @ cell center 3D
Base.@propagate_inbounds function velocity2center(Vx, Vy, Vz, i, j, k)
    i1, j1, k1 = @add 1 i j k
    Vxᵢⱼ = 0.5 * (Vx[i, j1, k1] + Vx[i1, j1, k1])
    Vyᵢⱼ = 0.5 * (Vy[i1, j, k1] + Vy[i1, j1, k1])
    Vzᵢⱼ = 0.5 * (Vz[i1, j1, k] + Vz[i1, j1, k1])
    return Vxᵢⱼ, Vyᵢⱼ, Vzᵢⱼ
end

# 2D
Base.@propagate_inbounds function cross_derivatives(Vx, Vy, _dx, _dy, i, j)
    i1, j1 = @add 1 i j
    i2, j2 = @add 2 i j
    # average @ cell center
    ∂Vx∂y =
        0.25 *
        _dy *
        (
            Vx[i, j1] - Vx[i, j] + Vx[i, j2] - Vx[i, j1] + Vx[i1, j1] - Vx[i1, j] +
            Vx[i1, j2] - Vx[i1, j1]
        )
    ∂Vy∂x =
        0.25 *
        _dx *
        (
            Vy[i1, j] - Vy[i, j] + Vy[i2, j] - Vy[i1, j] + Vy[i1, j1] - Vy[i, j1] +
            Vy[i2, j1] - Vy[i1, j1]
        )
    return ∂Vx∂y, ∂Vy∂x
end

Base.@propagate_inbounds function cross_derivatives(Vx, Vy, Vz, _dx, _dy, _dz, i, j, k)
    i1, j1, k2 = @add 1 i j k
    i2, j2, k2 = @add 2 i j k
    # cross derivatives @ cell centers
    ∂Vx∂y =
        0.25 *
        _dy *
        (
            Vx[i, j1, k1] - Vx[i, j, k1] + Vx[i, j2, k1] - Vx[i, j1, k1] + Vx[i1, j1, k1] -
            Vx[i1, j, k1] + Vx[i1, j2, k1] - Vx[i1, j1, k1]
        )
    ∂Vx∂z =
        0.25 *
        _dz *
        (
            Vx[i, j1, k1] - Vx[i, j, k] + Vx[i, j2, k2] - Vx[i, j1, k1] + Vx[i1, j1, k1] -
            Vx[i1, j, k] + Vx[i1, j2, k2] - Vx[i1, j1, k1]
        )
    ∂Vy∂x =
        0.25 *
        _dx *
        (
            Vy[i1, j, ki] - Vy[i, j, ki] + Vy[i2, j, ki] - Vy[i1, j, ki] + Vy[i1, j1, ki] -
            Vy[i, j1, ki] + Vy[i2, j1, ki] - Vy[i1, j1, ki]
        )
    ∂Vy∂z =
        0.25 *
        _dz *
        (
            Vy[i1, j, k1] - Vy[i, j, k] + Vy[i2, j, k2] - Vy[i1, j, k1] + Vy[i1, j1, k1] -
            Vy[i, j1, k] + Vy[i2, j1, k2] - Vy[i1, j1, k1]
        )
    ∂Vz∂x =
        0.25 *
        _dx *
        (
            Vz[i1, j, k] - Vz[i, j, k] + Vz[i2, j, k] - Vz[i1, j, k] + Vz[i1, j1, k1] -
            Vz[i, j1, 1k] + Vz[i2, j1, k1] - Vz[i1, j1, 1k]
        )
    ∂Vz∂y =
        0.25 *
        _dy *
        (
            Vz[i1, j, k] - Vz[i, j, k] + Vz[i2, j, k] - Vz[i1, j, k] + Vz[i1, j1, k1] -
            Vz[i, j1, k1] + Vz[i2, j1, k1] - Vz[i1, j1, k1]
        )
    return ∂Vx∂y, ∂Vx∂z, ∂Vy∂x, ∂Vy∂z, ∂Vz∂x, ∂Vz∂y
end

Base.@propagate_inbounds @inline function compute_vorticity(∂V∂x::NTuple{2,T}) where {T}
    return ∂V∂x[1] - ∂V∂x[2]
end # 2D
Base.@propagate_inbounds @inline function compute_vorticity(∂V∂x::NTuple{3,T}) where {T}
    return ∂V∂x[3] - ∂V∂x[2], ∂V∂x[1] - ∂V∂x[3], ∂V∂x[2] - ∂V∂x[1]
end # 3D


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

# #update density depending on melt fraction and number of phases
# @parallel_indices (i, j) function compute_ρg!(ρg, ϕ, rheology, args)
#     i1, j1 = i + 1, j + 1
#     i2 = i + 2
#     @inline av(T) = 0.25 * (T[i1, j] + T[i2, j] + T[i1, j1] + T[i2, j1])

#     ρg[i, j] =
#         compute_density_ratio(
#             (1 - ϕ[i, j], ϕ[i, j], 0.0), rheology, (; T=av(args.T), P=args.P[i, j])
#         ) * compute_gravity(rheology[1])
#     return nothing
# end
# ----------------------------

@parallel_indices (i, j) function init_Viscosity!(η, Phases, rheology)
    @inbounds η[i, j] = GeoParams.nphase(get_viscosity, Phases[i, j], rheology)
    return nothing
end

get_viscosity(v)=v.CompositeRheology[1][1].η.val

@parallel_indices (i, j) function compute_viscosity_gp!(η, phase_c, phase_v, args, MatParam)

    # convinience closure
    @inline av(T)     = (T[i + 1, j] + T[i + 2, j] + T[i + 1, j + 1] + T[i + 2, j + 1]) * 0.25
    @inline gather(A) = A[i, j], A[i + 1, j], A[i, j + 1], A[i + 1, j + 1] 

    @inbounds begin
        args_ij       = (; dt = args.dt, P = (args.P[i, j]), depth = abs(args.depth[j]), T=(args.T), τII_old=0.0)
        εij_p         = 1.0, 1.0, (1.0, 1.0, 1.0, 1.0)
        τij_p_o       = 0.0, 0.0, (0.0, 0.0, 0.0, 0.0)
        phases        = phase_c[i,j], phase_c[i,j], gather(phase_v) # for now hard-coded for a single phase
        # # update stress and effective viscosity
        _, _, η[i, j] = compute_τij(MatParam, εij_p, args_ij, τij_p_o, phases)
    end
    
    return nothing
end

# plasticity
@parallel_indices (i, j) function initViscosity_pl!(η, v, args)
    @inline av(T) = (T[i + 1, j] + T[i + 2, j] + T[i + 1, j + 1] + T[i + 2, j + 1]) * 0.25

    @inbounds η[i, j] = computeViscosity_εII(v, 1.0, (; T=av(args.T)))

    return nothing
end

@parallel function computeViscosity!(η, ϕ, S, mfac, η_f, η_s)
    # We assume that η is f(ϕ), following Deubelbeiss, Kaus, Connolly (2010) EPSL 
    # factors for hexagons
    @all(η) = min(
        η_f * (1.0 - S * (1.0 - @all(ϕ)))^mfac,
        η_s, # upper cutoff
    )
    return nothing
end

@parallel function update_viscosity(η, ϕ, S, mfac, η_f, η_s)                   #update Viscosity
    # We assume that η is f(ϕ), following Deubelbeiss, Kaus, Connolly (2010) EPSL 
    # factors for hexagons
    @all(η) = min(
        η_f * (1.0 - S * (1.0 - @all(ϕ)))^mfac,
        η_s, # upper cutoff
    )
    return nothing
end

@parallel_indices (i,j)  function update_G!(G, MatParam, phase_c)
    G[i,j] = get_G(MatParam, phase_c[i, j])
    return nothing
end

@parallel_indices (i,j)  function update_Kb!(Kb, MatParam, phase_c)
    Kb[i,j] = get_Kb(MatParam, phase_c[i, j])
    return nothing
end

@parallel_indices (i, j) function compute_melt_fraction!(ϕ, rheology, phase_c, args)
    ϕ[i, j] = compute_meltfraction(rheology, phase_c[i, j], ntuple_idx(args, i, j))
    return nothing
end

#update density depending on melt fraction and number of phases
@parallel_indices (i, j) function compute_ρg!(ρg, ϕ, rheology, args)
    # i1, j1 = i + 1, j + 1
    # i2 = i + 2
    # @inline av(T) = 0.25 * (T[i1, j] + T[i2, j] + T[i1, j1] + T[i2, j1])

    ρg[i, j] = 
        compute_density_ratio(
            # (1 - ϕ[i, j], ϕ[i, j], 0.0), rheology, (; T=av(args.T), P=args.P[i, j])
            (1 - ϕ[i, j], ϕ[i, j], 0.0), rheology, (; T=args.T[i, j], P=args.P[i, j])
        ) * compute_gravity(rheology[1])
    return nothing
end

@parallel_indices (i, j) function compute_ρg2!(ρg, ϕ, rheology, args)
    # i1, j1 = i + 1, j + 1
    # i2 = i + 2
    # @inline av(T) = 0.25 * (T[i1, j] + T[i2, j] + T[i1, j1] + T[i2, j1])

    ρg[i, j] = ρg[i, j] * (1-0.95) + 0.95 *
        compute_density_ratio(
            # (1 - ϕ[i, j], ϕ[i, j], 0.0), rheology, (; T=av(args.T), P=args.P[i, j])
            (1 - ϕ[i, j], ϕ[i, j], 0.0), rheology, (; T=args.T[i, j], P=args.P[i, j])
        ) * compute_gravity(rheology[1])
    return nothing
end


@parallel_indices (i, j) function update_phase(phase, ϕ)

    if !(phase[i, j] == 2) && ϕ[i, j] > 1e-2
        phase[i, j] = 2
    end
    
    return nothing
end

@parallel_indices (i, j) function compute_ρg_phase!(ρg, phase, rheology, args)

    ρg[i, j] = 
        compute_density(
            rheology,  phase[i, j], (; T=args.T[i, j], P=args.P[i, j])
        ) * compute_gravity(rheology[1])
    return nothing
end

# @parallel_indices (i, j) function compute_ρg_new!(ρg,rheology, phases, args)
#     ρg[i, j] =   compute_density!(ρg, rheology, phases, args) * compute_gravity(rheology[1])
#     return nothing
# end

@parallel_indices (i, j, k) function _elliptical_perturbation!(T, δT, xc, yc, zc, r, x, y, z)
    if (((x[i]-xc ))^2 + ((y[j] - yc))^2 + ((z[k] - zc))^2) ≤ r^2
        T[i,j,k] *= δT/100 + 1
    end
    return nothing
end

@parallel function compute_maxRatio!(Musτ2::AbstractArray, Musτ::AbstractArray)
    @inn(Musτ2) = @maxloc(Musτ) / @minloc(Musτ)
    return nothing
end

@parallel function compute_qT!(
    qTx::AbstractArray,
    qTy::AbstractArray,
    T::AbstractArray,
    κ::Number,
    dx::Number,
    dy::Number,
)
    @all(qTx) = -κ * @d_xi(T) / dx
    @all(qTy) = -κ * @d_yi(T) / dy
    return nothing
end

@parallel_indices (ix, iy) function advect_T!(
    dT_dt::AbstractArray,
    qTx::AbstractArray,
    qTy::AbstractArray,
    T::AbstractArray,
    Vx::AbstractArray,
    Vy::AbstractArray,
    dx::Number,
    dy::Number,
)
    if (ix <= size(dT_dt, 1) && iy <= size(dT_dt, 2))
        dT_dt[ix, iy] =
            -((qTx[ix + 1, iy] - qTx[ix, iy]) / dx + (qTy[ix, iy + 1] - qTy[ix, iy]) / dy) -
            (Vx[ix + 1, iy + 1] > 0) *
            Vx[ix + 1, iy + 1] *
            (T[ix + 1, iy + 1] - T[ix, iy + 1]) / dx -
            (Vx[ix + 2, iy + 1] < 0) *
            Vx[ix + 2, iy + 1] *
            (T[ix + 2, iy + 1] - T[ix + 1, iy + 1]) / dx -
            (Vy[ix + 1, iy + 1] > 0) *
            Vy[ix + 1, iy + 1] *
            (T[ix + 1, iy + 1] - T[ix + 1, iy]) / dy -
            (Vy[ix + 1, iy + 2] < 0) *
            Vy[ix + 1, iy + 2] *
            (T[ix + 1, iy + 2] - T[ix + 1, iy + 1]) / dy
    end
    return nothing
end

@parallel function update_T!(
    T::AbstractArray, T_old::AbstractArray, dT_dt::AbstractArray, Δt::Number
)
    @inn(T) = @inn(T_old) + @all(dT_dt) * Δt
    return nothing
end

@parallel_indices (ix, iy) function no_fluxY_T!(T::AbstractArray)
    if (ix == size(T, 1) && iy <= size(T, 2))
        T[ix, iy] = T[ix - 1, iy]
    end
    if (ix == 1 && iy <= size(T, 2))
        T[ix, iy] = T[ix + 1, iy]
    end
    return nothing
end

# compute τ with plasticity

@inline isplastic(x::AbstractPlasticity) = true
@inline isplastic(x) = false
@inline plastic_params(v) = plastic_params(v.CompositeRheology[1].elements)

@generated function plastic_params(v::NTuple{N, Any}) where N
    quote
        Base.@_inline_meta
        Base.@nexprs $N i -> isplastic(v[i]) && return true, v[i].C.val, v[i].sinϕ.val, v[i].η_vp.val
        (false, 0.0, 0.0, 0.0)
    end
end

@generated function plastic_params(v::NTuple{N, Any}, phase) where N
    quote
        Base.@_inline_meta
        Base.@nexprs $N i -> i==phase && isplastic(v[i]) && return true, v[i].C.val, v[i].sinϕ.val, v[i].η_vp.val
        (false, 0.0, 0.0, 0.0)
    end
end


@parallel_indices (i, j) function compute_τ_new!(
    τxx,
    τyy,
    τxy,
    τII,
    τxx_old,
    τyy_old,
    τxyv_old,
    εxx,
    εyy,
    εxyv,
    P,
    η,
    η_vep,
    MatParam,
    phase_c,
    dt,
    θ_dτ,
    λ0
)
    nx, ny = size(η)

    # convinience closure
    @inline Base.@propagate_inbounds gather(A) = A[i, j], A[i + 1, j], A[i, j + 1], A[i + 1, j + 1] 
    @inline Base.@propagate_inbounds av(A)     = (A[i + 1, j] + A[i + 2, j] + A[i + 1, j + 1] + A[i + 2, j + 1]) * 0.25
    @inline Base.@propagate_inbounds function maxloc(A)
        max(
            A[i, j],
            A[min(i+1, nx), j],
            A[max(i-1, 1), j],
            A[i, min(j+1, ny)],
            A[i, max(j-1, 1),],
        )
    end

    @inbounds begin
        # _Gdt        = inv(get_G(MatParam[1]) * dt)
        _Gdt        = inv(get_G(MatParam, phase_c[i,j]) * dt)
        # _Gdt        = inv(G[i,j]* dt)
        ηij         = η[i, j]
        dτ_r        = inv(θ_dτ + ηij * _Gdt + 1.0) # original
        # cache tensors
        εij_p       = εxx[i, j], εyy[i, j], gather(εxyv)
        τij_p_o     = τxx_old[i,j], τyy_old[i,j], gather(τxyv_old) 
        τij         = τxx[i,j], τyy[i,j], τxy[i, j]

        εxy_p       = 0.25 * sum(εij_p[3])
        τxy_p_o     = 0.25 * sum(τij_p_o[3])

        # Stress increment
        dτxx      = dτ_r * (-(τij[1] - τij_p_o[1]) * ηij * _Gdt - τij[1] + 2.0 * ηij * (εij_p[1]))
        dτyy      = dτ_r * (-(τij[2] - τij_p_o[2]) * ηij * _Gdt - τij[2] + 2.0 * ηij * (εij_p[2])) 
        dτxy      = dτ_r * (-(τij[3] - τxy_p_o   ) * ηij * _Gdt - τij[3] + 2.0 * ηij * (εxy_p   )) 
        τII_trial = GeoParams.second_invariant(dτxx + τij[1], dτyy + τij[2], dτxy + τij[3])

        is_pl, C, sinϕ, η_reg = plastic_params(MatParam, phase_c[i,j])
        τy = C + P[i,j]*sinϕ

        if is_pl && τII_trial > τy && P[i,j] > 0.0
            # yield function
            F            = τII_trial - τy
            λ = λ0[i,j]  = 0.8 * λ0[i,j] + 0.2 * (F>0.0) * F /(ηij * 1 + η_reg) * is_pl
            
            λdQdτxx      = 0.5 * (τij[1] + dτxx) / τII_trial * λ
            λdQdτyy      = 0.5 * (τij[2] + dτyy) / τII_trial * λ
            λdQdτxy      = 0.5 * (τij[3] + dτxy) / τII_trial * λ
           
            # corrected stress
            dτxx_pl  = dτ_r * (-(τij[1] - τij_p_o[1]) * ηij * _Gdt - τij[1] + 2.0 * ηij * (εij_p[1] - λdQdτxx))
            dτyy_pl  = dτ_r * (-(τij[2] - τij_p_o[2]) * ηij * _Gdt - τij[2] + 2.0 * ηij * (εij_p[2] - λdQdτyy)) 
            dτxy_pl  = dτ_r * (-(τij[3] - τxy_p_o)    * ηij * _Gdt - τij[3] + 2.0 * ηij * (εxy_p    - λdQdτxy)) 
            τxx[i,j] += dτxx_pl
            τyy[i,j] += dτyy_pl
            τxy[i,j] += dτxy_pl
        
            # visco-elastic strain rates
            εxx_ve     = εij_p[1] + 0.5 * τij_p_o[1] * _Gdt
            εyy_ve     = εij_p[2] + 0.5 * τij_p_o[2] * _Gdt
            εxy_ve     = εxy_p    + 0.5 * τxy_p_o    * _Gdt
            εII_ve     = GeoParams.second_invariant(εxx_ve, εyy_ve, εxy_ve)
            τII[i,j]   = GeoParams.second_invariant(τxx[i, j], τyy[i, j], τxy[i, j])
            η_vep[i,j] = τII[i,j] * 0.5 / εII_ve

        else
            τxx[i,j] += dτxx
            τyy[i,j] += dτyy
            τxy[i,j] += dτxy

            # visco-elastic strain rates
            τII[i,j]   = GeoParams.second_invariant(τxx[i, j], τyy[i, j], τxy[i, j])
            η_vep[i,j] = ηij
        end
        
        # η_vep[i,j] = ηij
    end
    
    return nothing
end

# Updated JR PR#41 

#StressRotation
# inner kernel to compute the plastic stress update within Pseudo-Transient stress continuation
function _compute_τ_nonlinear!(
    τ::NTuple{N1,T},
    τII,
    τ_old::NTuple{N1,T},
    ε::NTuple{N1,T},
    P,
    ηij,
    η_vep,
    λ,
    dτ_r,
    _Gdt,
    plastic_parameters,
    idx::Vararg{Integer,N2},
) where {N1,N2,T}

    # cache tensors
    τij, τij_p_o, εij_p = cache_tensors(τ, τ_old, ε, idx...)

    # Stress increment and trial stress
    dτij, τII_trial = compute_stress_increment_and_trial(
        τij, τij_p_o, ηij, εij_p, _Gdt, dτ_r
    )

    # get plastic paremeters (if any...)
    (; is_pl, C, sinϕ, η_reg) = plastic_parameters
    Pij = P[idx...]
    τy = C + Pij * sinϕ

    if isyielding(is_pl, τII_trial, τy, Pij)
        # derivatives plastic stress correction
        dτ_pl, λ[idx...] = compute_dτ_pl(
            τij, dτij, τij_p_o, εij_p, τy, τII_trial, ηij, λ[idx...], η_reg, _Gdt, dτ_r
        )
        τij = τij .+ dτ_pl
        correct_stress!(τ, τij, idx...)
        # visco-elastic strain rates
        εij_ve = ntuple(Val(N1)) do i
            εij_p[i] + 0.5 * τij_p_o[i] * _Gdt
        end
        τII[idx...] = τII_ij = second_invariant(τij...)
        η_vep[idx...] = τII_ij * 0.5 * inv(second_invariant(εij_ve...))

    else
        τij = τij .+ dτij
        correct_stress!(τ, τij, idx...)
        τII[idx...] = second_invariant(τij...)
        η_vep[idx...] = ηij
    end

    return nothing
end

# check if plasticity is active
# @inline isyielding(is_pl, τII_trial, τy, Pij) = is_pl && τII_trial > τy && Pij > 0
@inline isyielding(is_pl, τII_trial, τy, Pij) = is_pl && τII_trial > τy 

@inline JustRelax.compute_dτ_r(θ_dτ, ηij, _Gdt) = inv(θ_dτ + ηij * _Gdt + 1.0)

# cache tensors
function cache_tensors(
    τ::NTuple{3,Any}, τ_old::NTuple{3,Any}, ε::NTuple{3,Any}, idx::Vararg{Integer,2}
)
    @inline av_shear(A) = 0.25 * sum(_gather(A, idx...))

    εij = ε[1][idx...], ε[2][idx...], av_shear(ε[3])
    τij_o = τ_old[1][idx...], τ_old[2][idx...], av_shear(τ_old[3])
    τij = getindex.(τ, idx...)

    return τij, τij_o, εij
end

function cache_tensors(
    τ::NTuple{6,Any}, τ_old::NTuple{6,Any}, ε::NTuple{6,Any}, idx::Vararg{Integer,3}
)
    @inline av_yz(A) = 0.125 * sum(_gather_yz(A, idx...))
    @inline av_xz(A) = 0.125 * sum(_gather_xz(A, idx...))
    @inline av_xy(A) = 0.125 * sum(_gather_xy(A, idx...))

    Val3 = Val(3)

    # normal components of the strain rate and old-stress tensors
    ε_normal = ntuple(i -> ε[i][idx...], Val3)
    τ_old_normal = ntuple(i -> τ_old[i][idx...], Val3)
    # shear components of the strain rate and old-stress tensors
    ε_shear = av_yz(ε[4]), av_xz(ε[5]), av_xy(ε[6])
    τ_old_shear = av_yz(τ_old[4]), av_xz(τ_old[5]), av_xy(τ_old[6])
    # cache ij-th components of the tensors into a tuple in Voigt notation 
    εij = (ε_normal..., ε_shear...)
    τij_o = (τ_old_normal..., τ_old_shear...)
    τij = getindex.(τ, idx...)

    return τij, τij_o, εij
end

function compute_stress_increment_and_trial(
    τij::NTuple{N,T}, τij_p_o, ηij, εij_p, _Gdt, dτ_r
) where {N,T}
    dτij = ntuple(Val(N)) do i
        Base.@_inline_meta
        dτ_r * (-(τij[i] - τij_p_o[i]) * ηij * _Gdt - τij[i] + 2.0 * ηij * εij_p[i])
    end
    return dτij, second_invariant((τij .+ dτij)...)
end

function compute_dτ_pl(
    τij::NTuple{N,T}, dτij, τij_p_o, εij_p, τy, τII_trial, ηij, λ0, η_reg, _Gdt, dτ_r
) where {N,T}
    # yield function
    F = τII_trial - τy
    # Plastic multiplier
    ν = 0.9
    λ = ν * λ0 + (1 - ν) * (F > 0.0) * F * inv(ηij + η_reg)
    λ_τII = λ * 0.5 * inv(τII_trial)

    dτ_pl = ntuple(Val(N)) do i
        Base.@_inline_meta
        # derivatives of the plastic potential
        λdQdτ = (τij[i] + dτij[i]) * λ_τII
        # corrected stress
        dτ_r *
        (-(τij[i] - τij_p_o[i]) * ηij * _Gdt - τij[i] + 2.0 * ηij * (εij_p[i] - λdQdτ))
    end
    return dτ_pl, λ
end

# update the global arrays τ::NTuple{N, AbstractArray} with the local τij::NTuple{3, Float64} at indices idx::Vararg{Integer, N}
@inline function correct_stress!(τ, τij, idx::Vararg{Integer,2})
    Base.@nexprs 3 i -> τ[i][idx...] = τij[i]
end
@inline function correct_stress!(τ, τij, idx::Vararg{Integer,3})
    Base.@nexprs 6 i -> τ[i][idx...] = τij[i]
end
@inline function correct_stress!(τxx, τyy, τxy, τij, idx::Vararg{Integer,2})
    return correct_stress!((τxx, τyy, τxy), τij, idx...)
end
@inline function correct_stress!(τxx, τyy, τzz, τyz, τxz, τxy, τij, idx::Vararg{Integer,3})
    return correct_stress!((τxx, τyy, τzz, τyz, τxz, τxy), τij, idx...)
end

@inline isplastic(x::AbstractPlasticity) = true
@inline isplastic(x) = false

@inline plastic_params(v) = plastic_params(v.CompositeRheology[1].elements)

@generated function plastic_params(v::NTuple{N,Any}) where {N}
    quote
        Base.@_inline_meta
        Base.@nexprs $N i ->
            isplastic(v[i]) && return true, v[i].C.val, v[i].sinϕ.val, v[i].η_vp.val
        (false, 0.0, 0.0, 0.0)
    end
end

## DIMENSION AGNOSTIC KERNELS


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

function update_τ_o!(stokes::StokesArrays{ViscoElastoPlastic,A,B,C,D,2}) where {A,B,C,D}
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
const idx_j = INDICES[2]
macro all_j(A)
    esc(:($A[$idx_j]))
end

@parallel function init_P!(P, ρg, z)
    @all(P) = @all(ρg)*abs(@all_j(z))
    return nothing
end

## Incompressible 
@parallel function compute_P!(P, RP, ∇V, η, r, θ_dτ)
    @all(RP) = -@all(∇V)
    @all(P) = @all(P) + @all(RP) * r / θ_dτ * @all(η)
    return nothing
end

## Compressible 
@parallel function compute_P!(P, P0, RP, ∇V, η, K, dt, r, θ_dτ)
    @all(RP) = -@all(∇V) - (@all(P) - @all(P0)) / (@all(K) * dt)
    @all(P) = @all(P) + @all(RP) / (1.0 / (r / θ_dτ * @all(η)) + 1.0 / (@all(K) * dt))
    return nothing
end

## Compressible - GeoParams
@parallel function compute_P!(P, P0, RP, ∇V, η, K::Number, dt, r, θ_dτ)
    @all(RP) = -@all(∇V) - (@all(P) - @all(P0)) / (K * dt)
    @all(P) = @all(P) + @all(RP) / (1.0 / (r / θ_dτ * @all(η)) + 1.0 / (K * dt))
    return nothing
end

@parallel_indices (i, j) function compute_P!(
    P, P0, RP, ∇V, η, rheology::NTuple{N,MaterialParams}, phase, dt, r, θ_dτ
) where {N}
    @inbounds begin
        _Kdt = inv(get_Kb(rheology, phase[i, j]) * dt)
        RP[i, j] = RP_ij = -∇V[i, j] - (P[i, j] - P0[i, j]) * _Kdt
        P[i, j] += RP_ij * inv(inv(r / θ_dτ * η[i, j]) + _Kdt)
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
    # Closures
    @inline d_xa(A) = _d_xa(A, i, j, _dx)
    @inline d_ya(A) = _d_ya(A, i, j, _dy)
    @inline d_xi(A) = _d_xi(A, i, j, _dx)
    @inline d_yi(A) = _d_yi(A, i, j, _dy)
    @inline av_xa(A) = _av_xa(A, i, j)
    @inline av_ya(A) = _av_ya(A, i, j)

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

    # Closures
    @inline d_xa(A) = _d_xa(A, i, j, _dx)
    @inline d_ya(A) = _d_ya(A, i, j, _dy)
    @inline d_xi(A) = _d_xi(A, i, j, _dx)
    @inline d_yi(A) = _d_yi(A, i, j, _dy)
    @inline av_xa(A) = _av_xa(A, i, j)
    @inline av_ya(A) = _av_ya(A, i, j)

    @inbounds begin
        if all((i, j) .≤ size(Rx))
            R = Rx[i, j] = d_xa(τxx) + d_yi(τxy) - d_xa(P) - av_xa(ρgx)
            Vx[i + 1, j + 1] += R * ηdτ * inv(av_xa(ητ))
        end
        if all((i, j) .≤ size(Ry))
            R = Ry[i, j] = d_ya(τyy) + d_xi(τxy) - d_ya(P) - av_ya(ρgy)
            Vy[i + 1, j + 1] += R * ηdτ * inv(av_ya(ητ))
        end
    end

    return nothing
end

# Stress kernels

# viscous
@parallel function compute_τ!(τxx, τyy, τxy, εxx, εyy, εxy, η, θ_dτ)
    @all(τxx) = @all(τxx) + (-@all(τxx) + 2.0 * @all(η) * @all(εxx)) * inv(θ_dτ + 1.0)
    @all(τyy) = @all(τyy) + (-@all(τyy) + 2.0 * @all(η) * @all(εyy)) * inv(θ_dτ + 1.0)
    @inn(τxy) = @inn(τxy) + (-@inn(τxy) + 2.0 * @av(η) * @inn(εxy)) * inv(θ_dτ + 1.0)
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
        ) * inv(θ_dτ + @all(η) * inv(@all(G) * dt) + 1.0)
    @all(τyy) =
        @all(τyy) +
        (
            -(@all(τyy) - @all(τyy_o)) * @all(η) / (@all(G) * dt) - @all(τyy) +
            2.0 * @all(η) * @all(εyy)
        ) * inv(θ_dτ + @all(η) * inv(@all(G) * dt) + 1.0)
    @inn(τxy) =
        @inn(τxy) +
        (
            -(@inn(τxy) - @inn(τxy_o)) * @av(η) / (@av(G) * dt) - @inn(τxy) +
            2.0 * @av(η) * @inn(εxy)
        ) * inv(θ_dτ + @av(η) * inv(@av(G) * dt) + 1.0)

    return nothing
end

# visco-elasto-plastic with GeoParams - with single phases
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
    T,
    args_η,
    rheology,
    dt,
    θ_dτ,
)
    #! format: off
    # convinience closure
    Base.@propagate_inbounds @inline gather(A) = _gather(A, i, j)
    Base.@propagate_inbounds @inline function av(T)
        (T[i, j] + T[i + 1, j] + T[i, j + 1] + T[i + 1, j + 1]) * 0.25
    end
    #! format: on

    @inbounds begin
        k = keys(args_η)
        v = getindex.(values(args_η), i, j)
        # numerics
        # dτ_r                = 1.0 / (θ_dτ + η[i, j] / (get_G(rheology[1]) * dt) + 1.0) # original
        dτ_r = 1.0 / (θ_dτ / η[i, j] + 1.0 / η_vep[i, j]) # equivalent to dτ_r = @. 1.0/(θ_dτ + η/(G*dt) + 1.0)
        # # Setup up input for GeoParams.jl
        args = (; zip(k, v)..., dt=dt, T=av(T), τII_old=0.0)
        εij_p = εxx[i, j] + 1e-25, εyy[i, j] + 1e-25, gather(εxyv) .+ 1e-25
        τij_p_o = τxx_o[i, j], τyy_o[i, j], gather(τxyv_o)
        phases = 1, 1, (1, 1, 1, 1) # there is only one phase...
        # update stress and effective viscosity
        τij, τII[i, j], ηᵢ = compute_τij(rheology, εij_p, args, τij_p_o, phases)
        τxx[i, j] += dτ_r * (-(τxx[i, j]) + τij[1]) / ηᵢ # NOTE: from GP Tij = 2*η_vep * εij
        τyy[i, j] += dτ_r * (-(τyy[i, j]) + τij[2]) / ηᵢ
        τxy[i, j] += dτ_r * (-(τxy[i, j]) + τij[3]) / ηᵢ
        η_vep[i, j] = ηᵢ
    end

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
    Base.@propagate_inbounds @inline gather(A) = _gather(A, i, j)
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

# single phase visco-elasto-plastic flow
@parallel_indices (i, j) function compute_τ_nonlinear!(
    τxx,
    τyy,
    τxy,
    τII,
    τxx_old,
    τyy_old,
    τxyv_old,
    εxx,
    εyy,
    εxyv,
    P,
    η,
    η_vep,
    λ,
    rheology,
    dt,
    θ_dτ,
)
    idx = i, j

    # numerics
    ηij = η[i, j]
    _Gdt = inv(get_G(rheology[1]) * dt)
    dτ_r = compute_dτ_r(θ_dτ, ηij, _Gdt)

    # get plastic paremeters (if any...)
    is_pl, C, sinϕ, η_reg = plastic_params(rheology[1])
    plastic_parameters = (; is_pl, C, sinϕ, η_reg)

    τ = τxx, τyy, τxy
    τ_old = τxx_old, τyy_old, τxyv_old
    ε = εxx, εyy, εxyv

    _compute_τ_nonlinear!(
        τ, τII, τ_old, ε, P, ηij, η_vep, λ, dτ_r, _Gdt, plastic_parameters, idx...
    )

    return nothing
end

# multi phase visco-elasto-plastic flow, where phases are defined in the cell center
@parallel_indices (i, j) function compute_τ_nonlinear!(
    τxx,
    τyy,
    τxy,
    τII,
    τxx_old,
    τyy_old,
    τxyv_old,
    εxx,
    εyy,
    εxyv,
    P,
    η,
    η_vep,
    λ,
    phase_center,
    rheology,
    dt,
    θ_dτ,
)
    idx = i, j

    # numerics
    ηij = @inbounds η[i, j]
    phase = @inbounds phase_center[i, j]
    _Gdt = inv(get_G(rheology, phase) * dt)
    dτ_r = compute_dτ_r(θ_dτ, ηij, _Gdt)

    # get plastic paremeters (if any...)
    is_pl, C, sinϕ, η_reg = plastic_params(rheology, phase)
    plastic_parameters = (; is_pl, C, sinϕ, η_reg)

    τ = τxx, τyy, τxy
    τ_old = τxx_old, τyy_old, τxyv_old
    ε = εxx, εyy, εxyv

    _compute_τ_nonlinear!(
        τ, τII, τ_old, ε, P, ηij, η_vep, λ, dτ_r, _Gdt, plastic_parameters, idx...
    )

    return nothing
end

# new stokes solver used for 2D plasticity

# module maxloc
#     using ParallelStencil
#     using ParallelStencil.FiniteDifferences2D
#     # using JustRelax
#     using LinearAlgebra
#     using CUDA
#     using Printf

#     export  compute_maxloc!
#     @parallel function compute_maxloc!(A::AbstractArray, B::AbstractArray)
#         @inn(A) = @maxloc(B)
#         return nothing
#     end
# end

module Maxloc_JR
    using ParallelStencil
    using ParallelStencil.FiniteDifferences2D
    import JustRelax: @idx
    using LinearAlgebra
    using CUDA
    using Printf
    function compute_maxloc!(B, A; window=(1, 1, 1))
        ni = size(A)
        width_x, width_y, width_z = window

        @parallel_indices (i, j) function _maxloc!(
            B::T, A::T
        ) where {T<:AbstractArray{<:Number,2}}
            B[i, j] = _maxloc_window_clamped(A, i, j, width_x, width_y)
            return nothing
        end

        @parallel_indices (i, j, k) function _maxloc!(
            B::T, A::T
        ) where {T<:AbstractArray{<:Number,3}}
            B[i, j, k] = _maxloc_window_clamped(A, i, j, k, width_x, width_y, width_z)
            return nothing
        end

        @parallel (@idx ni) _maxloc!(B, A)
    end

    @inline function _maxloc_window_clamped(A, I, J, width_x, width_y)
        nx, ny = size(A)
        I_range = (I - width_x):(I + width_x)
        J_range = (J - width_y):(J + width_y)
        x = -Inf
        for i in I_range
            ii = clamp(i, 1, nx)
            for j in J_range
                jj = clamp(j, 1, ny)
                Aij = A[ii, jj]
                if Aij > x
                    x = Aij
                end
            end
        end
        return x
    end

    @inline function _maxloc_window_clamped(A, I, J, K, width_x, width_y, width_z)
        nx, ny, nz = size(A)
        I_range = (I - width_x):(I + width_x)
        J_range = (J - width_y):(J + width_y)
        K_range = (K - width_z):(K + width_z)
        x = -Inf
        for i in I_range
            ii = clamp(i, 1, nx)
            for j in J_range
                jj = clamp(j, 1, ny)
                for k in K_range
                    kk = clamp(k, 1, nz)
                    Aijk = A[ii, jj, kk]
                    if Aijk > x
                        x = Aijk
                    end
                end
            end
        end
        return x
    end

end

function MTK_solve2!(
    stokes::StokesArrays{ViscoElastoPlastic,A,B,C,D,2},
    thermal::ThermalArrays,
    pt_stokes::PTStokesCoeffs,
    di::NTuple{2,T},
    flow_bcs,
    ϕ,
    ρg,
    η,
    η_vep,
    G,
    phase_v,
    phase_c,
    args,
    rheology::NTuple{N,MaterialParams},
    dt,
    igg::IGG;
    iterMax=10e3,
    nout=500,
    b_width=(4, 4, 0),
    verbose=true,
) where {A,B,C,D,N,T}

    # unpack
    _di = inv.(di)
    ϵ, r, θ_dτ, ηdτ = pt_stokes.ϵ, pt_stokes.r, pt_stokes.θ_dτ, pt_stokes.ηdτ
    ni = nx, ny = size(stokes.P)
    P_old = deepcopy(stokes.P)
    z = LinRange(di[2] * 0.5, 1.0 - di[2] * 0.5, ny)
    # ~preconditioner
    ητ = deepcopy(η)
    # @hide_communication b_width begin # communication/computation overlap
        @parallel Main.Maxloc_JR.Maxloc_JR.compute_maxloc!(ητ, η)
        update_halo!(ητ)
    # end

    λ = @zeros(ni...)

    # errors
    err = 2 * ϵ
    iter = 0
    err_evo1 = Float64[]
    err_evo2 = Float64[]
    norm_Rx = Float64[]
    norm_Ry = Float64[]
    norm_∇V = Float64[]

    # solver loop
    wtime0 = 0.0
    while iter < 2 || (err > ϵ && iter ≤ iterMax)
        wtime0 += @elapsed begin
            @parallel (@idx ni) compute_∇V!(stokes.∇V, stokes.V.Vx, stokes.V.Vy, _di...)

            @parallel (@idx ni) compute_P!(
                stokes.P, P_old, stokes.R.RP, stokes.∇V, η, rheology, phase_c, dt, r, θ_dτ
            )
            @parallel (@idx ni) compute_strain_rate!(
                @tuple(stokes.ε)..., stokes.∇V, @tuple(stokes.V)..., _di...
            )
            @parallel (@idx ni) compute_ρg!(ρg[2], ϕ, rheology, (T=thermal.T, P=stokes.P))

            ν = 0.05
            @parallel (@idx ni) compute_viscosity!(
                η, ν, @strain(stokes)..., args, tupleize(rheology)
            )
            Main.Maxloc_JR.Maxloc_JR.compute_maxloc!(ητ, η)
            update_halo!(ητ)


            @parallel (@idx ni) compute_τ_nonlinear!(
                @tensor_center(stokes.τ)...,
                stokes.τ.II,
                @tensor(stokes.τ_o)...,
                @strain(stokes)...,
                stokes.P,
                η,
                η_vep,
                λ,
                phase_c,
                tupleize(rheology), # needs to be a tuple
                dt,
                θ_dτ,
            )

            # @parallel (@idx ni) compute_τ_new!(
            #     stokes.τ.xx,
            #     stokes.τ.yy,
            #     stokes.τ.xy_c,
            #     stokes.τ.II,
            #     stokes.τ_o.xx,
            #     stokes.τ_o.yy,
            #     stokes.τ_o.xy,
            #     stokes.ε.xx,
            #     stokes.ε.yy,
            #     stokes.ε.xy,
            #     stokes.P,
            #     η,
            #     η_vep,
            #     rheology, # needs to be a tuple
            #     phase_c,
            #     dt,
            #     θ_dτ,
            #     λ
            # )


            @parallel center2vertex!(stokes.τ.xy, stokes.τ.xy_c)
            @hide_communication b_width begin # communication/computation overlap
                @parallel compute_V!(
                    stokes.V.Vx,
                    stokes.V.Vy,
                    stokes.P,
                    stokes.τ.xx,
                    stokes.τ.yy,
                    stokes.τ.xy,
                    ηdτ,
                    ρg[1],
                    ρg[2],
                    ητ,
                    _di...,
                )
                update_halo!(stokes.V.Vx, stokes.V.Vy)
            end
            # apply boundary conditions boundary conditions
            # apply_free_slip!(freeslip, stokes.V.Vx, stokes.V.Vy)
            flow_bcs!(stokes, flow_bcs, di)
        end

        iter += 1
        if iter % nout == 0 && iter > 1
            @parallel (@idx ni) compute_Res!(
                stokes.R.Rx,
                stokes.R.Ry,
                stokes.P,
                stokes.τ.xx,
                stokes.τ.yy,
                stokes.τ.xy,
                ρg[1],
                ρg[2],
                _di...,
            )
            errs = maximum.((abs.(stokes.R.Rx), abs.(stokes.R.Ry), abs.(stokes.R.RP)))
            push!(norm_Rx, errs[1])
            push!(norm_Ry, errs[2])
            push!(norm_∇V, errs[3])
            err = maximum(errs)
            push!(err_evo1, err)
            push!(err_evo2, iter)
            if igg.me == 0 && (verbose || iter == iterMax)
                @printf(
                    "Total steps = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_∇V=%1.3e] \n",
                    iter,
                    err,
                    norm_Rx[end],
                    norm_Ry[end],
                    norm_∇V[end]
                )
            end
            isnan(err) && error("NaN(s)")
        end

        if igg.me == 0 && err ≤ ϵ
            println("Pseudo-transient iterations converged in $iter iterations")
        end
    end

    if -Inf < dt < Inf
        update_τ_o!(stokes)
        @parallel (@idx ni) rotate_stress!(@tuple(stokes.V), @tuple(stokes.τ_o), _di, dt)
    end

    return (
        iter=iter,
        err_evo1=err_evo1,
        err_evo2=err_evo2,
        norm_Rx=norm_Rx,
        norm_Ry=norm_Ry,
        norm_∇V=norm_∇V,
    )
end

function MTK_solve!(
    stokes::StokesArrays{ViscoElastoPlastic,A,B,C,D,2},
    pt_stokes::PTStokesCoeffs,
    di::NTuple{2,T},
    flow_bcs,
    ρg,
    η,
    η_vep,
    phase_c,
    args,
    rheology::NTuple{N,MaterialParams},
    dt,
    igg::IGG;
    iterMax=10e3,
    nout=500,
    b_width=(4, 4, 0),
    verbose=true,
) where {A,B,C,D,N,T}

    # unpack
    _di = inv.(di)
    ϵ, r, θ_dτ, ηdτ = pt_stokes.ϵ, pt_stokes.r, pt_stokes.θ_dτ, pt_stokes.ηdτ
    ni = nx, ny = size(stokes.P)
    P_old = deepcopy(stokes.P)
    z = LinRange(di[2] * 0.5, 1.0 - di[2] * 0.5, ny)
    # ~preconditioner
    ητ = deepcopy(η)
    # @hide_communication b_width begin # communication/computation overlap
    JustRelax.compute_maxloc!(ητ, η)
        update_halo!(ητ)
    # end

    Kb = get_Kb(rheology[1])
    λ = @zeros(ni...)

    # errors
    err = 2 * ϵ
    iter = 0
    err_evo1 = Float64[]
    err_evo2 = Float64[]
    norm_Rx = Float64[]
    norm_Ry = Float64[]
    norm_∇V = Float64[]

    # solver loop
    wtime0 = 0.0
    while iter < 2 || (err > ϵ && iter ≤ iterMax)
        wtime0 += @elapsed begin
            @parallel (@idx ni) compute_∇V!(stokes.∇V, @velocity(stokes)..., _di...)
            @parallel (@idx ni) compute_P!(  
                stokes.P, stokes.P0, stokes.R.RP, stokes.∇V, η, rheology, phase_c, dt, r, θ_dτ 
            )  # switched to stokes.P0 rather than a deepcopy of stokes.P
            # display(heatmap(stokes.∇V, title="∇Vx $iter"))
            # @parallel compute_P!(
            #     stokes.P, stokes.P0, stokes.R.RP, stokes.∇V, η, Kb, dt, r, θ_dτ
            # )
            # display(heatmap(stokes.P, title="P $iter"))
            @parallel (@idx ni .+ 1) compute_strain_rate!(
                @strain(stokes)..., stokes.∇V, @velocity(stokes)..., _di...
            )
            # display(heatmap(stokes.ε.xy, title="εxy $iter"))
            # @parallel (@idx ni) compute_ρg2!(ρg[2], args.ϕ, rheology, (T=args.T, P=args.P))
            @parallel (@idx ni) compute_ρg_phase!(ρg[2], phase_c, rheology, (T=args.T, P=args.P))

            # # @parallel (@idx ni) JustRelax.compute_ρg!(ρg[2], rheology[i], (T=args.T, P=args.P))

            @parallel (@idx ni) computeViscosity!(η, args.ϕ, args.S, args.mfac, args.η_f, args.η_s) # viscosity calculation 1. based on melt fraction AND then strain rate 
            # ν = 0.05
            # @parallel (@idx ni) compute_viscosity_MTK!(
            #     η, ν, @strain(stokes)..., args, tupleize(rheology), phase_c
            # )
            # JustRelax.compute_maxloc!(ητ, η)
            # update_halo!(ητ)
            # # display(heatmap(η, title="η $iter"))

            @parallel (@idx ni) compute_τ_nonlinear!(
                @tensor_center(stokes.τ)...,
                stokes.τ.II,
                @tensor(stokes.τ_o)...,
                @strain(stokes)...,
                stokes.P,
                η,
                η_vep,
                λ,
                phase_c,
                tupleize(rheology), # needs to be a tuple
                dt,
                θ_dτ,
            )
            # display(heatmap(stokes.τ.xy, title="τxy $iter"))
            @parallel JustRelax.center2vertex!(stokes.τ.xy, stokes.τ.xy_c)

            @hide_communication b_width begin # communication/computation overlap
                @parallel compute_V!(
                    @velocity(stokes)...,
                    stokes.P,
                    @stress(stokes)...,
                    ηdτ,
                    ρg...,
                    ητ,
                    _di...,
                )
                update_halo!(stokes.V.Vx, stokes.V.Vy)
            end
            # display(heatmap(stokes.V.Vy, title="Vy $iter"))
            # apply boundary conditions boundary conditions
            # apply_free_slip!(freeslip, stokes.V.Vx, stokes.V.Vy)
            flow_bcs!(stokes, flow_bcs, di)
        end

        iter += 1
        if iter % nout == 0 && iter > 1
            @parallel (@idx ni) compute_Res!(
                stokes.R.Rx, stokes.R.Ry, stokes.P, @stress(stokes)..., ρg..., _di...
            )
            errs = maximum.((abs.(stokes.R.Rx), abs.(stokes.R.Ry), abs.(stokes.R.RP)))
            push!(norm_Rx, errs[1])
            push!(norm_Ry, errs[2])
            push!(norm_∇V, errs[3])
            err = maximum(errs)
            push!(err_evo1, err)
            push!(err_evo2, iter)
            if igg.me == 0 && (verbose || iter == iterMax)
                @printf(
                    "Total steps = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_∇V=%1.3e] \n",
                    iter,
                    err,
                    norm_Rx[end],
                    norm_Ry[end],
                    norm_∇V[end]
                )
            end
            isnan(err) && error("NaN(s)")
        end

        if igg.me == 0 && err ≤ ϵ
            println("Pseudo-transient iterations converged in $iter iterations")
        end
    end

    if -Inf < dt < Inf
        update_τ_o!(stokes)
        @parallel (@idx ni) rotate_stress!(@tuple(stokes.V), @tuple(stokes.τ_o), _di, dt)
    end

    return (
        iter=iter,
        err_evo1=err_evo1,
        err_evo2=err_evo2,
        norm_Rx=norm_Rx,
        norm_Ry=norm_Ry,
        norm_∇V=norm_∇V,
    )
end

function circular_perturbation!(T, δT, xc, yc, r, xvi)

    @parallel_indices (i, j) function _circular_perturbation!(T, δT, xc, yc, r, x, y)
        @inbounds if (((x[i]-xc ))^2 + ((y[j] - yc))^2) ≤ r^2
            T[i, j] *= δT/100 + 1
        end
        return nothing
    end

    @parallel _circular_perturbation!(T, δT, xc, yc, r, xvi...)
end

function random_perturbation!(T, δT, xbox, ybox, xvi)

    @parallel_indices (i, j) function _random_perturbation!(T, δT, xbox, ybox, x, y)
        @inbounds if (xbox[1] ≤ x[i] ≤ xbox[2]) && (abs(ybox[1]) ≤ abs(y[j]) ≤ abs(ybox[2]))
            δTi = δT * (rand() -  0.5) # random perturbation within ±δT [%]
            T[i, j] *= δTi/100 + 1
        end
        return nothing
    end
    
    @parallel (@idx size(T)) _random_perturbation!(T, δT, xbox, ybox, xvi...)
end

function circular_anomaly!(T, anomaly, phases, xc, yc, r, xvi)

    @parallel_indices (i, j) function _circular_anomaly!(T, anomaly, phases, xc, yc, r, x, y)
        @inbounds if (((x[i].-xc ))^2 + ((y[j] .+ yc))^2) ≤ r^2
            T[i, j] = anomaly
            phases[i, j] = 2
            
        end
        return nothing
    end

    @parallel _circular_anomaly!(T, anomaly, phases, xc, yc, r, xvi...)

end

function circular_anomaly_center!(phases, xc, yc, r, xvi)

    @parallel_indices (i, j) function _circular_anomaly_center!(phases, xc, yc, r, x, y)
        @inbounds if (((x[i].-xc ))^2 + ((y[j] .+ yc))^2) ≤ r^2
            phases[i, j] = 2
            
        end
        return nothing
    end

    @parallel _circular_anomaly_center!(phases, xc, yc, r, xvi...)

end


function ρg_solver!(
    stokes::StokesArrays{ViscoElastic,A,B,C,D,2},
    pt_stokes::PTStokesCoeffs,
    di::NTuple{2,T},
    flow_bcs,
    ρg,
    η,
    η_vep,
    phase_v,
    phase_c,
    args,
    rheology::NTuple{N,MaterialParams},
    dt,
    igg::IGG;
    iterMax=10e3,
    nout=500,
    b_width=(4, 4, 1),
    verbose=true,
) where {A,B,C,D,N,T}

    # unpack
    _di = inv.(di)
    (; ϵ, r, θ_dτ, ηdτ) = pt_stokes
    ni = size(stokes.P)
    # ~preconditioner
    ητ = deepcopy(η)
    # @hide_communication b_width begin # communication/computation overlap
    compute_maxloc!(ητ, η)
    update_halo!(ητ)
    # end

    # errors
    err = 2 * ϵ
    iter = 0
    err_evo1 = Float64[]
    err_evo2 = Float64[]
    norm_Rx = Float64[]
    norm_Ry = Float64[]
    norm_∇V = Float64[]

    # solver loop
    wtime0 = 0.0
    while iter < 2 || (err > ϵ && iter ≤ iterMax)
        wtime0 += @elapsed begin
            @parallel (@idx ni) compute_∇V!(stokes.∇V, @velocity(stokes)..., _di...)
            @parallel (@idx ni) compute_P!(
                stokes.P,
                stokes.P0,
                stokes.R.RP,
                stokes.∇V,
                η,
                rheology,
                phase_c,
                dt,
                r,
                θ_dτ,
            )
            @parallel (@idx ni .+ 1) compute_strain_rate!(
                @strain(stokes)..., stokes.∇V, @velocity(stokes)..., _di...
            )
            @parallel (@idx ni) compute_ρg!(ρg[2], args.ϕ, rheology, (T=args.T, P=args.P))
            @parallel (@idx ni) compute_τ_gp!(
                @tensor_center(stokes.τ)...,
                stokes.τ.II,
                @tensor(stokes.τ_o)...,
                @strain(stokes)...,
                η,
                η_vep,
                thermal.T,
                phase_v,
                phase_c,
                args_η,
                tupleize(rheology), # needs to be a tuple
                dt,
                θ_dτ,
            )
            @parallel center2vertex!(stokes.τ.xy, stokes.τ.xy_c)
            @hide_communication b_width begin # communication/computation overlap
                @parallel compute_V!(
                    @velocity(stokes)...,
                    stokes.P,
                    @stress(stokes)...,
                    ηdτ,
                    ρg...,
                    ητ,
                    _di...,
                )
                update_halo!(stokes.V.Vx, stokes.V.Vy)
            end
            # apply boundary conditions boundary conditions
            flow_bcs!(stokes, flow_bcs, di)
        end

        iter += 1
        if iter % nout == 0 && iter > 1
            @parallel (@idx ni) compute_Res!(
                stokes.R.Rx, stokes.R.Ry, stokes.P, @stress(stokes)..., ρg..., _di...
            )
            errs = maximum.((abs.(stokes.R.Rx), abs.(stokes.R.Ry), abs.(stokes.R.RP)))
            push!(norm_Rx, errs[1])
            push!(norm_Ry, errs[2])
            push!(norm_∇V, errs[3])
            err = maximum(errs)
            push!(err_evo1, err)
            push!(err_evo2, iter)
            if igg.me == 0 && (verbose || iter == iterMax)
                @printf(
                    "Total steps = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_∇V=%1.3e] \n",
                    iter,
                    err,
                    norm_Rx[end],
                    norm_Ry[end],
                    norm_∇V[end]
                )
            end
            isnan(err) && error("NaN(s)")
        end

        if igg.me == 0 && err ≤ ϵ
            println("Pseudo-transient iterations converged in $iter iterations")
        end
    end

    if !isinf(dt) # if dt is inf, then we are in the non-elastic case 
        update_τ_o!(stokes)
        @parallel (@idx ni) rotate_stress!(@velocity(stokes), @tensor(stokes.τ_o), _di, dt)
    end

    return (
        iter=iter,
        err_evo1=err_evo1,
        err_evo2=err_evo2,
        norm_Rx=norm_Rx,
        norm_Ry=norm_Ry,
        norm_∇V=norm_∇V,
    )
end

function MTK_solve3!(
    stokes::StokesArrays{ViscoElastic,A,B,C,D,2},
    pt_stokes::PTStokesCoeffs,
    di::NTuple{2,T},
    flow_bcs,
    ρg,
    η,
    η_vep,
    args,
    rheology::NTuple{N,MaterialParams},
    dt,
    igg::IGG;
    iterMax=10e3,
    nout=500,
    b_width=(4, 4, 0),
    verbose=true,
) where {A,B,C,D,N,T}

    # unpack
    _di = inv.(di)
    (; ϵ, r, θ_dτ, ηdτ) = pt_stokes
    ni = size(stokes.P)

    # ~preconditioner
    ητ = deepcopy(η)
    # @hide_communication b_width begin # communication/computation overlap
    JustRelax.compute_maxloc!(ητ, η)
    update_halo!(ητ)
    # end

    Kb = get_Kb(rheology[1])

    # errors
    err = 2 * ϵ
    iter = 0
    err_evo1 = Float64[]
    err_evo2 = Float64[]
    norm_Rx = Float64[]
    norm_Ry = Float64[]
    norm_∇V = Float64[]

    # solver loop
    wtime0 = 0.0
    nonlinear = true
    λ = @zeros(ni...)
    while iter < 2 || (err > ϵ && iter ≤ iterMax)
        wtime0 += @elapsed begin
            @parallel (@idx ni) compute_∇V!(stokes.∇V, @velocity(stokes)..., _di...)
            @parallel compute_P!(
                stokes.P, stokes.P0, stokes.R.RP, stokes.∇V, η, Kb, dt, r, θ_dτ
            )


            @parallel (@idx ni .+ 1) compute_strain_rate!(
                @strain(stokes)..., stokes.∇V, @velocity(stokes)..., _di...
            )

            ν = 0.05
            @parallel (@idx ni) compute_viscosity!(
                η, ν, @strain(stokes)..., args, tupleize(rheology)
            )
            compute_maxloc!(ητ, η)
            update_halo!(ητ)

            @parallel (@idx ni) compute_τ_nonlinear!(
                @tensor_center(stokes.τ)...,
                stokes.τ.II,
                @tensor(stokes.τ_o)...,
                @strain(stokes)...,
                stokes.P,
                η,
                η_vep,
                λ,
                tupleize(rheology), # needs to be a tuple
                dt,
                θ_dτ,
            )

            @parallel center2vertex!(stokes.τ.xy, stokes.τ.xy_c)
            @hide_communication b_width begin # communication/computation overlap
                @parallel compute_V!(
                    @velocity(stokes)...,
                    stokes.P,
                    @stress(stokes)...,
                    ηdτ,
                    ρg...,
                    ητ,
                    _di...,
                )
                update_halo!(stokes.V.Vx, stokes.V.Vy)
            end
            # apply boundary conditions boundary conditions
            flow_bcs!(stokes, flow_bcs, di)
        end

        iter += 1
        if iter % nout == 0 && iter > 1
            @parallel (@idx ni) compute_Res!(
                stokes.R.Rx, stokes.R.Ry, stokes.P, @stress(stokes)..., ρg..., _di...
            )
            errs = maximum.((abs.(stokes.R.Rx), abs.(stokes.R.Ry), abs.(stokes.R.RP)))
            push!(norm_Rx, errs[1])
            push!(norm_Ry, errs[2])
            push!(norm_∇V, errs[3])
            err = maximum(errs)
            push!(err_evo1, err)
            push!(err_evo2, iter)
            if igg.me == 0 && ((verbose && err > ϵ) || iter == iterMax)
                @printf(
                    "Total steps = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_∇V=%1.3e] \n",
                    iter,
                    err,
                    norm_Rx[end],
                    norm_Ry[end],
                    norm_∇V[end]
                )
            end
            isnan(err) && error("NaN(s)")
        end

        if igg.me == 0 && err ≤ ϵ
            println("Pseudo-transient iterations converged in $iter iterations")
        end
    end

    if !isinf(dt) # if dt is inf, then we are in the non-elastic case
        update_τ_o!(stokes)
        @parallel (@idx ni) rotate_stress!(@velocity(stokes), @tensor(stokes.τ_o), _di, dt)
    end

    return (
        iter=iter,
        err_evo1=err_evo1,
        err_evo2=err_evo2,
        norm_Rx=norm_Rx,
        norm_Ry=norm_Ry,
        norm_∇V=norm_∇V,
    )
end



@inline function local_viscosity_args(args, I::Vararg{Integer,N}) where {N}
    v = getindex.(values(args), I...)
    local_args = (; zip(keys(args), v)..., dt=args.dt, τII_old=0.0)
    return local_args
end

@inline function local_args(args, I::Vararg{Integer,N}) where {N}
    v = getindex.(values(args), I...)
    local_args = (; zip(keys(args), v)...)
    return local_args
end

#others
@inline function _gather(A, i, j)
    return A[i, j], A[i + 1, j], A[i, j + 1], A[i + 1, j + 1]
end


# 2D kernel
@parallel_indices (i, j) function compute_viscosity_MTK!(η, ν, εxx, εyy, εxyv, args, rheology, phases)

    # convinience closure
    @inline gather(A) = _gather(A, i, j)

    @inbounds begin
        # we need strain rate not to be zero, otherwise we get NaNs
        εII_0 = (εxx[i, j] == εyy[i, j] == 0) * 1e-15

        # argument fields at local index
    
        args_ij = local_args(args, i, j)

        # compute second invariant of strain rate tensor
        εij = εII_0 + εxx[i, j], -εII_0 + εyy[i, j], gather(εxyv)
        εII = second_invariant(εij...)

        # compute and update stress viscosity
        ηi = compute_viscosity_εII(rheology,  phases[i, j], εII, args_ij)
        η[i, j] = continuation_log(ηi, η[i, j], ν)
    end

    return nothing
end

# @parallel (@idx ni) compute_viscosity_MTK!(
#     η, 0, @strain(stokes)..., args, tupleize(MatParam), phase_c
# )