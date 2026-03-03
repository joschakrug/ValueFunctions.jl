using Revise
using ValueFunctionIteration
using GriddedFunctions


@kwdef struct PlantState <: State
    a::Float64
    θ::Float64
end

@kwdef struct PlantAction <: Action
    q::Float64
    b::Float64
    κ::Float64
end

@kwdef struct World
    σ::Float64
    w::Float64
    f_b::Float64
    f_κ::Float64
    β::Float64
    η_l_lc::Float64
    η_l_fc::Float64
    η_m_lc::Float64
    η_m_fc::Float64
end

@kwdef struct EnvState
    A::Float64
    a_hat::Float64
    τ::Float64
    c_κ::Float64
end

struct PlantParams
    w::World
    es::EnvState
end

"""Revenue r(q; A, σ) = A^(1/σ) · q^((σ−1)/σ)."""
revenue(q, e::Environment) = let A = e.es.A, σ = e.w.σ
    A^(1/σ) * q^((σ - 1)/σ)
end

"""
Permit-market expenditure ψ(b; τ, f_b):
    τ·b + f_b   if b ≠ 0
    0           if b = 0
"""
ψ(b, e::Environment) = let τ = e.params.es.τ, f_b = e.params.w.f_b
    iszero(b) ? zero(τ) : τ * b + f_b
end

"""
Investment expenditure ω(κ; c_κ, f_κ):
    0             if κ = 0
    c_κ·κ + f_κ  if κ > 0
"""
ω(κ, e::Environment) = let c_κ = e.params.es.c_κ, f_κ = e.params.w.f_κ
   iszero(κ) ? zero(c_κ) : c_κ * κ + f_κ 
end

"""Labour intensity η_l(θ) = θ · η̄_l + (1−θ) · η̲_l."""
η_l(θ, e::Environment) = θ * e.w.η_l_lc + (1 - θ) * e.w.η_l_fc

"""Emissions intensity η_m(θ) = θ · η̲_m + (1−θ) · η̄_m."""
η_m(θ, e::Environment) = θ * e.w.η_m_lc + (1 - θ) * e.w.η_m_fc

function ValueFunctionIteration.update(s::PlantState, e::Environment)
    let w = e.params.w, es = e.params.es, a = s.a, a_hat = es.a_hat, q = act.q, θ = s.θ, κ = act.κ, b = act.b
        a_next = a + a_hat + b - η_m(θ, w) * q
        θ_next = min(θ + κ, 1)
        PlantState(a_next, θ_next)
    end
end

function ValueFunctionIteration.currentvalue(act::PlantAction, s::PlantState, V_next_itp, e::Environment)
    q, b, κ = act
    a, θ = s
    let β = e.params.w.β, s_next = update(s, act, e)
        a_next, θ_next = s_next

        if a_next < 0
            return -Inf
        else
            revenue(q, e) - η_l(θ, w) * q + ψ(b, e) - ω(κ, e) + β * V_next_itp(s_next)
        end
    end
end

# now, define a terminal value function (and a corresponding state grid), an
# action grid and a parametrised plant environment
# (potentially create a type GriddedBellmanEquation for this)

env = Environment(
    PlantParams(
        World(
            σ = 1.5,
            w = 1.,
            f_b = 0.,
            f_κ = 0.,
            β = 0.9,
            η_l_lc = 0.7,
            η_l_fc = 0.5,
            η_m_lc = 0.1,
            η_m_fc = 0.7
        ),
        EnvState(
            A = 1.,
            a_hat = 0.5,
            τ = 1.,
            c_κ = 0.
        )
    ),
    VariableGrid(
        PlantAction,
        (
            q = LinearAxis(range(0, 1, length = 100)),
            b = LinearAxis(range(0, 0.5, length = 100)),
            κ = DiscreteAxis([0.])
        )
    ),
    VariableGrid(
        PlantState,
        (
            a = LinearAxis(range(0, 5, length = 200)),
            θ = DiscreteAxis([0.])
        )
    )
)

V_terminal = ValueFunctionIteration.GriddedVariableFunction(
    Float64,
    env.stategrid,
    state -> iszero(state.θ) ? 1. : 0.
)

V_terminal[2, 1] = 2.

V_terminal[1, 1]