using Test
using ValueFunctions
using GriddedFunctions

# ---------------------------------------------------------------------------
# Type definitions
# (Julia struct definitions are always module-scoped, so they live here.)
# ---------------------------------------------------------------------------

struct SimpleState <: State
    k::Float64
end

struct SimpleAction <: Action
    kp::Float64
end

struct MultiState <: State
    k::Float64
    theta::Float64
end

struct MultiAction <: Action
    q::Float64
    i::Float64
end

struct MixedFieldAction <: Action
    n::Int
    x::Float64
end

# --- Capital accumulation model (used for the bellman testset) ---

struct CapState <: State
    k::Float64
end

struct CapAction <: Action
    kp::Float64
end

struct CapParams
    β::Float64
    α::Float64
    A::Float64
end

function ValueFunctions.nextstate(a::CapAction, ::CapState, ::Environment)
    CapState(a.kp)
end

function ValueFunctions.actionvalue(
        a::CapAction, s::CapState, V_itp::GFInterpolation, e::Environment)
    β, α, A = e.params.β, e.params.α, e.params.A
    c = A * s.k^α - a.kp
    c > 0 || return -Inf
    (minimum(gridaxes(stategrid(e), 1)) <= a.kp <= maximum(gridaxes(stategrid(e), 1))) || return -Inf
    log(c) + β * V_itp(CapState(a.kp))
end

# ---------------------------------------------------------------------------

@testset "ValueFunctions" begin

    @testset "Variable interface — single-field State" begin
        s = SimpleState(1.5)

        @test SimpleState <: State
        @test SimpleState <: ValueFunctions.Variable

        @test length(s) == 1
        @test s[1]     == 1.5
        @test s[begin] == 1.5
        @test s[end]   == 1.5
        @test s[:k]    == 1.5

        @test collect(s) == [1.5]
        @test eltype(SimpleState) == Float64
    end

    @testset "Variable interface — multi-field Action" begin
        a = MultiAction(0.5, 0.3)

        @test MultiAction <: Action
        @test MultiAction <: ValueFunctions.Variable

        @test length(a) == 2
        @test a[1]  == 0.5
        @test a[2]  == 0.3
        @test a[:q] == 0.5
        @test a[:i] == 0.3

        @test collect(a) == [0.5, 0.3]
        @test eltype(MultiAction) == Float64
    end

    @testset "Variable interface — mixed-type eltype" begin
        @test eltype(MixedFieldAction) == Float64
    end

    @testset "Grid construction for State" begin
        sg = Grid(SimpleState, LinearAxis(range(0.1, 2.0; length=20)))

        @test eltype(sg) == SimpleState
        @test size(sg)   == (20,)
        @test length(sg) == 20

        @test sg[1]   isa SimpleState
        @test sg[1]   == SimpleState(0.1)
        @test sg[end] == SimpleState(2.0)

        states = collect(sg)
        @test all(s isa SimpleState for s in states)
        @test first(states) == SimpleState(0.1)
        @test last(states)  == SimpleState(2.0)
    end

    @testset "Grid construction for Action" begin
        ag = Grid(SimpleAction, LinearAxis(range(0.01, 0.8; length=30)))

        @test eltype(ag) == SimpleAction
        @test size(ag)   == (30,)
        @test ag[1]   isa SimpleAction
        @test ag[1]   == SimpleAction(0.01)
        @test ag[end] == SimpleAction(0.8)

        actions = collect(ag)
        @test all(a isa SimpleAction for a in actions)
    end

    @testset "Grid construction — 2D multi-field Action" begin
        ag2 = Grid(
            MultiAction,
            LinearAxis(range(0.0, 1.0; length=10)),
            LinearAxis(range(0.0, 0.5; length=8))
        )

        ag3 = Grid(
            MultiAction,
            i = LinearAxis(range(0.0, 0.5; length=8)),
            q = LinearAxis(range(0.0, 1.0; length=10))
        )

        @test ag2 == ag3

        @test eltype(ag2)    == MultiAction
        @test size(ag2)      == (10, 8)
        @test ag2[1, 1]      == MultiAction(0.0, 0.0)
        @test ag2[end, end]  == MultiAction(1.0, 0.5)
    end

    @testset "GriddedFunction over State grid — construction and indexing" begin
        sg = Grid(SimpleState, LinearAxis(range(0.1, 2.0; length=50)))

        # constant initialisation via f(k) -> 0.0
        V0 = GriddedFunction(Float64, sg, _ -> 0.0)
        @test size(V0) == (50,)
        @test all(==(0.0), V0)

        # function of the continuous coordinate
        V_log = GriddedFunction(Float64, sg, k -> log(k.k))
        @test V_log[1]   ≈ log(0.1)
        @test V_log[end] ≈ log(2.0)

        # in-place argmap! over grid points
        argmap!(state -> state.k^2, V0)
        @test V0[1]   ≈ 0.1^2
        @test V0[end] ≈ 2.0^2
    end

    @testset "GriddedFunction over State grid — points" begin
        sg    = Grid(SimpleState, LinearAxis(range(0.1, 2.0; length=10)))
        V_log = GriddedFunction(Float64, sg, k -> log(k.k))

        ps = collect(points(V_log))
        @test length(ps) == 10
        @test all(p isa Pair for p in ps)

        # keys are grid-point tuples; values match the defining function
        for p in ps
            @test p.first  isa SimpleState
            @test p.second ≈ log(only(p.first))
        end
    end

    @testset "Interpolation over State grid" begin
        sg    = Grid(SimpleState, LinearAxis(range(0.1, 2.0; length=200)))
        V_log = GriddedFunction(Float64, sg, k -> log(k.k))
        V_itp = interpolate(V_log)

        @test V_itp isa GFInterpolation

        # callable with a State instance (TX dispatch path)
        @test V_itp(SimpleState(1.0)) ≈ log(1.0) atol=1e-4
        @test V_itp(SimpleState(0.5)) ≈ log(0.5) atol=1e-3
        @test V_itp(SimpleState(1.5)) ≈ log(1.5) atol=1e-3

        # callable with raw coordinate (Vararg dispatch path)
        @test V_itp(1.0) ≈ log(1.0) atol=1e-4
    end

    @testset "Environment construction" begin
        sg  = Grid(SimpleState,  LinearAxis(range(0.1, 2.0; length=20)))
        ag  = Grid(SimpleAction, LinearAxis(range(0.01, 0.8; length=20)))
        env = Environment(nothing, ag, sg)

        @test env.actiongrid === ag
        @test env.stategrid  === sg
        @test env.params     === nothing
    end

    @testset "Bellman operator — capital accumulation" begin
        β, α, A = 0.95, 0.36, 1.0
        params   = CapParams(β, α, A)

        sg  = Grid(CapState,  LinearAxis(range(0.05, 1.5;  length=100)))
        ag  = Grid(CapAction, LinearAxis(range(0.001, 0.8; length=100)))
        env = Environment(params, ag, sg)

        # zero initial value function
        Vn = GriddedFunction(Float64, sg, _ -> 0.0)

        tol, max_iter = 1e-6, 3000
        for _ in 1:max_iter
            Vn_new, policy = bellman(Vn, env)
            dist           = maximum(abs(a - b) for (a, b) in zip(Vn_new, Vn))
            Vn             = Vn_new
            dist < tol && break
        end

        # Analytical solution: V*(k) = E + F·ln(k)
        F_true = α / (1 - α * β)
        βF     = β * F_true
        E_true = (1 / (1 - β)) * (log(1 / (1 + βF)) + βF * log(βF * A / (1 + βF)) + log(A))

        for p in points(Vn)
            k     = only(p.first)
            v_ana = E_true + F_true * log(k)
            @test abs(p.second - v_ana) < 0.05
        end
    end

end
