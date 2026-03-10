# thread-benchmark.jl
#
# Benchmarks the bellman operator with however many threads Julia was started with.
# Run with different thread counts to compare performance:
#
#   julia --threads 1    test/thread-benchmark.jl
#   julia --threads 2    test/thread-benchmark.jl
#   julia --threads 4    test/thread-benchmark.jl
#   julia --threads auto test/thread-benchmark.jl
#
# You must restart Julia to change the thread count.

using ValueFunctions
using GriddedFunctions

# ---------------------------------------------------------------------------
# Problem: capital accumulation (same model as in runtests.jl)
# ---------------------------------------------------------------------------

struct BenchState <: State
    k::Float64
end

struct BenchAction <: Action
    kp::Float64
end

struct BenchParams
    β::Float64
    α::Float64
    A::Float64
end

function ValueFunctions.nextstate(a::BenchAction, ::BenchState, ::Environment)
    BenchState(a.kp)
end

function ValueFunctions.actionvalue(
        a::BenchAction, s::BenchState, V_itp::GFInterpolation, e::Environment)
    β, α, A = e.params.β, e.params.α, e.params.A
    c = A * s.k^α - a.kp
    c > 0 || return -Inf
    k_min = minimum(gridaxes(stategrid(e), 1))
    k_max = maximum(gridaxes(stategrid(e), 1))
    (k_min <= a.kp <= k_max) || return -Inf
    log(c) + β * V_itp(BenchState(a.kp))
end

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

const N_STATES  = 500
const N_ACTIONS = 300
const N_ITERS   = 20   # bellman iterations to time

params = BenchParams(0.95, 0.36, 1.0)
sg  = Grid(BenchState,  LinearAxis(range(0.05, 1.5;  length = N_STATES)))
ag  = Grid(BenchAction, LinearAxis(range(0.001, 0.8; length = N_ACTIONS)))
env = Environment(params, ag, sg)
V0  = GriddedFunction(Float64, sg, _ -> 0.0)

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

println("=" ^ 55)
println("Bellman operator — thread benchmark")
println("  Julia version : $(VERSION)")
println("  Threads       : $(Threads.nthreads())")
println("  State points  : $N_STATES")
println("  Action points : $N_ACTIONS")
println("=" ^ 55)

# Warmup: run one iteration to trigger compilation before timing.
print("\nWarming up (compiling)... ")
bellman(V0, env)
println("done.\n")

# Timed run: N_ITERS full bellman iterations.
println("Timing $N_ITERS bellman iterations...")
stats = @timed begin
    Vn = V0
    for _ in 1:N_ITERS
        global Vn = bellman(Vn, env)
    end
    Vn
end

total_s   = stats.time
avg_ms    = total_s / N_ITERS * 1_000
total_mem = stats.bytes
avg_mem   = round(Int, total_mem / N_ITERS)

println()
println("Results")
println("-" ^ 40)
println("  Total time      : $(round(total_s;  digits = 3)) s")
println("  Avg / iteration : $(round(avg_ms;   digits = 2)) ms")
println("  Total allocs    : $(Base.format_bytes(total_mem))")
println("  Avg  / iteration: $(Base.format_bytes(avg_mem))")
println()
