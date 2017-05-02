@test_throws DimensionMismatch stratifiedobs((X, rand(149)))
@test_throws DimensionMismatch stratifiedobs((X, rand(149)), obsdim=:last)

srand(1335)

ty = [:a, :a, :a, :a, :a, :a, :b, :b, :b, :b]
if Int === Int64
    @test splitobs(labelmap(ty), at = 0.5) == ([1,2,3,7,8],[4,5,6,9,10])
else
    @test splitobs(labelmap(ty), at = 0.5) == ([7,8,1,2,3],[9,10,4,5,6])
end

@testset "Type Stability" begin
    for var in vars
        srand(1335)
        @test_throws ArgumentError stratifiedobs(var, 0.)
        @test_throws ArgumentError stratifiedobs(var, 1.)
        @test_throws ArgumentError stratifiedobs(var, (0.2,0.0))
        @test_throws ArgumentError stratifiedobs(var, (0.2,0.8))
        @test_throws MethodError stratifiedobs(var, 0.5, ObsDim.Undefined())
        @test typeof(@inferred(stratifiedobs(var))) <: NTuple{2}
        @test eltype(@inferred(stratifiedobs(var))) <: SubArray
        @test typeof(@inferred(stratifiedobs(var, 0.5))) <: NTuple{2}
        @test typeof(@inferred(stratifiedobs(var, 0.5, true))) <: NTuple{2}
        @test typeof(@inferred(stratifiedobs(var, (0.5,0.2)))) <: NTuple{3}
        @test typeof(@inferred(stratifiedobs(var, 0.5, true, ObsDim.Last()))) <: NTuple{2}
        @test typeof(@inferred(stratifiedobs(var, 0.5, true, ObsDim.First()))) <: NTuple{2}
        @test eltype(@inferred(stratifiedobs(var, 0.5, true, ObsDim.First()))) <: SubArray
    end
    for tup in tuples
        srand(1335)
        @test_throws ArgumentError stratifiedobs(tup, 0.)
        @test_throws ArgumentError stratifiedobs(tup, 1.)
        @test_throws ArgumentError stratifiedobs(tup, (0.2,0.0))
        @test_throws ArgumentError stratifiedobs(tup, (0.2,0.8))
        @test_throws MethodError stratifiedobs(tup, 0.5, ObsDim.Undefined())
        @test typeof(@inferred(stratifiedobs(tup))) <: NTuple{2}
        @test eltype(@inferred(stratifiedobs(tup))) <: Tuple
        @test typeof(@inferred(stratifiedobs(tup, 0.5))) <: NTuple{2}
        @test typeof(@inferred(stratifiedobs(tup, 0.5, true))) <: NTuple{2}
        @test typeof(@inferred(stratifiedobs(tup, (0.5,0.2)))) <: NTuple{3}
        @test typeof(@inferred(stratifiedobs(tup, 0.5, true, ObsDim.Last()))) <: NTuple{2}
    end
end

println("<HEARTBEAT>")

srand(1335)
@testset "SparseArray" begin
    @test nobs.(stratifiedobs(round, ys)) == (105,45)
    @test nobs.(stratifiedobs(round, ys, p=0.7)) == (105,45)
    @test nobs.(stratifiedobs(round, ys, p=(.2,.3))) == (30,45,75)
    @test nobs.(stratifiedobs(round, ys, p=(.2,.3), obsdim=:last)) == (30,45,75)
    @test nobs.(stratifiedobs(round, ys, p=(.1,.2,.3))) == (15,30,45,60)
end

@testset "Array, and SubArray" begin
    for var in (yv, y)
        @test nobs.(stratifiedobs(var)) == (105,45)
        @test nobs.(stratifiedobs(x->x, var)) == (105,45)
        @test nobs.(stratifiedobs(var, p=0.7)) == (105,45)
        @test nobs.(stratifiedobs(x->x, var, p=0.7)) == (105,45)
        @test nobs.(stratifiedobs(var, p=(.2,.3))) == (30,45,75)
        @test nobs.(stratifiedobs(var, p=(.2,.3), obsdim=:last)) == (30,45,75)
        @test nobs.(stratifiedobs(var, p=(.1,.2,.3))) == (15,30,45,60)
    end
    ty = [:a, :a, :a, :a, :a, :a, :b, :b, :b, :b]
    train, test = @inferred stratifiedobs(ty, 0.5, false)
    @test train == [:a, :a, :a, :b, :b] || train == [:b, :b, :a, :a, :a]
    @test test == [:a, :a, :a, :b, :b] || test == [:b, :b, :a, :a, :a]

    train, test = @inferred stratifiedobs(x->x, ty, 0.5, false)
    @test train == [:a, :a, :a, :b, :b] || train == [:b, :b, :a, :a, :a]
    @test test == [:a, :a, :a, :b, :b] || test == [:b, :b, :a, :a, :a]

    train, test, val = @inferred stratifiedobs(ty, (0.5,0.3), false)
    @test train == [:a, :a, :a, :b, :b] || train == [:b, :b, :a, :a, :a]
    @test test == [:a, :a, :b] || test == [:b, :a, :a]
    @test val == [:a, :b] || val == [:b, :a]

    ty = [:a, :a, :a, :a, :a, :a, :b, :b, :b, :b, :c, :c]
    train, test = @inferred stratifiedobs(ty, 0.5, false)
    @test train == [:c, :a, :a, :a, :b, :b] || train == [:a, :a, :a, :c, :b, :b] || train == [:b, :b, :c, :a, :a, :a] || train == [:c, :b, :b, :a, :a, :a] || train == [:b, :b, :a, :a, :a, :c] || train == [:a, :a, :a, :b, :b, :c]
    @test test == [:c, :a, :a, :a, :b, :b] || test == [:a, :a, :a, :c, :b, :b] || test == [:b, :b, :c, :a, :a, :a] || test == [:c, :b, :b, :a, :a, :a] || test == [:b, :b, :a, :a, :a, :c] || test == [:a, :a, :a, :b, :b, :c]
end
