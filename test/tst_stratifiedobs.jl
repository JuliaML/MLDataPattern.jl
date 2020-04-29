@test_throws DimensionMismatch stratifiedobs((X, rand(149)))
@test_throws DimensionMismatch stratifiedobs((X, rand(149)), obsdim=:last)

Random.seed!(1335)

tmpty = [:a, :a, :a, :a, :a, :a, :b, :b, :b, :b]
@test sort.(splitobs(labelmap(tmpty), at = 0.5)) == ([1,2,3,7,8],[4,5,6,9,10])

@testset "Type Stability" begin
    for var in vars
        Random.seed!(1335)
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
        Random.seed!(1335)
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

Random.seed!(1335)
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

@testset "ObsView" begin
    ty = obsview([:a, :a, :a, :a, :a, :a, :b, :b, :b, :b])
    train, test = getobs.(@inferred(stratifiedobs(ty, 0.5, false)))
    @test train == [:a, :a, :a, :b, :b] || train == [:b, :b, :a, :a, :a]
    @test test == [:a, :a, :a, :b, :b] || test == [:b, :b, :a, :a, :a]

    train, test, val = getobs.(@inferred(stratifiedobs(ty, (0.5,0.3), false)))
    @test train == [:a, :a, :a, :b, :b] || train == [:b, :b, :a, :a, :a]
    @test test == [:a, :a, :b] || test == [:b, :a, :a]
    @test val == [:a, :b] || val == [:b, :a]
end

Random.seed!(42)
@testset "RNG" begin
    # tests reproducibility using explicit and global RNGs
    tX = [1:20;]
    ty = [:b, :b, :a, :b, :b, :b, :b, :a, :b, :a, :b, :b, :b, :a, :a, :b, :b, :b, :b, :b]
    (def_train_x, def_train_y), (def_test_x, def_test_y) = stratifiedobs((tX, ty))
    (exp_train_x, exp_train_y), (exp_test_x, exp_test_y) = stratifiedobs((tX, ty), rng=MersenneTwister(42))
    @test ((def_train_x, def_train_y), (def_test_x, def_test_y)) == ((exp_train_x, exp_train_y), (exp_test_x, exp_test_y))
    @test ((exp_train_x, exp_train_y), (exp_test_x, exp_test_y)) == stratifiedobs((tX, ty), rng=MersenneTwister(42))
end

@testset "RNG with callback" begin
    ty = [:b, :b, :a, :b, :b, :b, :b, :a, :b, :a, :b, :b, :b, :a, :a, :b, :b, :b, :b, :b]
    exp_train_y, exp_test_y = stratifiedobs(y->y==:a, ty, rng=MersenneTwister(42))
    @test (exp_train_y, exp_test_y) == stratifiedobs(y->y==:a, ty, rng=MersenneTwister(42))
end
