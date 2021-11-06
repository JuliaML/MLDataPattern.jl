@testset "various types" begin
    for var in (vars..., tuples...)
        # @inferred randobs(var)
        @inferred randobs(var, 4)
        @test typeof(randobs(var)) == typeof(getobs(var, 1))
        @test typeof(randobs(var, 4)) == typeof(getobs(var, 1:4))
        @test nobs(randobs(var, 4)) == nobs(getobs(var, 1:4))
    end
end

@testset "check that sampled obs exist" begin
    tX = DataSubset(X)
    for tX in (X, DataSubset(X))
        X_rnd = getobs(@inferred(randobs(tX, 30)), 1:30)
        for i = 1:30
            @testset "random obs $i" begin
                found = false
                for j = 1:150
                    if all(X_rnd[:,i] .== X[:,j])
                        found = true
                    end
                end
                @test found
            end
        end
    end
end

# test if obs in tuple match each other
@testset "Tuple" begin
    for i = 1:30
        @test 0 < randobs(CustomType()) <= 150
        x1, y1 = randobs((X1,Y1))
        @test all(x1 .== y1)
        x1, y1 = randobs((X1,Y1), 5)
        @test all(x1' .== y1)
    end
end
