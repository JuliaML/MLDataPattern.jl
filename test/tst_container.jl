@testset "nobs" begin
    @testset "custom types" begin
        # test that fallback bouncing doesn't cause stackoverflow
        @test_throws MethodError nobs(EmptyType())
        @test_throws MethodError nobs(EmptyType(); obsdim = 1)
        @test_throws MethodError nobs(EmptyType(); obsdim = (1, 2))
        @test nobs(CustomType()) === 100
    end
end

@testset "getobs" begin
    @testset "type without getobs support" begin
        # test that fallback bouncing doesn't cause stackoverflow
        @test_throws MethodError getobs(EmptyType(), 1)
        @test_throws MethodError getobs(EmptyType(), 1:10)
        @test_throws MethodError getobs(EmptyType(), 1; obsdim=1)
    end

    @testset "custom type with getobs support" begin
        @test getobs(CustomType(), 4:40; obsdim=1) == collect(4:40)
        @test @inferred(getobs(CustomType(), 11)) === 11
        @test @inferred(getobs(CustomType(), 4:40)) == collect(4:40)
        # No bounds checking here
        @test @inferred(getobs(CustomType(), 200)) === 200
        @test @inferred(getobs(CustomType(), [2,200,1])) == [2,200,1]
    end
end

@testset "getobs!" begin
    @testset "type without getobs support" begin
        # buffer is ignored if getobs! is not defined
        @test_throws MethodError getobs!(nothing, EmptyType())
        @test_throws MethodError getobs!(nothing, EmptyType(), 1)
        @test_throws MethodError getobs!(nothing, EmptyType(), obsdim=1)
        @test_throws MethodError getobs!(nothing, EmptyType(), obsdim=LearnBase.default_obsdim(EmptyType()))
        @test_throws MethodError getobs!(nothing, CustomType(), obsdim=1)
        @test_throws MethodError getobs!(nothing, CustomType(), LearnBase.default_obsdim(CustomType()))
    end

    @testset "custom type with getobs support" begin
        # No-op unless defined
        @test @inferred(getobs!(nothing, CustomType(), 11)) === 11
        @test @inferred(getobs!(nothing, CustomType(), 4:40)) == collect(4:40)
        # No bounds checking here
        @test @inferred(getobs!(nothing, CustomType(), 200)) === 200
        @test @inferred(getobs!(nothing, CustomType(), [2,200,1])) == [2,200,1]
    end
end
