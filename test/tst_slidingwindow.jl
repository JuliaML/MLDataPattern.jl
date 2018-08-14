@testset "UnlabeledSlidingWindow" begin
    @test_throws UndefVarError UnlabeledSlidingWindow
    @test MLDataPattern.UnlabeledSlidingWindow <: AbstractVector
    @test MLDataPattern.UnlabeledSlidingWindow <: DataView
    @test !(MLDataPattern.UnlabeledSlidingWindow <: AbstractObsIterator)
    @test !(MLDataPattern.UnlabeledSlidingWindow <: AbstractBatchIterator)

    @testset "constructor" begin
        @test_throws DimensionMismatch slidingwindow((rand(2,10),rand(9)), 1)
        @test_throws DimensionMismatch slidingwindow((rand(2,10),rand(4,9,10),rand(9)), 1)
        @test_throws MethodError slidingwindow(EmptyType(), 1)
        @test_throws MethodError slidingwindow(EmptyType(), 1, ObsDim.Last())
        @test_throws MethodError slidingwindow(EmptyType(), 1, ObsDim.Undefined())
        @test_throws MethodError slidingwindow(EmptyType(), 1, ObsDim.Last())
        @test_throws MethodError slidingwindow((EmptyType(),EmptyType()))
        @test_throws MethodError slidingwindow(CustomType(), 1, ObsDim.Last())
        @test_throws MethodError slidingwindow(EmptyType(), 1, obsdim=1)
        @test_throws MethodError slidingwindow((EmptyType(),EmptyType()), 1)

        for var in (vars..., tuples..., Xs, ys)
            @test_throws MethodError slidingwindow(var...)
            @test_throws MethodError slidingwindow(var..., 1)
            @test_throws MethodError slidingwindow(var..., obsdim=:last)
            @test_throws MethodError slidingwindow(var..., 1, obsdim=:last)
            @test_throws ArgumentError slidingwindow(var, 151)
            @test_throws ArgumentError slidingwindow(var, 0)
            @test_throws ArgumentError slidingwindow(var, 1, 0)
            A = @inferred slidingwindow(var, 5, 6)
            @test A == slidingwindow(var, 5, stride = 6)
            @test @inferred(length(A)) == A.count == 25
            @test A.size == 5
            @test A.stride == 6
            @test A.obsdim == LearnBase.default_obsdim(var)
            @test @inferred(nobs(A)) == length(A)
            @test @inferred(parent(A)) === var
            A = @inferred(slidingwindow(var, 4))
            @test A == slidingwindow(var, 4, stride = 4)
            @test length(A) == A.count == 37
            @test A.size == 4
            @test A.stride == 4
            @test A.obsdim == LearnBase.default_obsdim(var)
            @test nobs(A) == length(A)
            @test @inferred(parent(A)) === var
        end
        @test slidingwindow((X,X), 5) == @inferred(slidingwindow((X,X), 5, (ObsDim.Last(),ObsDim.Last())))
        @test slidingwindow((X,X), 5) == @inferred(slidingwindow((X,X), 5, 5, (ObsDim.Last(),ObsDim.Last())))
        if Int == Int64
            @test_reference "references/unlabeledslidingwindow.txt" @io2str show(::IO, MIME"text/plain"(), slidingwindow(1:6, 3, 2))
        end
        A = slidingwindow(X',5,obsdim=1)
        @test A == @inferred(slidingwindow(X',5,ObsDim.First()))
        @test A == @inferred(slidingwindow(X',5,5,ObsDim.First()))
        @test A == slidingwindow(X',5,obsdim=:first)
        @test A == slidingwindow(X',5,stride=5,obsdim=:first)
    end

    println("<HEARTBEAT>")

    @testset "typestability" begin
        @test_throws ErrorException @inferred(slidingwindow(X, 5, obsdim=2))
        for var in (vars..., tuples..., Xs, ys)
            @test typeof(@inferred(slidingwindow(var, 5))) <: MLDataPattern.UnlabeledSlidingWindow
            @test typeof(@inferred(slidingwindow(var, 5, 3))) <: MLDataPattern.UnlabeledSlidingWindow
            @test typeof(@inferred(slidingwindow(var, 5, ObsDim.Last()))) <: MLDataPattern.UnlabeledSlidingWindow
            @test typeof(@inferred(slidingwindow(var, 5, 3, ObsDim.Last()))) <: MLDataPattern.UnlabeledSlidingWindow
            @test typeof(@inferred(slidingwindow(var, 5))) <: MLDataPattern.UnlabeledSlidingWindow
            @test_throws ErrorException @inferred(slidingwindow(var, 5, obsdim=:last))
        end
        for tup in tuples
            @test typeof(@inferred(slidingwindow(tup,5,(fill(ObsDim.Last(),length(tup))...,)))) <: MLDataPattern.UnlabeledSlidingWindow
            @test typeof(@inferred(slidingwindow(tup,5,3,(fill(ObsDim.Last(),length(tup))...,)))) <: MLDataPattern.UnlabeledSlidingWindow
        end
        @test typeof(@inferred(slidingwindow(CustomType(),5))) <: MLDataPattern.UnlabeledSlidingWindow
        @test typeof(@inferred(slidingwindow(CustomType(), 5, ObsDim.Undefined()))) <: MLDataPattern.UnlabeledSlidingWindow
    end

    println("<HEARTBEAT>")

    @testset "AbstractArray interface" begin
        for var in (vars..., tuples..., Xs, ys)
            A = slidingwindow(var, 5)
            @test_throws BoundsError A[-1]
            @test_throws BoundsError A[length(A)+1]
            @test @inferred(nobs(A)) == 30
            @test @inferred(length(A)) == 30
            @test @inferred(size(A)) == (30,)
            @test @inferred(A[2:3]) == [datasubset(var,(1*5+1):(2*5)), datasubset(var,(2*5+1):(3*5))]
            @test eltype(A[2:3]) == typeof(datasubset(var,(1*5+1):(2*5)))
            @test @inferred(A[[1,3]]) == [datasubset(var,(0*5+1):(1*5)), datasubset(var,(2*5+1):(3*5))]
            @test @inferred(A[1]) == datasubset(var, 1:5)
            @test @inferred(A[30]) == datasubset(var, (29*5+1):(30*5))
            @test @inferred(A[30]) == datasubset(var, nobs(var)-4:nobs(var))
            @test A[end] == A[30]
            @test @inferred(getobs(A,1)) == getobs(var, 1:5)
            @test @inferred(getobs(A,2:3)) == [getobs(var,(1*5+1):(2*5)), getobs(var,(2*5+1):(3*5))]
            if !(var isa Tuple)
                @test eltype(getobs(A,1)) == eltype(var)
            end
            @test typeof(getobs(A,1)) == typeof(getobs(var, 1:5))
            @test eltype(getobs(A,2:3)) == typeof(getobs(var, 1:5))
            @test typeof(@inferred(collect(A))) <: Vector
            # custom stride
            A = slidingwindow(var, 5, 2)
            @test_throws BoundsError A[-1]
            @test_throws BoundsError A[length(A)+1]
            @test @inferred(nobs(A)) == 73
            @test @inferred(length(A)) == 73
            @test @inferred(size(A)) == (73,)
            @test A[end] == A[73]
            @test @inferred(getobs(A,1)) == getobs(var, 1:5)
            @test @inferred(A[2:3]) == [datasubset(var,3:3+4), datasubset(var,5:5+4)]
            if !(var isa Tuple)
                @test eltype(getobs(A,1)) == eltype(var)
            end
            @test typeof(@inferred(collect(A))) <: Vector
        end
    end

    println("<HEARTBEAT>")

    @testset "batchview" begin
        bv = batchview(slidingwindow(1:100, 5))
        @test_throws MethodError shuffleobs(bv)
        @test_throws MethodError splitobs(bv)
        @test_throws MethodError oversample(bv)
        @test_throws MethodError undersample(bv)
        @test_throws MethodError stratifiedobs(bv)
        for var in vars # (vars..., tuples..., Xs, ys)
            A = slidingwindow(var, 5)
            bv = @inferred batchview(A, 2)
            go = @inferred getobs(bv)
            @test go isa Array
            @test go == bv
            @test length(bv) == length(A) / 2
            @test eltype(bv) <: Array{<:SubArray}
            bs = @inferred bv[1]
            @test length(bs) == 5
            @test all(nobs.(bs) .== 2)
        end
        A = batchview(slidingwindow(1:20, 2), 5)
        @test length(A) == 2
        @test A[1] == [[1,3,5,7,9],[2,4,6,8,10]]
        @test A[2] == [[11,13,15,17,19],[12,14,16,18,20]]
        @test A[1] isa Array
    end

    println("<HEARTBEAT>")

    @testset "special values" begin
        A = @inferred slidingwindow(1:10, 2)
        @test length(A) == 5
        @test getobs(A) == [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10],
        ]
        A = @inferred slidingwindow(1:10, 3)
        @test length(A) == 3
        @test getobs(A) == [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
        A = @inferred slidingwindow(1:10, 3, 2)
        @test length(A) == 4
        @test getobs(A) == [
            [1, 2, 3],
            [3, 4, 5],
            [5, 6, 7],
            [7, 8, 9],
        ]
        A = @inferred slidingwindow(1:10, 3, 4)
        @test length(A) == 2
        @test getobs(A) == [
            [1, 2, 3],
            [5, 6, 7],
        ]
    end
end

@testset "LabeledSlidingWindow" begin
    @test_throws UndefVarError LabeledSlidingWindow
    @test MLDataPattern.LabeledSlidingWindow <: AbstractVector
    @test MLDataPattern.LabeledSlidingWindow <: DataView
    @test !(MLDataPattern.LabeledSlidingWindow <: AbstractObsIterator)
    @test !(MLDataPattern.LabeledSlidingWindow <: AbstractBatchIterator)

    @testset "constructor" begin
        @test_throws DimensionMismatch slidingwindow(i->i, (rand(2,10),rand(9)), 1)
        @test_throws DimensionMismatch slidingwindow(i->i, (rand(2,10),rand(4,9,10),rand(9)), 1)
        @test_throws MethodError slidingwindow(i->i, EmptyType(), 1)
        @test_throws MethodError slidingwindow(i->i, EmptyType(), 1, ObsDim.Last())
        @test_throws MethodError slidingwindow(i->i, EmptyType(), 1, ObsDim.Undefined())
        @test_throws MethodError slidingwindow(i->i, EmptyType(), 1, ObsDim.Last())
        @test_throws MethodError slidingwindow(i->i, (EmptyType(),EmptyType()))
        @test_throws MethodError slidingwindow(i->i, CustomType(), 1, ObsDim.Last())
        @test_throws MethodError slidingwindow(i->i, EmptyType(), 1, obsdim=1)
        @test_throws MethodError slidingwindow(i->i, (EmptyType(),EmptyType()), 1)

        for var in (vars..., tuples..., Xs, ys)
            @test_throws MethodError slidingwindow(i->i, var...)
            @test_throws MethodError slidingwindow(i->i, var..., 1)
            @test_throws MethodError slidingwindow(i->i, var..., obsdim=:last)
            @test_throws MethodError slidingwindow(i->i, var..., 1, obsdim=:last)
            @test_throws ArgumentError slidingwindow(i->i, var, 151)
            @test_throws ArgumentError slidingwindow(i->i, var, 0)
            @test_throws ArgumentError slidingwindow(i->i, var, 1, 0)
            A = @inferred slidingwindow(identity, var, 5, 6)
            @test A == slidingwindow(i->i, var, 5, stride = 6)
            @test A == slidingwindow(i->i, var, 5, stride = 6, excludetarget=false)
            @test typeof(A) == typeof(slidingwindow(identity, var, 5, stride = 6, excludetarget=false))
            @test typeof(A) != typeof(slidingwindow(identity, var, 5, stride = 6, excludetarget=true))
            @test @inferred(length(A)) == A.count == 25
            @test A.size == 5
            @test A.stride == 6
            @test A.offset == 0
            @test A.obsdim == LearnBase.default_obsdim(var)
            @test @inferred(nobs(A)) == length(A)
            @test @inferred(parent(A)) === var
            A = @inferred(slidingwindow(i->i, var, 4))
            @test A == slidingwindow(i->i, var, 4, stride = 4)
            @test length(A) == A.count == 37
            @test A.size == 4
            @test A.stride == 4
            @test A.offset == 0
            @test A.obsdim == LearnBase.default_obsdim(var)
            @test nobs(A) == length(A)
            @test @inferred(parent(A)) === var
        end
        @test slidingwindow(i->i,(X,X), 5) == @inferred(slidingwindow(i->i,(X,X), 5, (ObsDim.Last(),ObsDim.Last())))
        @test slidingwindow(i->i,(X,X), 5) == @inferred(slidingwindow(i->i,(X,X), 5, Val{false}))
        @test slidingwindow(i->i,(X,X), 5) == @inferred(slidingwindow(i->i,(X,X), 5, 5, (ObsDim.Last(),ObsDim.Last())))
        @test slidingwindow(i->i,(X,X), 5) == @inferred(slidingwindow(i->i,(X,X), 5, 5, Val{false}))
        if Int == Int64
            @test_reference "references/labeledslidingwindow.txt" @io2str show(::IO, MIME"text/plain"(), slidingwindow(identity, 1:6, 3, 2, ObsDim.First()))
        end
        A = slidingwindow(i->i-1,X',5,obsdim=1)
        @test A.size == 5
        @test A.stride == 5
        @test A.offset == 1
        @test A.count == 29
        @test A.obsdim == LearnBase.ObsDim.First()
        @test A == @inferred(slidingwindow(i->i-1,X',5,ObsDim.First()))
        @test A == @inferred(slidingwindow(i->i-1,X',5,5,ObsDim.First()))
        @test A == slidingwindow(i->i-1,X',5,obsdim=:first)
        @test A == slidingwindow(i->i-1,X',5,stride=5,obsdim=:first)
    end

    println("<HEARTBEAT>")

    @testset "typestability" begin
        @test_throws ErrorException @inferred(slidingwindow(identity, X, 5, obsdim=2))
        for var in (vars..., tuples..., Xs, ys)
            @test typeof(@inferred(slidingwindow(identity, var, 5))) <: MLDataPattern.LabeledSlidingWindow
            @test typeof(@inferred(slidingwindow(identity, var, 5, 3))) <: MLDataPattern.LabeledSlidingWindow
            @test typeof(@inferred(slidingwindow(identity, var, 5, ObsDim.Last()))) <: MLDataPattern.LabeledSlidingWindow
            @test typeof(@inferred(slidingwindow(identity, var, 5, 3, ObsDim.Last()))) <: MLDataPattern.LabeledSlidingWindow
            @test typeof(@inferred(slidingwindow(identity, var, 5))) <: MLDataPattern.LabeledSlidingWindow
            @test_throws ErrorException @inferred(slidingwindow(identity, var, 5, obsdim=:last))
        end
        for tup in tuples
            @test typeof(@inferred(slidingwindow(identity,tup,5,(fill(ObsDim.Last(),length(tup))...,)))) <: MLDataPattern.LabeledSlidingWindow
            @test typeof(@inferred(slidingwindow(identity,tup,5,3,(fill(ObsDim.Last(),length(tup))...,)))) <: MLDataPattern.LabeledSlidingWindow
        end
        @test typeof(@inferred(slidingwindow(identity,CustomType(),5))) <: MLDataPattern.LabeledSlidingWindow
        @test typeof(@inferred(slidingwindow(identity,CustomType(), 5, ObsDim.Undefined()))) <: MLDataPattern.LabeledSlidingWindow
    end

    println("<HEARTBEAT>")

    @testset "AbstractArray interface" begin
        for var in (vars..., tuples..., Xs, ys)
            A = slidingwindow(i->i+5, var, 5)
            @test_throws BoundsError A[-1]
            @test_throws BoundsError A[length(A)+1]
            @test @inferred(nobs(A)) == 29
            @test @inferred(length(A)) == 29
            @test @inferred(size(A)) == (29,)
            ref = [
                (datasubset(var,(1*5+1):(2*5)), datasubset(var,2*5+1)),
                (datasubset(var,(2*5+1):(3*5)), datasubset(var,3*5+1))
            ]
            @test @inferred(A[2:3]) == ref
            @test eltype(A[2:3]) == typeof(ref[1])
            ref = [
                (datasubset(var,(0*5+1):(1*5)), datasubset(var,1*5+1)),
                (datasubset(var,(2*5+1):(3*5)), datasubset(var,3*5+1))
            ]
            @test @inferred(A[[1,3]]) == ref
            ref = (datasubset(var,(0*5+1):(1*5)), datasubset(var,1*5+1))
            @test @inferred(A[1]) == ref
            ref = (datasubset(var,(28*5+1):(29*5)), datasubset(var,29*5+1))
            @test @inferred(A[29]) == ref
            @test A[end] == A[29]
            ref = (getobs(var,1:5), getobs(var,6))
            @test @inferred(getobs(A,1)) == ref
            @test typeof(getobs(A,1)) == typeof(ref)
            ref = [
                (getobs(var,(1*5+1):(2*5)), getobs(var,2*5+1)),
                (getobs(var,(2*5+1):(3*5)), getobs(var,3*5+1))
            ]
            @test @inferred(getobs(A,2:3)) == ref
            @test typeof(getobs(A,2:3)) == typeof(ref)
            @test typeof(@inferred(collect(A))) <: Vector
            # custom stride
            A = slidingwindow(i->i-1, var, 5, 2)
            @test_throws BoundsError A[-1]
            @test_throws BoundsError A[length(A)+1]
            @test @inferred(nobs(A)) == 72
            @test @inferred(length(A)) == 72
            @test @inferred(size(A)) == (72,)
            @test A[end] == A[72]
            ref = (getobs(var,3:3+4), getobs(var,2))
            @test @inferred(getobs(A,1)) == ref
            ref = [
                (getobs(var,5:5+4), getobs(var,4)),
                (getobs(var,7:7+4), getobs(var,6)),
            ]
            @test @inferred(getobs(A,2:3)) == ref
            @test typeof(@inferred(collect(A))) <: Vector
            # TODO: exclude target
        end
    end

    println("<HEARTBEAT>")

    @testset "batchview" begin
        # number as target
        sw = slidingwindow(i->i+5, 1:31, 5)
        @test length(sw) == 6
        bv = @inferred batchview(sw, 2)
        @test_throws MethodError shuffleobs(bv)
        @test_throws MethodError splitobs(bv)
        @test_throws MethodError oversample(bv)
        @test_throws MethodError undersample(bv)
        @test_throws MethodError stratifiedobs(bv)
        @test length(bv) == 3
        @test nobs(bv) == 6
        tx, ty = @inferred bv[1]
        @test tx isa Array{<:SubArray}
        @test ty isa Array{<:SubArray}
        @test tx == [[1,6],[2,7],[3,8],[4,9],[5,10]]
        @test ty == [[6,11]]
        tx, ty = @inferred bv[2]
        @test tx == [[11,16],[12,17],[13,18],[14,19],[15,20]]
        @test ty == [[16,21]]
        tx, ty = @inferred bv[3]
        @test tx == [[21,26],[22,27],[23,28],[24,29],[25,30]]
        @test ty == [[26,31]]
        # also test with array as target
        sw = slidingwindow(i->i.+(5:6), 1:32, 5)
        @test length(sw) == 6
        bv = @inferred batchview(sw, 2)
        @test_throws MethodError shuffleobs(bv)
        @test_throws MethodError splitobs(bv)
        @test_throws MethodError oversample(bv)
        @test_throws MethodError undersample(bv)
        @test_throws MethodError stratifiedobs(bv)
        @test length(bv) == 3
        @test nobs(bv) == 6
        tx, ty = @inferred bv[1]
        @test tx isa Array{<:SubArray}
        @test ty isa Array{<:SubArray}
        @test tx == [[1,6],[2,7],[3,8],[4,9],[5,10]]
        @test ty == [[6,11],[7,12]]
        tx, ty = @inferred bv[2]
        @test tx == [[11,16],[12,17],[13,18],[14,19],[15,20]]
        @test ty == [[16,21],[17,22]]
        tx, ty = @inferred bv[3]
        @test tx == [[21,26],[22,27],[23,28],[24,29],[25,30]]
        @test ty == [[26,31],[27,32]]
    end

    println("<HEARTBEAT>")

    @testset "special values" begin
        A = @inferred slidingwindow(i->i+2, 1:10, 2)
        @test length(A) == 4
        @test getobs(A) == [
            ([1,2], 3),
            ([3,4], 5),
            ([5,6], 7),
            ([7,8], 9),
        ]
        A = @inferred slidingwindow(i->i+2, 1:10, 2, 3)
        @test length(A) == 3
        @test getobs(A) == [
            ([1,2], 3),
            ([4,5], 6),
            ([7,8], 9),
        ]
        A = @inferred slidingwindow(i->i+2:i+3, 1:10, 2)
        @test length(A) == 4
        @test getobs(A) == [
            ([1,2], [3,4]),
            ([3,4], [5,6]),
            ([5,6], [7,8]),
            ([7,8], [9,10]),
        ]
        A = @inferred slidingwindow(i->[i-2:i-1; i+1:i+2], 1:10, 1)
        @test length(A) == 6
        @test getobs(A) == [
            ([3], [1, 2, 4, 5]),
            ([4], [2, 3, 5, 6]),
            ([5], [3, 4, 6, 7]),
            ([6], [4, 5, 7, 8]),
            ([7], [5, 6, 8, 9]),
            ([8], [6, 7, 9, 10]),
        ]
        A = @inferred slidingwindow(i->i-1, 1:10, 2)
        @test length(A) == 4
        @test getobs(A) == [
            ([3,4], 2),
            ([5,6], 4),
            ([7,8], 6),
            ([9,10], 8),
        ]
        A = @inferred slidingwindow(i->i+1, 1:10, 3)
        @test length(A) == 3
        @test getobs(A) == [
            ([1,2,3], 2),
            ([4,5,6], 5),
            ([7,8,9], 8),
        ]
        A = @inferred slidingwindow(i->i+1, 1:10, 3, Val{true})
        @test length(A) == 3
        @test getobs(A) == [
            ([1,3], 2),
            ([4,6], 5),
            ([7,9], 8),
        ]
    end
end
