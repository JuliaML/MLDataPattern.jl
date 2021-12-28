@testset "DataSubset constructor" begin
    @testset "bounds check" begin
        for var in (vars..., tuples..., )
            @test_throws BoundsError getobs(DataSubset(var, -1:100), 1)
            @test_throws BoundsError getobs(DataSubset(var, 1:151), 151)
            @test_throws BoundsError getobs(DataSubset(var, [1, 10, 0, 3]), 3)
            @test_throws BoundsError getobs(DataSubset(var, [1, 10, -10, 3]), 3)
            @test_throws BoundsError getobs(DataSubset(var, [1, 10, 180, 3]), 3)
        end
        # index validation is done in getobs function, if a type does not have it, getobs passes
        var = CustomType()
        @test getobs(DataSubset(var, -1:100), 1) == -1
        @test getobs(DataSubset(var, 1:151), 151) == 151
        @test getobs(DataSubset(var, [1, 10, 0, 3]), 3) == 0
        @test getobs(DataSubset(var, [1, 10, -10, 3]), 3) == -10
        @test getobs(DataSubset(var, [1, 10, 180, 3]), 3) == 180
    end

    @testset "Tuple unrolling" begin
        @test typeof(@inferred(DataSubset((X,X), (1:nobs(X), 1:nobs(X))))) <: Tuple
        @test eltype(@inferred(DataSubset((X,X), (1:nobs(X), 1:nobs(X))))) <: DataSubset
        @test typeof(@inferred(DataSubset((X,X), 1:150))) <: Tuple
        @test eltype(@inferred(DataSubset((X,X), 1:150))) <: DataSubset
        D1 = @inferred(DataSubset((X', X), (1:nobs(X), 1:nobs(X))))
        D2 = @inferred(DataSubset((X', X), 1:150))
        for (s1, s2) in (D1, D2)
            @test typeof(datasubset(s1, 2:10)) <: DataSubset
            @test @inferred(datasubset(s1, 2:10)) == @inferred(s1[2:10])
            @test @inferred(datasubset(s1, 2:10)) == @inferred(DataSubset(s1, 2:10))
            @test getobs(s1, 2; obsdim = 1) == getobs(s2, 2)
            @test getobs(s1, 9:10; obsdim = 1) == getobs(s2, 9:10)'
            @test getobs((s1, s2), 9:10; obsdim = (1, 2)) == (getobs(s1, 9:10; obsdim = 1), getobs(s2, 9:10))
            @test nobs(s1; obsdim = 1) == nobs(s2) == 150
        end
    end

    @testset "Array, SubArray, SparseArray" begin
        @test nobs(DataSubset(X, 1:nobs(X; obsdim = 1)); obsdim = 1) == 4
        @test nobs(DataSubset(X, 1:3); obsdim = 1) == 3
        @test_reference "references/DataSubset1.txt" DataSubset(X, 1:nobs(X)) by=matrix_compat_isequal
        @test_reference "references/DataSubset2.txt" @io2str(showcompact(::IO, DataSubset(X, 1:nobs(X)))) by=matrix_compat_isequal
        # var = Xs
        for var in (Xs, ys, vars...)
            subset = @inferred(DataSubset(var, 1:nobs(var)))
            @test subset.data === var
            @test subset.indices === 1:150
            @test typeof(subset) <: DataSubset
            @test @inferred(nobs(subset)) === nobs(var)
            @test @inferred(getobs(subset, subset.indices)) == getobs(var, 1:nobs(var))
            @test @inferred(DataSubset(subset, 1:nobs(subset))) == subset
            @test @inferred(DataSubset(subset, 1:150)) == subset
            @test subset[end] == DataSubset(var, 150)
            @test @inferred(subset[150]) == DataSubset(var, 150)
            @test @inferred(subset[20:25]) == DataSubset(var, 20:25)
            for idx in (1:100, [1,10,150,3], [2])
                @test DataSubset(var, 1:nobs(var))[idx] == DataSubset(var, idx)
                @test DataSubset(var, 1:nobs(var))[idx] == DataSubset(var, collect(idx))
                subset = @inferred(DataSubset(var, idx))
                @test typeof(subset) <: DataSubset{typeof(var), typeof(idx)}
                @test subset.data === var
                @test subset.indices === idx
                @test @inferred(nobs(subset)) === length(idx)
                @test @inferred(getobs(subset, 1:nobs(subset))) == getobs(var, idx)
                @test @inferred(DataSubset(subset, 1:nobs(subset))) == subset
                @test @inferred(subset[1]) == DataSubset(var, idx[1])
                if typeof(idx) <: AbstractRange
                    @test typeof(@inferred(subset[1:1])) == typeof(DataSubset(var, idx[1:1]))
                    @test nobs(subset[1:1]) == nobs(DataSubset(var, idx[1:1]))
                else
                    @test typeof(@inferred(subset[1:1])) == typeof(DataSubset(var, view(idx, 1:1)))
                    @test nobs(subset[1:1]) == nobs(DataSubset(var, view(idx, 1:1)))
                end
            end
        end
    end

    @testset "custom types" begin
        N = nobs(CustomType())
        @test_throws MethodError getobs(DataSubset(EmptyType(), 1:10), 1)
        @test_throws MethodError DataSubset(CustomType(), 2:10, obsdim = 1)
        @test_throws BoundsError getobs(DataSubset(CustomType(), 11:20), 11)
        @test typeof(@inferred(DataSubset(CustomType(), 1:N))) <: DataSubset
        @test nobs(DataSubset(CustomType(), 1:N)) === 100
        @test nobs(DataSubset(CustomType(), 11:20)) === 10
        @test getobs(DataSubset(CustomType(), 1:N), 1:N) == collect(1:100)
        @test getobs(DataSubset(CustomType(), 11:20), 10) == 20
        @test getobs(DataSubset(CustomType(), 11:20), [3, 5]) == [13, 15]
    end
end

@testset "DataSubset getindex and getobs" begin
    @testset "Matrix and SubArray{T,2}" begin
        for var in (X, Xv)
            subset = @inferred(DataSubset(var, 101:150))
            @test typeof(@inferred(getobs(subset, 1:nobs(subset)))) <: Array{Float64,2}
            @test @inferred(nobs(subset)) == length(subset) == 50
            @test @inferred(subset[10:20]) == DataSubset(X, 110:120)
            @test @inferred(subset[11:21]) != DataSubset(X, 110:120)
            @test @inferred(getobs(subset, 10:20)) == X[:, 110:120]
            @test @inferred(getobs(subset, [11,10,14])) == X[:, [111,110,114]]
            @test typeof(subset[10:20]) <: DataSubset
            @test @inferred(subset[collect(10:20)]) == DataSubset(X, collect(110:120))
            @test typeof(subset[collect(10:20)]) <: DataSubset
            @test @inferred(getobs(subset, 1:nobs(subset))) == getobs(subset[1:end], 1:nobs(subset)) == X[:, 101:150]
        end
    end

    @testset "Vector and SubArray{T,1}" begin
        for var in (y, yv)
            subset = @inferred(DataSubset(var, 101:150))
            @test typeof(getobs(subset, 1:nobs(subset))) <: Array{String,1}
            @test @inferred(nobs(subset)) == length(subset) == 50
            @test @inferred(subset[10:20]) == DataSubset(y, 110:120)
            @test @inferred(getobs(subset, 10:20)) == y[110:120]
            @test @inferred(getobs(subset, [11,10,14])) == y[[111,110,114]]
            @test typeof(subset[10:20]) <: DataSubset
            @test @inferred(subset[collect(10:20)]) == DataSubset(y, collect(110:120))
            @test typeof(subset[collect(10:20)]) <: DataSubset
            @test @inferred(getobs(subset, 1:nobs(subset))) == getobs(subset[1:end], 1:nobs(subset)) == y[101:150]
        end
    end

    @testset "2-Tuple of Matrix, Vector, or SubArray"  begin
        for v1 in (X, Xv), v2 in (y, yv)
            subset = @inferred(DataSubset((v1,v2), 101:150))
            @test typeof(getobs(subset, 1:nobs(subset))) <: Tuple{Array{Float64,2},Array{String,1}}
            @test @inferred(nobs(subset)) == nobs(subset[1]) == nobs(subset[2]) == 50
            @test @inferred(subset[1][10:20]) == DataSubset(X, 110:120)
            @test @inferred(subset[2][10:20]) == DataSubset(y, 110:120)
            @test @inferred(getobs(subset, 1:nobs(subset))) == (X[:, 101:150], y[101:150])
        end
    end

    @testset "2-Tuple of SparseArray"  begin
        subset = @inferred(DataSubset((Xs,ys), 101:150))
        @test typeof(subset) <: Tuple
        @test typeof(subset[1]) <: DataSubset
        @test typeof(subset[2]) <: DataSubset
        @test typeof(@inferred(getobs(subset, 1:nobs(subset)))) <: Tuple
        @test typeof(getobs(subset, 1:nobs(subset))[1]) <: SparseMatrixCSC
        @test typeof(getobs(subset, 1:nobs(subset))[2]) <: SparseVector
        @test @inferred(nobs(subset)) == nobs(subset[1]) == nobs(subset[2]) == 50
        @test @inferred(getobs(subset[1][10:20], 1:11)) == getindex(Xs, :, 110:120)
        @test @inferred(getobs(subset[2][10:20], 1:11)) == getindex(ys, 110:120)
        @test @inferred(getobs(subset, 1:nobs(subset))) == (getindex(Xs, :, 101:150), getindex(ys, 101:150))
    end
end

@testset "datasubset" begin
    @testset "Array and SubArray" begin
        @test getobs(@inferred(datasubset(X, 1:nobs(X))), 1:nobs(X)) == Xv
        @test typeof(getobs(datasubset(X, 1:nobs(X)), 1:nobs(X))) <: AbstractArray
        @test typeof(datasubset(X, 1:nobs(X))) <: SubArray
        @test getobs(@inferred(datasubset(XX, 1:nobs(XX))), 1:nobs(XX)) == XX
        @test getobs(@inferred(datasubset(XXX, 1:nobs(XXX))), 1:nobs(XXX)) == XXX
        @test typeof(datasubset(XXX, 1:nobs(XXX))) <: SubArray
        @test getobs(@inferred(datasubset(y, 1:nobs(y))), 1:nobs(y)) == y
        @test typeof(getobs(datasubset(y, 1:nobs(y)), 1:nobs(y))) <: AbstractArray
        for i in (2, 1:150, 2:10, [2,5,7], [2,1])
            idx = (length(i) == 1) ? 1 : 1:length(i)
            extract = (length(i) == 1) ? only : identity
            @test getobs(@inferred(datasubset(X, i)), idx)   == view(X, :, i)
            @test getobs(@inferred(datasubset(Xv, i)), idx)  == view(X, :, i)
            @test getobs(@inferred(datasubset(Xv, i)), idx)  == view(Xv, :, i)
            @test getobs(@inferred(datasubset(XX, i)), idx)  == view(XX, :, :, i)
            @test getobs(@inferred(datasubset(XXX, i)), idx) == view(XXX, :, :, :, i)
            @test getobs(@inferred(datasubset(y, i)), idx)   == view(y, i) |> extract
            @test getobs(@inferred(datasubset(yv, i)), idx)  == view(y, i) |> extract
            @test getobs(@inferred(datasubset(yv, i)), idx)  == view(yv, i) |> extract
            @test getobs(@inferred(datasubset(Y, i)), idx)   == view(Y, :, i)
        end
    end

    @testset "Tuple of Array and Subarray" begin
        @test @inferred(datasubset((X, y), (1:nobs(X), 1:nobs(y)))) == (X, y)
        @test @inferred(datasubset((X, yv), (1:nobs(X), 1:nobs(yv)))) == (X, yv)
        @test @inferred(datasubset((Xv, y), (1:nobs(Xv), 1:nobs(y)))) == (Xv, y)
        @test @inferred(datasubset((Xv, yv), (:, :))) === (Xv, yv)
        @test @inferred(datasubset((X, Y), (1:nobs(X), 1:nobs(Y)))) == (X,Y)
        @test @inferred(datasubset((XX,X,y), map(x -> 1:nobs(x), (XX, X, y)))) == (XX, X, y)
        for i in (1:150, 2:10, [2,5,7], [2,1])
            @test @inferred(datasubset((X,y),i))   === (view(X,:,i), view(y,i))
            @test @inferred(datasubset((Xv,y),i))  === (view(X,:,i), view(y,i))
            @test @inferred(datasubset((X,yv),i))  === (view(X,:,i), view(y,i))
            @test @inferred(datasubset((Xv,yv),i)) === (view(X,:,i), view(y,i))
            @test @inferred(datasubset((XX,X,y),i)) === (view(XX,:,:,i), view(X,:,i),view(y,i))
            # compare if obs match in tuple
            x1, y1 = getobs(datasubset((X1,Y1), i), 1:length(i))
            @test all(x1' .== y1)
            x1, y1, z1 = getobs(datasubset((X1,Y1,X1), i), 1:length(i))
            @test all(x1' .== y1)
            @test all(x1 .== z1)
        end
    end

    @testset "SparseArray" begin
        @test @inferred(datasubset(Xs, 1:nobs(Xs))) === DataSubset(Xs, 1:nobs(Xs))
        @test @inferred(datasubset(ys, 1:nobs(ys))) === DataSubset(ys, 1:nobs(ys))
        for i in (1:150, 2:10, [2,5,7], [2,1])
            @test @inferred(datasubset(Xs,i)) === DataSubset(Xs,i)
            @test @inferred(datasubset(ys,i)) === DataSubset(ys,i)
        end
    end

    @testset "Tuple of SparseArray" begin
        @test @inferred(datasubset((Xv,ys), (:, :))) === (Xv, DataSubset(ys, :))
        @test @inferred(datasubset((X,ys), 1:nobs(X)))  === (datasubset(X, 1:nobs(X)), DataSubset(ys, 1:nobs(X)))
        @test @inferred(datasubset((Xs,y), (:, :)))  === (DataSubset(Xs, :),datasubset(y, :))
        @test @inferred(datasubset((Xs,ys), 1:nobs(Xs))) === (DataSubset(Xs, 1:nobs(X)),DataSubset(ys, 1:nobs(X)))
        @test @inferred(datasubset((Xs,Xs), :)) === (DataSubset(Xs, :),DataSubset(Xs, :))
        @test @inferred(datasubset((ys,Xs), :)) === (DataSubset(ys, :),DataSubset(Xs, :))
        @test @inferred(datasubset((XX,Xs,yv), :)) === (datasubset(XX, :),DataSubset(Xs, :),yv)
        for i in (1:150, 2:10, [2,5,7], [2,1])
            @test @inferred(datasubset((X,ys),i))  === (view(X,:,i), DataSubset(ys,i))
            @test @inferred(datasubset((Xs,y),i))  === (DataSubset(Xs,i), view(y,i))
            @test @inferred(datasubset((Xs,ys),i)) === (DataSubset(Xs,i), DataSubset(ys,i))
            @test @inferred(datasubset((Xs,Xs),i)) === (DataSubset(Xs,i), DataSubset(Xs,i))
            @test @inferred(datasubset((ys,Xs),i)) === (DataSubset(ys,i), DataSubset(Xs,i))
            @test @inferred(datasubset((XX,Xs,y),i)) === (view(XX,:,:,i),DataSubset(Xs,i),view(y,i))
            # compare if obs match in tuple
            x1, y1 = getobs(datasubset((X1,sparse(Y1)), i), 1:length(i))
            @test all(x1' .== y1)
            x1, y1, z1 = getobs(datasubset((X1,Y1,sparse(X1)), i), 1:length(i))
            @test all(x1' .== y1)
            @test all(x1 .== z1)
        end
    end

    @testset "custom types" begin
        @test_throws MethodError getobs(datasubset(EmptyType(), 1:10), 1)
        @test_throws BoundsError getobs(datasubset(CustomType(), 11:20), 11)
        @test typeof(@inferred(datasubset(CustomType(), 1:nobs(CustomType())))) <: DataSubset
        @test datasubset(CustomType(), 1:nobs(CustomType())) == DataSubset(CustomType(), 1:nobs(CustomType()))
        @test datasubset(CustomType(), 11:20) == DataSubset(CustomType(), 11:20)
        @test nobs(datasubset(CustomType(), 1:nobs(CustomType()))) === 100
        @test nobs(datasubset(CustomType(), 11:20)) === 10
        @test getobs(datasubset(CustomType(), 1:nobs(CustomType())), 1:nobs(CustomType())) == collect(1:100)
        @test getobs(datasubset(CustomType(), 11:20), 10) == 20
        @test getobs(datasubset(CustomType(), 11:20), [3,5]) == [13,15]
    end
end

@testset "getobs!" begin
    @test getobs!(nothing, datasubset(y, 1), 1) == "setosa"

    @testset "DataSubset" begin
        xbuf1 = zeros(4,8)
        s1 = DataSubset(X, 2:9)
        @test @inferred(getobs!(xbuf1,s1,1:nobs(s1))) === xbuf1
        @test xbuf1 == getobs(s1, 1:nobs(s1))
        xbuf1 = zeros(4,5)
        s1 = DataSubset(X, 10:17)
        @test @inferred(getobs!(xbuf1,s1,2:6)) === xbuf1
        @test xbuf1 == getobs(s1,2:6) == getobs(X,11:15)

        xbuf2 = zeros(8,4)
        s2 = DataSubset(X', 2:9)
        @test @inferred(getobs!(xbuf2,s2,1:nobs(s2); obsdim=1)) === xbuf2
        @test xbuf2 == getobs(s2,1:nobs(s2); obsdim=1)
        xbuf2 = zeros(5,4)
        s2 = DataSubset(X', 10:17)
        @test @inferred(getobs!(xbuf2,s2,2:6; obsdim=1)) === xbuf2
        @test xbuf2 == getobs(s2,2:6; obsdim=1) == getobs(X',11:15,obsdim=1)

        s3 = DataSubset(Xs, 11:15)
        @test @inferred(getobs!(nothing,s3,1:nobs(s3))) == getobs(Xs,11:15)

        s4 = DataSubset(CustomType(), 6:10)
        @test @inferred(getobs!(nothing,s4,1:nobs(s4))) == getobs(s4, 1:nobs(s4))
        s5 = DataSubset(CustomType(), 9:20)
        @test @inferred(getobs!(nothing,s5,2:6)) == getobs(s5,2:6)
    end

    @testset "Tuple with DataSubset" begin
        xbuf = zeros(4,2)
        ybuf = ["foo", "bar"]
        s1 = DataSubset(Xs, 5:9)
        s2 = DataSubset(X, 5:9)
        @test_throws BoundsError getobs!((nothing,xbuf),(s1,s2),2:3; obsdim=1)
        @test getobs!((nothing,xbuf),(s1,s2),2:3; obsdim=(1,2)) == (getobs(Xs,6:7; obsdim=1),xbuf)
        @test getobs!((nothing,xbuf),(s1,s2), 2:3) == (getobs(Xs,6:7),xbuf)
        @test xbuf == getobs(X,6:7)
        @test getobs!((nothing,xbuf),(s1,s2), 2:3; obsdim=2) == (getobs(Xs,6:7),xbuf)
        @test getobs!((nothing,xbuf),(s1,s2), 2:3; obsdim=(2,2)) == (getobs(Xs,6:7),xbuf)
    end
end
