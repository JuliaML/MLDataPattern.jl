@testset "nobs" begin
    @test_throws MethodError nobs(X,X)
    @test_throws MethodError nobs(X,y)

    @testset "Array, SparseArray, and Tuple" begin
        @test_throws DimensionMismatch nobs((X,XX,rand(100)))
        @test_throws DimensionMismatch nobs((X,X'))
        @test_throws DimensionMismatch nobs((X,XX), obsdim = :first)
        for var in (Xs, ys, vars...)
            @test @inferred(nobs(var, obsdim = :last)) === 150
            @test @inferred(nobs(var, obsdim = 100)) === 1
            @test @inferred(nobs(var)) === 150
            @test @inferred(nobs(var, ObsDim.Last())) === 150
        end
        @test @inferred(nobs(())) === 0
        @test @inferred(nobs((), obsdim = :first)) === 0
        @test @inferred(nobs((), obsdim = :last)) === 0
        @test @inferred(nobs((), obsdim = 3)) === 0
    end

    @testset "0-dim SubArray" begin
        v = view([3], 1)
        @test @inferred(nobs(v)) === 1
        @test @inferred(getobs(v)) === 3
        @test @inferred(getobs(v, 1)) === 3
        @test_throws BoundsError getobs(v, 2)
        @test_throws ArgumentError getobs(v, 2:3)
    end

    @testset "SubArray" begin
        @test @inferred(nobs(view(X,:,:))) === 150
        @test @inferred(nobs(view(X,:,:))) === 150
        @test @inferred(nobs(view(XX,:,:,:))) === 150
        @test @inferred(nobs(view(XXX,:,:,:,:))) === 150
        @test @inferred(nobs(view(y,:))) === 150
        @test @inferred(nobs(view(Y,:,:))) === 150
        @test @inferred(nobs(view(X,:,:), obsdim = :last)) === 150
        @test @inferred(nobs(view(XX,:,:,:), obsdim = :last)) === 150
        @test @inferred(nobs(view(XXX,:,:,:,:), obsdim = :last)) === 150
        @test @inferred(nobs(view(y,:), obsdim = :last)) === 150
        @test @inferred(nobs(view(Y,:,:), obsdim = :last)) === 150
    end

    @testset "various obsdim" begin
        @test_throws ArgumentError nobs(X, obsdim = 1.0)
        @test_throws ArgumentError nobs(X, obsdim = :one)
        @test_throws MethodError nobs(X, obsdim = ObsDim.Undefined())
        @test_throws DimensionMismatch nobs((X',X), (ObsDim.First(),ObsDim.Last(),ObsDim.Last()))
        @test_throws DimensionMismatch nobs((X',X), (ObsDim.First(),))
        @test_throws DimensionMismatch nobs((X',X), obsdim=(1,2,2))
        @test @inferred(nobs(X, ObsDim.Undefined())) === 150 # fallback
        @test @inferred(nobs(Xs, obsdim = 1)) === 10
        @test @inferred(nobs(Xs, obsdim = :first)) === 10
        @test @inferred(nobs(XXX, obsdim = 1)) === 3
        @test @inferred(nobs(XXX, ObsDim.First())) === 3
        @test @inferred(nobs(XXX, obsdim = :first)) === 3
        @test @inferred(nobs(XXX, obsdim = 2)) === 20
        @test @inferred(nobs(XXX, obsdim = 3)) === 30
        @test @inferred(nobs(XXX, ObsDim.Constant(3))) === 30
        @test @inferred(nobs(XXX, obsdim = 4)) === 150
        @test @inferred(nobs((X,y), obsdim = :last)) === 150
        @test @inferred(nobs((X',y), obsdim = :first)) === 150
        @test @inferred(nobs((X',X'), obsdim = :first)) === 150
        @test @inferred(nobs((X',X), obsdim = (:first,:last))) === 150
        @test @inferred(nobs((X',X), obsdim = (1,2))) === 150
        @test @inferred(nobs((X',X,X), obsdim = (1,2,2))) === 150
        @test @inferred(nobs((X',X,X), obsdim = (1,2,:last))) === 150
        @test @inferred(nobs((X',X), (ObsDim.First(),ObsDim.Last()))) === 150
        @test @inferred(nobs((X,X), obsdim = :first)) === 4
    end

    @testset "custom types" begin
        # test that fallback bouncing doesn't cause stackoverflow
        @test_throws MethodError nobs(EmptyType())
        @test_throws MethodError nobs(EmptyType(), ObsDim.Undefined())
        @test_throws MethodError nobs(EmptyType(), ObsDim.Last())
        @test_throws MethodError nobs(EmptyType(), (ObsDim.Last(),ObsDim.Last()))
        @test_throws MethodError nobs(EmptyType(), obsdim = 1)
        @test_throws MethodError nobs(EmptyType(), obsdim = (1,1))
        @test_throws MethodError nobs(EmptyType(), obsdim = :last)
        # test types that don't use the obsdim
        @test_throws MethodError nobs(CustomType(), obsdim = 1)
        @test_throws MethodError nobs(CustomType(), ObsDim.Last())
        @test nobs(CustomType()) === 100
    end
end

@testset "getobs" begin
    @testset "Array and Subarray" begin
        # access outside nobs bounds
        @test_throws BoundsError getobs(X, -1)
        @test_throws BoundsError getobs(X, 0)
        @test_throws BoundsError getobs(X, 0, obsdim = 1)
        @test_throws BoundsError getobs(X, 151)
        @test_throws BoundsError getobs(X, 151, obsdim = 2)
        @test_throws BoundsError getobs(X, 151, obsdim = 1)
        @test_throws BoundsError getobs(X, 5, obsdim = 1)
        @test typeof(@inferred(getobs(Xv))) <: Array
        @test typeof(@inferred(getobs(yv))) <: Array
        @test all(getobs(Xv) .== X)
        @test all(getobs(yv) .== y)
        @test getobs(X, ObsDim.Undefined()) === X
        @test getobs(X, ObsDim.Constant(1)) === X
        @test_throws MethodError getobs(X, obsdim = ObsDim.Undefined())
        @test getobs(X, obsdim = ObsDim.Constant(1)) === X
        @test @inferred(getobs(X))   === X
        @test @inferred(getobs(XX))  === XX
        @test @inferred(getobs(XXX)) === XXX
        @test @inferred(getobs(y))   === y
        @test @inferred(getobs(X, 45)) == getobs(X', 45, obsdim = 1)
        @test @inferred(getobs(X, 3:10)) == getobs(X', 3:10, obsdim = 1)'
        for i in (2, 2:20, [2,1,4])
            @test_throws ErrorException @inferred(getobs(XX, i, obsdim = 1))
            @test @inferred(getobs(XX, i, obsdim = :last) == getindex(XX,:,:,i))
            @test @inferred(getobs(XX, i, ObsDim.First())) == getindex(XX,i,:,:)
            @test @inferred(getobs(XX, i, ObsDim.Constant(2))) == getindex(XX,:,i,:)
            @test @inferred(getobs(XX, i, ObsDim.Last())) == getindex(XX,:,:,i)
            @test getobs(XX, i, obsdim = 1) == XX[i,:,:]
            @test getobs(XX, i, obsdim = :first) == XX[i,:,:]
            @test getobs(XX, i, obsdim = 2) == XX[:,i,:]
            @test getobs(XX, i, obsdim = :last) == XX[:,:,i]
        end
        for i in (2, 1:150, 2:10, [2,5,7], [2,1])
            @test_throws MethodError getobs(X, i, ObsDim.Undefined())
            @test_throws DimensionMismatch getobs(X, i, obsdim = 12)
            @test typeof(getobs(Xv, i)) <: Array
            @test typeof(getobs(yv, i)) <: ((typeof(i) <: Int) ? String : Array)
            @test all(getobs(Xv,i) .== X[:,i])
            @test @inferred(getobs(Xv,i))  == X[:,i]
            @test @inferred(getobs(X,i))   == X[:,i]
            @test @inferred(getobs(XX,i))  == XX[:,:,i]
            @test @inferred(getobs(XXX,i)) == XXX[:,:,:,i]
            @test @inferred(getobs(y,i))   == ((typeof(i) <: Int) ? y[i] : y[i])
            @test @inferred(getobs(yv,i))  == ((typeof(i) <: Int) ? y[i] : y[i])
            @test @inferred(getobs(Y,i))   == Y[:,i]
        end
    end

    @testset "SparseArray" begin
        @test @inferred(getobs(Xs)) === Xs
        @test @inferred(getobs(ys)) === ys
        @test @inferred(getobs(Xs, 45)) == getobs(Xs', 45, obsdim = 1)
        @test @inferred(getobs(Xs, 3:9)) == getobs(Xs', 3:9, obsdim = 1)'
        @test typeof(getobs(Xs,2)) <: SparseVector
        @test typeof(getobs(Xs,1:5)) <: SparseMatrixCSC
        @test typeof(getobs(ys,2)) <: Float64
        @test typeof(getobs(ys,1:5)) <: SparseVector
        for i in (2, 2:10, [2,1,4])
            @test @inferred(getobs(Xs, i, ObsDim.First())) == Xs[i,:]
            @test_throws ErrorException @inferred(getobs(Xs, i, obsdim = 1))
            @test getobs(Xs, i, obsdim = 1) == Xs[i,:]
            @test getobs(Xs, i, obsdim = :first) == Xs[i,:]
            @test getobs(Xs, i, obsdim = 2) == Xs[:,i]
            @test getobs(Xs, i, obsdim = :last) == Xs[:,i]
        end
        for i in (2, 1:150, 2:10, [2,5,7], [2,1])
            @test_throws MethodError getobs(Xs, i, ObsDim.Undefined())
            @test_throws DimensionMismatch getobs(Xs, i, obsdim = 12)
            @test @inferred(getobs(Xs,i)) == Xs[:,i]
            @test @inferred(getobs(ys,i)) == ys[i]
            @test @inferred(getobs(ys, i, obsdim = :last)) == ys[i]
            @test getobs(ys, i, obsdim = :last) == ys[i]
            @test getobs(ys, i, obsdim = :first) == ys[i]
        end
    end

    @testset "Tuple" begin
        @test_throws DimensionMismatch getobs((X,yv), obsdim=2)
        # bounds checking correctly
        @test_throws BoundsError getobs((X,y), 151)
        # special case empty tuple
        @test @inferred(getobs((), 10, obsdim = 1)) === ()
        @test getobs((), obsdim=2) === ()
        @test @inferred(getobs(())) === ()
        @test @inferred(getobs((), ObsDim.Last())) === ()
        @test @inferred(getobs((), 10)) === ()
        @test getobs((), 10, obsdim = 1) === ()
        @test @inferred(getobs((X,y))) === (X,y)
        @test @inferred(getobs((X,yv))) == (X,y)
        @test @inferred(getobs((Xv,y))) == (X,y)
        @test @inferred(getobs((Xv,yv))) == (X,y)
        @test_throws DimensionMismatch @inferred(getobs((X',y))) == (X',y)
        @test @inferred(getobs((XX,X,y))) === (XX,X,y)
        @test @inferred(getobs((XXX,XX,X,y))) === (XXX,XX,X,y)
        tx, ty = getobs((Xv,yv))
        @test typeof(tx) <: Array
        @test typeof(ty) <: Array
        tx, ty = getobs((Xv,yv), 10:50)
        @test typeof(tx) <: Array
        @test typeof(ty) <: Array
        for i in (1:150, 2:10, [2,5,7], [2,1])
            @test_throws DimensionMismatch getobs((X',y), i)
            @test_throws DimensionMismatch getobs((X,y),  i, obsdim=2)
            @test_throws DimensionMismatch getobs((X',y), i, obsdim=2)
            @test_throws DimensionMismatch getobs((X,y), i, obsdim=(1,2))
            @test_throws DimensionMismatch getobs((X,y), i, obsdim=(2,1,1))
            @test_throws DimensionMismatch getobs((XX,X,y), i, obsdim=(2,2,1))
            @test_throws DimensionMismatch getobs((XX,X,y), i, obsdim=(3,2))
            @test @inferred(getobs((X,y), i))  == (X[:,i], y[i])
            @test @inferred(getobs((X,yv), i)) == (X[:,i], y[i])
            @test @inferred(getobs((Xv,y), i)) == (X[:,i], y[i])
            @test @inferred(getobs((X,Y), i))  == (X[:,i], Y[:,i])
            @test @inferred(getobs((X,yt), i)) == (X[:,i], yt[:,i])
            @test @inferred(getobs((XX,X,y), i)) == (XX[:,:,i], X[:,i], y[i])
            @test_throws ErrorException @inferred(getobs((XX,X,y), i, obsdim=(3,2,1)))
            @test getobs((XX,X,y), i, obsdim=(3,2,1)) == (XX[:,:,i], X[:,i], y[i])
            @test getobs((X, y), i, obsdim=:last)  == (X[:,i], y[i])
            @test getobs((X',y), i, obsdim=:first) == (X'[i,:], y[i])
            @test getobs((X,yv), i, obsdim=:last)  == (X[:,i], y[i])
            @test getobs((Xv,y), i, obsdim=:last)  == (X[:,i], y[i])
            @test getobs((X, Y), i, obsdim=:last)  == (X[:,i], Y[:,i])
            @test getobs((X',y), i, obsdim=:first)  == (X'[i,:], y[i])
            @test getobs((X, y), i, obsdim=(:last,:last))  == (X[:,i], y[i])
            @test getobs((X, y), i, obsdim=(2,1))  == (X[:,i], y[i])
            @test getobs((X',y), i, obsdim=(1,1))  == (X'[i,:], y[i])
            @test getobs((X',yt), i, obsdim=(1,2))  == (X'[i,:], yt[:,i])
            @test getobs((X',yt), i, obsdim=(:first,:last))  == (X'[i,:], yt[:,i])
            @test getobs((XX,X,y), i, obsdim=:last) == (XX[:,:,i], X[:,i], y[i])
            # compare if obs match in tuple
            x1, y1 = getobs((X1,Y1), i)
            @test all(x1' .== y1)
            x1, y1, z1 = getobs((X1,Y1,sparse(X1)), i)
            @test all(x1' .== y1)
            @test all(x1 .== z1)
        end
        @test @inferred(getobs((X,y), 2)) == (X[:,2], y[2])
        @test @inferred(getobs((X,y), 2, obsdim=:last)) == (X[:,2], y[2])
        @test_throws ErrorException @inferred(getobs((X,y), 2, obsdim=(:last,:first)))
        @test getobs((X,y), 2, obsdim=:last) == (X[:,2], y[2])
        @test getobs((X,y), 2, obsdim=(:last,:first)) == (X[:,2], y[2])
        @test getobs((X,y), 2, obsdim=(2,:first)) == (X[:,2], y[2])
        @test @inferred(getobs((Xv,y), 2)) == (X[:,2], y[2])
        @test @inferred(getobs((X,yv), 2)) == (X[:,2], y[2])
        @test @inferred(getobs((X,Y), 2)) == (X[:,2], Y[:,2])
        @test @inferred(getobs((XX,X,y), 2)) == (XX[:,:,2], X[:,2], y[2])
    end

    @testset "type without getobs support" begin
        @test_throws MethodError getobs(EmptyType())
        @test_throws MethodError getobs(EmptyType(), obsdim=1)
        # test that fallback bouncing doesn't cause stackoverflow
        @test_throws MethodError getobs(EmptyType(), 1, ObsDim.Last())
        @test_throws MethodError getobs(EmptyType(), 1)
        @test_throws MethodError getobs(EmptyType(), 1:10)
        @test_throws MethodError getobs(EmptyType(), 1, obsdim=:first)
        @test_throws MethodError getobs(EmptyType(), 1, obsdim=10)
        @test_throws MethodError getobs(EmptyType(), 1:10, obsdim=:last)
    end

    @testset "custom type with getobs support" begin
        @test_throws MethodError getobs(CustomType(), obsdim=1)
        @test_throws MethodError getobs(CustomType(), ObsDim.Last())
        @test_throws MethodError getobs(CustomType(), 4:40, obsdim=1)
        @test @inferred(getobs(CustomType(), 11)) === 11
        @test @inferred(getobs(CustomType(), 4:40)) == collect(4:40)
        # defaults to 1:nobs
        @test @inferred(getobs(CustomType())) == collect(1:100)
        # No bounds checking here
        @test @inferred(getobs(CustomType(), 200)) === 200
        @test @inferred(getobs(CustomType(), [2,200,1])) == [2,200,1]
    end
end

@testset "getobs!" begin
    @testset "Array and Subarray" begin
        Xbuf = similar(X)
        # interpreted as idx
        @test_throws Exception getobs!(Xbuf, X, ObsDim.Undefined())
        @test_throws Exception getobs!(Xbuf, X, ObsDim.Constant(1))
        # obsdim not defined without some idx
        @test_throws MethodError getobs!(Xbuf, X, obsdim = ObsDim.Undefined())
        @test_throws MethodError getobs!(Xbuf, X, obsdim = ObsDim.Constant(1))
        # access outside nobs bounds
        @test_throws BoundsError getobs!(Xbuf, X, -1)
        @test_throws BoundsError getobs!(Xbuf, X, 0)
        @test_throws BoundsError getobs!(Xbuf, X, 0, obsdim = 1)
        @test_throws BoundsError getobs!(Xbuf, X, 151)
        @test_throws BoundsError getobs!(Xbuf, X, 151, obsdim = 2)
        @test_throws BoundsError getobs!(Xbuf, X, 151, obsdim = 1)
        @test_throws BoundsError getobs!(Xbuf, X, 5, obsdim = 1)
        @test @inferred(getobs!(Xbuf, X)) === Xbuf
        @test Xbuf == X
        @test all(getobs!(similar(Xv), Xv) .== X)
        @test all(getobs!(similar(yv), yv) .== y)
        @test @inferred(getobs!(similar(XX), XX))   == XX
        @test @inferred(getobs!(similar(XXX), XXX)) == XXX
        @test @inferred(getobs!(similar(y), y))     == y
        xbuf1 = zeros(4)
        xbuf2 = zeros(4)
        @test @inferred(getobs!(xbuf1, X, 45)) == getobs!(xbuf2, X', 45, obsdim = 1)
        Xbuf1 = zeros(4,8)
        Xbuf2 = zeros(8,4)
        @test @inferred(getobs!(Xbuf1, X, 3:10)) == getobs!(Xbuf2, X', 3:10, obsdim = 1)'
        # obsdim = 2
        Xbuf1 = zeros(20,150)
        @test_throws ErrorException @inferred(getobs!(Xbuf1, XX, 2, obsdim = 2))
        @test @inferred(getobs!(Xbuf1, XX, 5, ObsDim.Constant(2))) == XX[:,5,:]
        @test getobs!(Xbuf1, XX, 11, obsdim = 2) == XX[:,11,:]
        Xbuf2 = zeros(20,5,150)
        @test_throws ErrorException @inferred(getobs!(Xbuf2, XX, 6:10, obsdim = 2))
        @test @inferred(getobs!(Xbuf2, XX, 6:10, ObsDim.Constant(2))) == XX[:,6:10,:]
        @test getobs!(Xbuf2, XX, 11:15, obsdim = 2) == XX[:,11:15,:]
        # string vector
        @test getobs!("setosa", y, 1) == "setosa"
        @test getobs!(nothing, y, 1) == "setosa"
    end

    @testset "SparseArray" begin
        # Sparse Arrays opt-out of buffer usage
        @test @inferred(getobs!(nothing, Xs)) === getobs(Xs)
        @test @inferred(getobs!(nothing, Xs, 1)) == getobs(Xs, 1)
        @test @inferred(getobs!(nothing, Xs, 5:10)) == getobs(Xs, 5:10)
        @test @inferred(getobs!(nothing, Xs, 2, ObsDim.First())) == getobs(Xs, 2, obsdim=1)
        @test getobs!(nothing, Xs, 2, obsdim = 1) == getobs(Xs, 2, obsdim=1)
        @test @inferred(getobs!(nothing, ys)) === getobs(ys)
        @test @inferred(getobs!(nothing, ys, 1)) === getobs(ys, 1)
        @test @inferred(getobs!(nothing, ys, 5:10)) == getobs(ys, 5:10)
        @test @inferred(getobs!(nothing, ys, 5:10, ObsDim.First())) == getobs(ys, 5:10)
        @test getobs!(nothing, ys, 5:10, obsdim=1) == getobs(ys, 5:10)
    end

    @testset "Tuple" begin
        @test_throws MethodError getobs!((nothing,nothing), (X,y))
        @test_throws MethodError getobs!((nothing,nothing), (X,y), 1:5)
        @test_throws DimensionMismatch getobs!((nothing,nothing,nothing), (X,y))
        xbuf = zeros(4,2)
        ybuf = ["foo", "bar"]
        @test_throws DimensionMismatch getobs!((xbuf,), (X,y))
        @test_throws DimensionMismatch getobs!((xbuf,ybuf,ybuf), (X,y))
        @test_throws DimensionMismatch getobs!((xbuf,), (X,y), 1:5)
        @test_throws DimensionMismatch getobs!((xbuf,ybuf,ybuf), (X,y), 1:5)
        @test @inferred(getobs!((xbuf,ybuf),(X,y), 2:3)) === (xbuf,ybuf)
        @test xbuf == getobs(X, 2:3)
        @test ybuf == getobs(y, 2:3)
        @test @inferred(getobs!((xbuf,ybuf),(X,y), [50,150])) === (xbuf,ybuf)
        @test xbuf == getobs(X, [50,150])
        @test ybuf == getobs(y, [50,150])

        xbuf2 = zeros(2,4)
        @test @inferred(getobs!((xbuf2,ybuf),(X',y), 4:5, ObsDim.First())) === (xbuf2,ybuf)
        @test xbuf2 == getobs(X', 4:5, obsdim=1)
        @test ybuf  == getobs(y, 2:3)

        @test @inferred(getobs!((xbuf2,ybuf,xbuf),(X',y,X), 99:100, (ObsDim.First(),ObsDim.Last(),ObsDim.Last()))) === (xbuf2,ybuf,xbuf)
        @test xbuf2 == getobs(X', 99:100, obsdim=1)
        @test ybuf  == getobs(y, 99:100)
        @test xbuf == getobs(X, 99:100)

        @test getobs!((xbuf2,ybuf,xbuf),(X',y,X), 9:10, obsdim=(1,1,2)) === (xbuf2,ybuf,xbuf)
        @test xbuf2 == getobs(X', 9:10, obsdim=1)
        @test ybuf  == getobs(y, 9:10)
        @test xbuf == getobs(X, 9:10)

        @test getobs!((nothing,xbuf),(Xs,X), 3:4) == (getobs(Xs,3:4),xbuf)
        @test xbuf == getobs(X,3:4)
    end

    @testset "type without getobs support" begin
        # buffer is ignored if getobs! is not defined
        @test_throws MethodError getobs!(nothing, EmptyType())
        @test_throws MethodError getobs!(nothing, EmptyType(), 1)
        @test_throws MethodError getobs!(nothing, EmptyType(), obsdim=1)
        @test_throws MethodError getobs!(nothing, EmptyType(), ObsDim.Last())
        @test_throws MethodError getobs!(nothing, CustomType(), obsdim=1)
        @test_throws MethodError getobs!(nothing, CustomType(), ObsDim.Last())
        @test_throws MethodError getobs!(nothing, CustomType(), 4:40, obsdim=1)
    end

    @testset "custom type with getobs support" begin
        # No-op unless defined
        @test @inferred(getobs!(nothing, CustomType())) == collect(1:100)
        @test @inferred(getobs!(nothing, CustomType(), 11)) === 11
        @test @inferred(getobs!(nothing, CustomType(), 4:40)) == collect(4:40)
        # No bounds checking here
        @test @inferred(getobs!(nothing, CustomType(), 200)) === 200
        @test @inferred(getobs!(nothing, CustomType(), [2,200,1])) == [2,200,1]
    end
end
