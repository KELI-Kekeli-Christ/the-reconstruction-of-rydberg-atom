using Pkg
Pkg.add("Optim")
using LinearAlgebra
using Random
using Optim


# ============================================================
# Q depuis une configuration réelle
# ============================================================

function build_Q(X,c6)

    n=size(X,1)

    Q=zeros(n,n)

    for i in 1:n
        for j in i+1:n

            r=norm(X[i,:]-X[j,:])

            Q[i,j]=c6/r^6
            Q[j,i]=Q[i,j]

        end
    end

    return Q
end



# ============================================================
# Q -> distances
# ============================================================

function Q_to_D(Q,c6)

    n=size(Q,1)

    D=zeros(n,n)

    for i in 1:n
        for j in i+1:n

            D[i,j]=(c6/Q[i,j])^(1/6)
            D[j,i]=D[i,j]

        end
    end

    return D
end



# ============================================================
# B double centrage
# ============================================================

function compute_B(D)

    n=size(D,1)

    H=I-ones(n,n)/n

    return -0.5*H*(D.^2)*H

end



# ============================================================
# Pseudo MDS initial
# ============================================================

function pseudo_init(B)


    F=eigen(Symmetric(B))


    λ=F.values
    V=F.vectors



    # prendre les 2 plus grandes valeurs propres
    idx=sortperm(abs.(λ),rev=true)[1:2]


    X=V[:,idx]*Diagonal(sqrt.(abs.(λ[idx])))


    return X

end




# ============================================================
# Distance avec métrique G = diag(1,-1)
# ============================================================

function distance_pseudo2D(xi,xj)


    dx1=xi[1]-xj[1]
    dx2=xi[2]-xj[2]


    d2=dx1^2-dx2^2


    return abs(d2)

end




# ============================================================
# Optimisation X G X'
# ============================================================

function optimize_pseudo2D(B,X0)


    n=size(X0,1)


    x0=vec(X0)



    function loss(z)


        X=reshape(z,n,2)


        err=0.0


        for i in 1:n
            for j in i+1:n


                diff=X[i,:]-X[j,:]


                d2 =
                diff[1]^2-diff[2]^2



                target=
                B[i,i]+B[j,j]-2B[i,j]


                err+=(d2-target)^2


            end
        end


        return err

    end



    res=optimize(
        loss,
        x0,
        LBFGS()
    )


    Xopt=
    reshape(Optim.minimizer(res),n,2)



    return Xopt,Optim.minimum(res)

end




# ============================================================
# Erreur Rydberg avec X 2D pseudo
# ============================================================

function rydberg_error_pseudo2D(X,Q,c6)


    n=size(X,1)

    err=0.0


    for i in 1:n

        for j in i+1:n


            dx=X[i,1]-X[j,1]
            dy=X[i,2]-X[j,2]


            d2=dx^2-dy^2


            if d2<=0
                continue
            end


            r=sqrt(d2)


            q=c6/r^6


            err+=(q-Q[i,j])^2

        end

    end


    return err

end




# ============================================================
# TEST
# ============================================================


Random.seed!(1)


n=10
c6=1.0



println("==============================")
println("Q PARFAIT")
println("==============================")


Xtrue=rand(n,2)


Q=build_Q(Xtrue,c6)


D=Q_to_D(Q,c6)


B=compute_B(D)



X0=pseudo_init(B)



Xopt,errB=
optimize_pseudo2D(B,X0)



println("Erreur B = ",errB)


println("Erreur Rydberg = ",
rydberg_error_pseudo2D(Xopt,Q,c6))





println("\n==============================")
println("Q ALEATOIRE")
println("==============================")


Qrand=zeros(n,n)


for i in 1:n
    for j in i+1:n

        Qrand[i,j]=rand()
        Qrand[j,i]=Qrand[i,j]

    end
end



D=Q_to_D(Qrand,c6)


B=compute_B(D)



X0=pseudo_init(B)



Xopt,errB=
optimize_pseudo2D(B,X0)



println("Erreur B = ",errB)


println("Erreur Rydberg = ",
rydberg_error_pseudo2D(Xopt,Qrand,c6))