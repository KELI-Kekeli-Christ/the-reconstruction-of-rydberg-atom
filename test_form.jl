using LinearAlgebra

function Q_bruit_dist(n, c6, L)
    Q = zeros(n, n)
    for i in 1:n, j in i+1:n
        d = rand() * L 
        Q[i,j] = Q[j,i] = c6 / d^6
    end
    return Q
end

n = 4
c6 = 1
L = 10

print(Q_bruit_dist(n, c6, L))