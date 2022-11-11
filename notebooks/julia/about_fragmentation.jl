# using Plots

sim_time = 3 # days
dt = 24 #hours
fragmentation_timescale = 5*24. #hours
fragmentation_prob = 0.5# exp(-dt/fragmentation_timescale)

N = 10
diameters = zeros(sim_time, N)
diameters[1,:] .= 5e-8
fragmenting_state = rand(sim_time, N) .> fragmentation_prob
#
# for i = 1:sim_time
#

#
# end

function number_fragments(k, p)
    N = 0
    n_fragments = zeros(Float32, k)

    for i = 1:k
        n = (1 - p)*(p*2^3)^i
        n_fragments[i] = n
        N += n
    end
    p_fragments = n_fragments./N

    return n_fragments, p_fragments

end

a, b =number_fragments(3, 1/2)
println(a, b)


Π(x) = (3*x^4 - 8*x^3 + 4*x^2 + 2)x^4

function p_n(p::Float64)
    a = Π(p)
    return a
end

p_n(0.5)

k = 3
p = 0.5

n_dist, p_fragmenting = number_fragments(10, 0.5)

## ---




a = L_is_4_lenght(5e-8, 10)


function Fragmentation (particle, fieldset, time)

        if ParcelsRandom.random(0., 1.) > fragmentation_prob:
            nummer = ParcelsRandom.random(0., 1.)
            plim0 = 8./14.5
            plim1 = 12./14.5
            plim2 = 14./14.5

            if nummer <= plim0:
                frag_mode = 8

            elif (plim0 < nummer) and (nummer <= plim1):
                frag_mode = 4

            elif (plim1 < nummer) and (nummer <= plim2):
                frag_mode = 2

            else:
                frag_mode = 1

            particle.diameter = particle.diameter*frag_mode # division for reverse
end
