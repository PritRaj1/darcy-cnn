struct ShannonWavelet <: Lux.AbstractLuxLayer
    in_dims::Int
    out_dims::Int
end

function Lux.initialparameters(rng::AbstractRNG, l::ShannonWavelet)
    return (weights = Lux.kaiming_uniform(rng, l.in_dims, l.out_dims),)
end

function _reactant_sinc(x)
    pix = Float32(pi) * x
    pix2 = pix * pix
    taylor = 1.0f0 - pix2 / 6.0f0 + pix2 * pix2 / 120.0f0
    direct = sin(pix) / (pix + eps(Float32))
    w = exp(-pix2 / 8.0f0)
    return w * taylor + (1.0f0 - w) * direct
end

function (l::ShannonWavelet)(x, ps, st)
    y = batch_mul(_reactant_sinc.(x .* Float32(2pi)), cos.(x .* Float32(pi / 3))) .* 2.0f0
    return node_mul(y, ps.weights), st
end
