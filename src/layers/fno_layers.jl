struct SpectralConv2d <: Lux.AbstractLuxLayer
    in_channels::Int
    out_channels::Int
    modes1::Int
    modes2::Int
end

function Lux.initialparameters(rng::AbstractRNG, l::SpectralConv2d)
    scale = Float32(1 / (l.in_channels * l.out_channels))
    dims = (l.modes1, l.modes2, l.in_channels, l.out_channels)
    return (
        w1 = scale .* randn(rng, ComplexF32, dims...),
        w2 = scale .* randn(rng, ComplexF32, dims...),
    )
end

function compl_mul2d(input, weights)
    m1, m2, ic, b = size(input)
    oc = size(weights, 4)
    inp_p = permutedims(reshape(input, m1 * m2, ic, b), (2, 3, 1))  # (in, batch, mm)
    wts_p = permutedims(reshape(weights, m1 * m2, ic, oc), (2, 3, 1))  # (in, out,   mm)
    out_p = NNlib.batched_mul(NNlib.batched_adjoint(wts_p), inp_p)     # (out, batch, mm)
    return reshape(permutedims(out_p, (3, 1, 2)), m1, m2, oc, b)
end

function (m::SpectralConv2d)(x, ps, st)
    H, W = size(x, 1), size(x, 2)
    batch = size(x, 4)
    x_ft = rfft(x, [1, 2])
    ft_h = size(x_ft, 1)

    modes1 = m.modes1
    modes2 = m.modes2

    # rfft axis (dim 1): +ve low freqs at [1:modes1]
    # full-FFT axis (dim 2): +ve low [1:modes2] and -ve low [end-modes2+1:end]
    out_pp = compl_mul2d(x_ft[1:modes1, 1:modes2, :, :], ps.w1)
    out_pn = compl_mul2d(x_ft[1:modes1, (end - modes2 + 1):end, :, :], ps.w2)

    mid_pad_w = zeros(ComplexF32, modes1, W - 2 * modes2, m.out_channels, batch)
    out_low = cat(out_pp, mid_pad_w, out_pn; dims = 2)
    hi_pad = zeros(ComplexF32, ft_h - modes1, W, m.out_channels, batch)
    out_ft = cat(out_low, hi_pad; dims = 1)
    return irfft(out_ft, H, [1, 2]), st
end


struct FNO_MLP{C1, C2} <: Lux.AbstractLuxContainerLayer{(:conv1, :conv2)}
    conv1::C1
    conv2::C2
    activation::Function
end

function FNO_MLP(in_channels::Int, out_channels::Int, hidden_channels::Int, activation::String)
    conv1 = Lux.Conv((1, 1), in_channels => hidden_channels)
    conv2 = Lux.Conv((1, 1), hidden_channels => out_channels)
    return FNO_MLP(conv1, conv2, get_activation(activation))
end

function (m::FNO_MLP)(x, ps, st)
    x, st_c1 = m.conv1(x, ps.conv1, st.conv1)
    x = m.activation.(x)
    x, st_c2 = m.conv2(x, ps.conv2, st.conv2)
    return x, (conv1 = st_c1, conv2 = st_c2)
end


struct FNOBlock{S, M, C} <: Lux.AbstractLuxContainerLayer{(:spect_conv, :mlp, :conv)}
    spect_conv::S
    mlp::M
    conv::C
    activation::Function
end

function FNOBlock(width::Int, modes1::Int, modes2::Int, activation::String)
    spect_conv = SpectralConv2d(width, width, modes1, modes2)
    mlp = FNO_MLP(width, width, width, activation)
    conv = Lux.Conv((1, 1), width => width)
    return FNOBlock(spect_conv, mlp, conv, get_activation(activation))
end

function (m::FNOBlock)(x, ps, st)
    x2, st_c = m.conv(x, ps.conv, st.conv)
    x_s, st_s = m.spect_conv(x, ps.spect_conv, st.spect_conv)
    x_m, st_m = m.mlp(x_s, ps.mlp, st.mlp)
    return m.activation.(x_m .+ x2), (spect_conv = st_s, mlp = st_m, conv = st_c)
end
