struct KANConv2D{K} <: Lux.AbstractLuxContainerLayer{(:dense_kernel,)}
    dense_kernel::K
    kernel_size::Tuple{Int, Int}
    stride::Int
    dilation::Int
    padding::Int
end

function KANConv2D(
        in_channels::Int, out_channels::Int, kernel_size::Tuple{Int, Int},
        wavelet_name::String, base_activation::String;
        stride::Int = 1, dilation::Int = 1, padding::Int = 1, norm::Bool = false
    )
    dense_kernel = KANdense(
        prod(kernel_size) * in_channels, out_channels,
        wavelet_name, base_activation;
        norm = norm, is_2d = true,
    )
    return KANConv2D(dense_kernel, kernel_size, stride, dilation, padding)
end

function (c::KANConv2D)(x, ps, st)
    H, W = size(x, 1), size(x, 2)
    in_c, batch_size = size(x, 3), size(x, 4)
    kh, kw = c.kernel_size
    out_h = _conv_out_dim(H, kh, c.padding, c.stride, c.dilation)
    out_w = _conv_out_dim(W, kw, c.padding, c.stride, c.dilation)

    # unfold out: (out_h * out_w, kh * kw * in_c, batch)
    patches = NNlib.unfold(
        x, (kh, kw, in_c, 1);
        stride = c.stride, pad = c.padding, dilation = c.dilation,
    )
    patches = permutedims(patches, (2, 1, 3))  # (kh*kw*in_c, out_h*out_w, batch)

    out, st_k = c.dense_kernel(patches, ps.dense_kernel, st.dense_kernel)
    out_channels = size(out, 1)
    out = reshape(out, out_channels, out_h, out_w, batch_size)
    return permutedims(out, (2, 3, 1, 4)), (dense_kernel = st_k,)
end


struct KANConvTranspose2D{K} <: Lux.AbstractLuxContainerLayer{(:dense_kernel,)}
    dense_kernel::K
    kernel_size::Tuple{Int, Int}
    stride::Int
    dilation::Int
    padding::Int
end

function KANConvTranspose2D(
        in_channels::Int, out_channels::Int, kernel_size::Tuple{Int, Int},
        wavelet_name::String, base_activation::String;
        stride::Int = 1, dilation::Int = 1, padding::Int = 1, norm::Bool = false
    )
    dense_kernel = KANdense(
        prod(kernel_size) * in_channels, out_channels,
        wavelet_name, base_activation;
        norm = norm, is_2d = true,
    )
    return KANConvTranspose2D(dense_kernel, kernel_size, stride, dilation, padding)
end

function (c::KANConvTranspose2D)(x, ps, st)
    x_up = NNlib.upsample_nearest(x, (c.stride, c.stride))
    H_up, W_up = size(x_up, 1), size(x_up, 2)
    in_c, batch_size = size(x_up, 3), size(x_up, 4)
    kh, kw = c.kernel_size
    out_h = _conv_out_dim(H_up, kh, c.padding, c.stride, c.dilation)
    out_w = _conv_out_dim(W_up, kw, c.padding, c.stride, c.dilation)

    patches = NNlib.unfold(
        x_up, (kh, kw, in_c, 1);
        stride = c.stride, pad = c.padding, dilation = c.dilation,
    )
    patches = permutedims(patches, (2, 1, 3))

    out, st_k = c.dense_kernel(patches, ps.dense_kernel, st.dense_kernel)
    out_channels = size(out, 1)
    out = reshape(out, out_channels, out_h, out_w, batch_size)
    return permutedims(out, (2, 3, 1, 4)), (dense_kernel = st_k,)
end
