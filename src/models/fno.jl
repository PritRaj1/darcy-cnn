struct FNO{I, H, O} <: Lux.AbstractLuxContainerLayer{(:input_layer, :hidden_layers, :output_layer)}
    input_layer::I
    hidden_layers::H
    output_layer::O
    pad::Int
end

function FNO(cfg::FNOConfig)
    input_layer = Lux.Conv((1, 1), 3 => cfg.width, get_activation(cfg.activation))
    hidden_layers = Lux.Chain(
        (FNOBlock(cfg.width, cfg.modes1, cfg.modes2, cfg.activation) for _ in 1:(cfg.num_blocks))...,
    )
    output_layer = FNO_MLP(cfg.width, 1, cfg.width * 4, cfg.activation)
    return FNO(input_layer, hidden_layers, output_layer, 9)
end

function (m::FNO)(x, ps, st)
    x = get_grid(x)
    x, st_i = m.input_layer(x, ps.input_layer, st.input_layer)
    x = NNlib.pad_zeros(x, (0, m.pad, 0, m.pad, 0, 0, 0, 0))
    x, st_h = m.hidden_layers(x, ps.hidden_layers, st.hidden_layers)
    x = x[1:(end - m.pad), 1:(end - m.pad), :, :]
    x, st_o = m.output_layer(x, ps.output_layer, st.output_layer)
    return x, (input_layer = st_i, hidden_layers = st_h, output_layer = st_o)
end
