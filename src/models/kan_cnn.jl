struct KANCNN{E, D} <: Lux.AbstractLuxContainerLayer{(:encoder, :decoder)}
    encoder::E
    decoder::D
end

function KANCNN(cfg::KANCNNConfig)
    h = cfg.hidden_dim
    ew, ea = cfg.encoder_wavelet_names, cfg.encoder_activations
    dw, da = cfg.decoder_wavelet_names, cfg.decoder_activations
    encoder_layers = [
        KANConv2D(1, 2h, (3, 3), ew[1], ea[1]; padding = 1, norm = cfg.norm),
        KANConv2D(2h, 4h, (3, 3), ew[2], ea[2]; padding = 1, norm = cfg.norm),
        KANConv2D(4h, 8h, (3, 3), ew[3], ea[3]; padding = 1, norm = cfg.norm),
    ]
    decoder_layers = [
        KANConvTranspose2D(8h, 4h, (3, 3), dw[1], da[1]; padding = 1, norm = cfg.norm),
        KANConvTranspose2D(4h, 2h, (3, 3), dw[2], da[2]; padding = 1, norm = cfg.norm),
        KANConvTranspose2D(2h, h, (3, 3), dw[3], da[3]; padding = 1, norm = cfg.norm),
        KANConvTranspose2D(h, 1, (3, 3), dw[4], da[4]; padding = 1, norm = false),
    ]
    return KANCNN(Lux.Chain(encoder_layers...), Lux.Chain(decoder_layers...))
end

function (m::KANCNN)(x, ps, st)
    x, st_enc = m.encoder(x, ps.encoder, st.encoder)
    x, st_dec = m.decoder(x, ps.decoder, st.decoder)
    return x, (encoder = st_enc, decoder = st_dec)
end
