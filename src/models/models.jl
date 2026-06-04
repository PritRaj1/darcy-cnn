include("cnn.jl")
include("fno.jl")
include("kan_cnn.jl")

create_model(cfg::CNNConfig) = CNN(cfg)
create_model(cfg::FNOConfig) = FNO(cfg)
create_model(cfg::KANCNNConfig) = KANCNN(cfg)
