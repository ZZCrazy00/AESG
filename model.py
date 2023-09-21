import torch
import torch.nn.functional as F
from keras.models import Model
from keras.layers import Dense, Input


def DeepAE1(x_train, args):
    input_img = Input(shape=(1373,))
    if args.dataset_name == "MDAD":
        input_img = Input(shape=(1373,))
    elif args.dataset_name == "aBiofilm":
        input_img = Input(shape=(1720,))
    elif args.dataset_name == "DrugVirus":
        input_img = Input(shape=(175,))

    # encoder layers
    encoded = Dense(512, activation='relu')(input_img)
    encoded = Dense(128, activation='relu')(encoded)
    encoder_output = Dense(args.AE_emb)(encoded)

    # decoder layers
    decoded = Dense(128, activation='relu')(encoder_output)
    decoded = Dense(512, activation='relu')(decoded)
    decoded = Dense(591, activation='tanh')(decoded)
    if args.dataset_name == "MDAD":
        decoded = Dense(1373, activation='tanh')(decoded)
    elif args.dataset_name == "aBiofilm":
        decoded = Dense(1720, activation='tanh')(decoded)
    elif args.dataset_name == "DrugVirus":
        decoded = Dense(175, activation='tanh')(decoded)

    # construct the autoencoder model
    autoencoder = Model(inputs=input_img, outputs=decoded)
    encoder = Model(inputs=input_img, outputs=encoder_output)

    # compile autoencoder
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(x_train, x_train, epochs=args.AE_epoch, batch_size=64, shuffle=True)
    encoded_imgs = encoder.predict(x_train)
    return encoder_output, torch.Tensor(encoded_imgs)


def DeepAE2(x_train, args):
    input_img = Input(shape=(1373,))
    if args.dataset_name == "MDAD":
        input_img = Input(shape=(173,))
    elif args.dataset_name == "aBiofilm":
        input_img = Input(shape=(140,))
    elif args.dataset_name == "DrugVirus":
        input_img = Input(shape=(95,))

    # encoder layers
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoder_output = Dense(args.AE_emb)(encoded)

    # decoder layers
    decoded = Dense(64, activation='relu')(encoder_output)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(853, activation='tanh')(decoded)
    if args.dataset_name == "MDAD":
        decoded = Dense(173, activation='tanh')(decoded)
    elif args.dataset_name == "aBiofilm":
        decoded = Dense(140, activation='tanh')(decoded)
    elif args.dataset_name == "DrugVirus":
        decoded = Dense(95, activation='tanh')(decoded)

    # construct the autoencoder model
    autoencoder = Model(inputs=input_img, outputs=decoded)
    encoder = Model(inputs=input_img, outputs=encoder_output)

    # compile autoencoder
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(x_train, x_train, epochs=args.AE_epoch, batch_size=64, shuffle=True)
    encoded_imgs = encoder.predict(x_train)
    return encoder_output, torch.Tensor(encoded_imgs)


class JDASA(torch.nn.Module):
    def __init__(self, args, num_features):
        super(AESG, self).__init__()
        self.subgraph_lin = torch.nn.Linear(args.hops*(args.hops+2), args.s_dim)
        self.bn1 = torch.nn.BatchNorm1d(args.s_dim)

        self.lin1 = torch.nn.Linear(num_features, args.hidden_dim)
        self.lin2 = torch.nn.Linear(args.hidden_dim, args.hidden_dim//2)
        self.bn2 = torch.nn.BatchNorm1d(args.hidden_dim//2)

        hidden_channels = args.s_dim + args.hidden_dim//2
        self.lin = torch.nn.Linear(hidden_channels, 1)

    def feature_forward(self, x):
        x = self.lin1(x)
        x = x[:, 0, :] * x[:, 1, :]
        x = self.lin2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.7)
        return x

    def forward(self, sf, node_features):
        x = self.subgraph_lin(sf)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.7)

        node_features = self.feature_forward(node_features)
        x = torch.cat([x, node_features.to(torch.float)], 1)

        x = self.lin(x)
        return x
