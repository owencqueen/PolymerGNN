import torch


class Vector_IV(torch.nn.Module):

    def __init__(self, input_feat, hidden_channels):
        super(Vector_IV, self).__init__()

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_feat, hidden_channels),
            torch.nn.PReLU(),
            torch.nn.Linear(hidden_channels, 1)
        )

    def forward(self, x):
        return self.fc(x)

class Vector_Tg(torch.nn.Module):

    def __init__(self, input_feat, hidden_channels):
        super(Vector_Tg, self).__init__()

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_feat, hidden_channels),
            torch.nn.PReLU(),
        )

        self.TG = torch.nn.Linear(hidden_channels, 1)
        self.mult_factor = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, 1),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        x = self.fc(x)
        return self.TG(x) * self.mult_factor(x)


class Vector_Joint(torch.nn.Module):
    def __init__(self, input_feat, hidden_channels):
        super(Vector_Joint, self).__init__()

        self.fc1 = torch.nn.Linear(input_feat, hidden_channels)
        self.leaky1 = torch.nn.PReLU()
        self.fc2 = torch.nn.Sequential( # IV model
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.PReLU(),
            torch.nn.Linear(hidden_channels, 1)
        )
        self.fc3 = torch.nn.Linear(hidden_channels, 1) # Tg

        self.mult_factor = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, 1),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        x = self.leaky1(self.fc1(x))

        # Run IV:
        x_IV = self.fc2(x)

        # Run Tg (+ multiplying factor):
        x_Tg = torch.exp(self.fc3(x))
        m = self.mult_factor(x)
        x_Tg = x_Tg * m

        return {'IV': x_IV, 'Tg': x_Tg}