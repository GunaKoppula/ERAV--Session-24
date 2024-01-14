# ERA-SESSION24 -  Reinforcement Learning

## Car Game
- Perform Experiments on different maps for running the car and figuring out the roads.

### Model Architecture

```python
class Network(nn.Module):
   
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30,30)
        self.fc3 = nn.Linear(30, nb_action)
   
    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values
```
### Results:

![image](https://github.com/GunaKoppula/ERAV--Session-24/assets/61241928/f74a946c-0ac2-4ac2-aac2-b6a707797911)

![image](https://github.com/GunaKoppula/ERAV--Session-24/assets/61241928/530967d2-37ed-491b-b6d0-84caeb0c2cd9)


## Reinforcement_UCBerkeley 
- Perform Experiments on puzzle game to achive reward as soon as possible.

### Results:


![reinforcement_result](https://github.com/GunaKoppula/ERAV--Session-24/assets/61241928/bcec4fa5-2a58-44f7-9dd0-47c335c43aee)




