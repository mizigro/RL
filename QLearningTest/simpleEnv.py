class Simple:
    def __init__(self):
        self.n_states = 5
        self.n_actions = 2 

        self.reset()

    def reset(self):
        self.state = 1
        return self.state

    def render(self):
        if self.state==0:
            out='*'
        else:
            out = '^'
        for i in range(1,self.n_states):
            if i == self.state:
                out += '0'
            else:
                out += '-'
        if self.state == self.n_states:
            out += '*'
        else:
            out += 'T'
        print("\r",out,end="")

    def step(self, action):
        r = 0
        done = False
        if action == 1:
            self.state += 1
            if self.state >= self.n_states:
                r = 1
                done = True

        elif action == 0:
            self.state -= 1
            if self.state <= 0:
                # r = -1
                done = True 
        else:
            print("Invalid Action :",action)
            return None
            
        return self.state, r, done
        
