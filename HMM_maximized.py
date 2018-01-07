import re, time, pickle, math, functools

def load_corpus(path):
    reg = re.compile('[^ a-zA-Z]')
    return " ".join([x.lower() for line in open(path, "r") for x in reg.sub('', line).split()])

def load_parameters(path):
    f = open(path, "r")
    p = pickle.Unpickler(f)
    initial, transition, emission = {}, {}, {}
    dicts = p.load()
    d1 = dicts[0]
    initial = {key: math.log(val) for key, val in d1.iteritems()} # initial
    d2 = dicts[1]
    for k in d2:
        inner = d2.get(k)
        transition[k] = {key: math.log(val) for key, val in inner.iteritems()}  # transition
    d3 = dicts[2]
    for k in d3:
        inner = d3.get(k)
        emission[k] = {key: math.log(val) for key, val in inner.iteritems()} # emission
    return (initial, transition, emission)

class HMM(object):
    def __init__(self, probabilities):
        self.initial = probabilities[0]
        self.state_count = len(self.initial)
        self.transition = probabilities[1]
        self.emission = probabilities[2]

    def get_parameters(self):
        i, t, e = {}, {}, {}
        i = {key: math.exp(val) for key, val in self.initial.iteritems()} # initial
        for k in self.transition:
            inner = self.transition.get(k)
            t[k] = {key: math.exp(val) for key, val in inner.iteritems()}  # transition
        for k in self.emission:
            inner = self.emission.get(k)
            e[k] = {key: math.exp(val) for key, val in inner.iteritems()} # emission
        return (i, t, e)

    def forward(self, sequence):
        result = list()
        # initialization
        o_1 = sequence[0]
        last = { s: self.initial.get(s) + self.emission.get(s).get(o_1) for s in xrange(1, self.state_count+1) }
        result.append(last) # t = 0
        # induction
        for t in xrange(1, len(sequence)): # 1 <= t <= T - 1
            d = {}
            char = sequence[t]
            for j in xrange(1, self.state_count+1): # j
                # find a
                x_n = [last.get(i) + self.transition.get(i).get(j) for i in xrange(1, self.state_count+1)]
                sum = self.sum_helper(x_n)
                sum = sum + self.emission.get(j).get(char)
                d[j] = sum
            # update alpha t - 1
            last = d
            result.append(d)        
        return result

    def forward_probability(self, alpha):
        res = 0
        l = [v for k, v in alpha[-1].iteritems()]
        return self.sum_helper(l)

    def backward(self, sequence):
        result = list()
        # initialization
        last = { s: math.log(1) for s in xrange(1, self.state_count+1) }
        result.insert(0, last)  
        # induction     
        for t in reversed(xrange(0, len(sequence) - 1)): # T - 1 >= t >= 1
            d = {}
            next_char = sequence[t+1]
            for i in xrange(1, self.state_count+1): # i
                # find a
                x_n = [self.transition.get(i).get(j) + self.emission.get(j).get(next_char) + \
                       last.get(j) for j in xrange(1, self.state_count+1)]
                d[i] = self.sum_helper(x_n)
            # update beta t+1
            last = d
            result.insert(0, d)                 
        return result

    def backward_probability(self, beta, sequence):
        res = 0
        char = sequence[0]
        l = [self.initial.get(state) + self.emission.get(state).get(char) + v for state, v in beta[0].iteritems()]
        return self.sum_helper(l)

    def forward_backward(self, sequence):
        fwd, bwd, length = self.forward(sequence), self.backward(sequence), len(sequence)
        x_i = {t : self.xi_matrix(t, sequence, fwd, bwd) for t in xrange(0, length - 1)} # store all the x_i besides T-1
        # store gammas = a dictionary of time t to dictionary of gamma at that t for each i
        gammas = {}
        for t in xrange(0, length - 1):
            matrix_t = x_i.get(t)
            gammas[t] = {i: self.gamma_helper(matrix_t.get(i)) for i in xrange(1, self.state_count + 1)}                                 
        # t = T - 1
        last_t = length - 1
        gammas[last_t] = {i : self.gamma_two_helper(last_t, fwd[last_t], bwd[last_t], i) for i in xrange(1, self.state_count + 1)}
       
        chars = self.emission[1]
        # helper dictionary for memoization of emission
        emission_d = {k:[gammas.get(t) for t in xrange(0, length) if sequence[t] == k ] for k in chars}
              
        one = gammas.get(0)        
        # re-estimate p_i, transition and emission
        init, trans, emis = {}, {}, {} 
        for i in xrange(1, self.state_count + 1):     
            init[i] = one.get(i)               
            # denominator stays constant - sum of gamma_t at i
            x_n = [gammas.get(t).get(i) for t in xrange(0, length)]
            e_denom = self.sum_helper(x_n)
            t_denom = self.sum_helper(x_n[:-1])
                       
            d = {}
            for k in chars:
                x_n = [elem.get(i) for elem in emission_d[k]]                
                d[k] = self.sum_helper(x_n) - e_denom                                     
            emis[i] = d
                        
            d = {}
            for j in xrange(1, self.state_count + 1):
                to_sum = [x_i.get(t).get(i).get(j) for t in xrange(0, length - 1)]
                d[j] = self.sum_helper(to_sum) - t_denom              
            trans[i] = d

        return init, trans, emis    

    # takes in the xi_matrix for a particular t at a particular state i
    
    def gamma_helper(self, x_i_t):
        x_n = [x_i_t.get(j) for j in xrange(1, self.state_count + 1)]
        return self.sum_helper(x_n)

    # used only for t = T - 1
    def gamma_two_helper(self, t, alpha_t, beta_t, i):
        numerator = alpha_t.get(i) + beta_t.get(i)       
        x_n = [alpha_t.get(j) + beta_t.get(j) for j in xrange(1, self.state_count + 1)]
        return numerator - self.sum_helper(x_n)

    def xi_matrix(self, t, sequence, alpha, beta):
        next_o = sequence[t+1]      # 0 <= t <= T-1
        alpha_t = alpha[t]          # store alpha t for all i
        beta_t1 = beta[t+1]         # store beta t+1 for all j
        # find a
        x = [(i, j, alpha_t.get(i) + beta_t1.get(j) + self.transition.get(i).get(j) \
                + self.emission.get(j).get(next_o)) for i in xrange(1, self.state_count+1) \
                 for j in xrange(1, self.state_count + 1)]        
        x_n = [i[2] for i in x]                              
        denom = self.sum_helper(x_n)
        xi = {}    
        for i, j, val in x:
            if i not in xi:
                xi[i] = {}
            xi[i][j] = val - denom
        return xi

    def sum_helper(self, x_n):
        a = max(x_n)
        sum = 0     
        for e in x_n:
            sum += math.exp(e - a)     
        return a + math.log(sum)

    def update(self, sequence, cutoff_value):
        delta = float('inf')
        curr_prob = self.forward_probability(self.forward(sequence))
        while delta >= cutoff_value:
            process = self.forward_backward(sequence)
            self.initial, self.transition, self.emission = process[0], process[1], process[2]
            new_prob = self.forward_probability(self.forward(sequence))
            delta = new_prob - curr_prob
            if delta < cutoff_value:
                break
            curr_prob = new_prob
