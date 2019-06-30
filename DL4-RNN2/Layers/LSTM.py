import numpy as np
from Layers import FullyConnected


class LSTM():

    def __init__(self,input_size,hidden_size,output_size,bptt_length):
        self.input_size=input_size      # 13
        self.hidden_size=hidden_size    # 7
        self.output_size=output_size    # 5
        self.bptt_length=bptt_length
        self.fcl_f=FullyConnected.FullyConnected(hidden_size + input_size, hidden_size)
        self.fcl_i=FullyConnected.FullyConnected(hidden_size + input_size, hidden_size)
        self.fcl_c_hat = FullyConnected.FullyConnected(hidden_size + input_size, hidden_size)
        self.fcl_o = FullyConnected.FullyConnected(hidden_size + input_size, hidden_size)

        self.fcl_y = FullyConnected.FullyConnected(hidden_size , output_size)
        self.hidden_state=None
        self.cell_state=None
        self.same_sequence=False
        self.last_iter_hidden_state=None
        self.optimizer=None
        self.delta = 1

        pass

    def forward(self,input_tensor):
        self.input_tensor=input_tensor  # 9*13
        self.batch_size=np.shape(self.input_tensor)[0]
        self.ft = np.zeros((self.batch_size, self.hidden_size ))
        self.it = np.zeros((self.batch_size, self.hidden_size ))
        self.c_hat_t = np.zeros((self.batch_size, self.hidden_size ))
        self.ot = np.zeros((self.batch_size, self.hidden_size ))
        self.cell_state = np.zeros((self.batch_size+1,self.hidden_size))

        if self.same_sequence==False:
            self.hidden_state = np.zeros((self.batch_size+1,self.hidden_size))
        else:
            if self.hidden_state is None:
                self.hidden_state = np.zeros((self.batch_size + 1, self.hidden_size))
            else:
                self.hidden_state[0]=self.last_iter_hidden_state

        y_out=np.zeros((self.batch_size,self.output_size))

        for b in range(self.batch_size):
            new_input=np.hstack((np.expand_dims(self.hidden_state[b],0),np.expand_dims(self.input_tensor[b],0)))
            self.ft[b]=self.sigmoid(self.fcl_f.forward(new_input))
            self.it[b]=self.sigmoid(self.fcl_i.forward(new_input))
            self.c_hat_t[b] = np.tanh(self.fcl_c_hat.forward(new_input))
            self.cell_state[b+1]=np.multiply(self.ft[b],self.cell_state[b])+np.multiply(self.it[b],self.c_hat_t[b])
            self.ot[b]=self.sigmoid(self.fcl_o.forward(new_input))
            self.hidden_state[b+1]=np.multiply(self.ot[b],np.tanh(self.cell_state[b+1]))
            y_out[b]=self.fcl_y.forward(np.expand_dims(self.hidden_state[b+1],0))

        self.last_iter_hidden_state=self.hidden_state[-1]

        return y_out

    def toggle_memory(self):
        self.same_sequence=True

    def backward(self,error_tensor):
        self.error_tensor=error_tensor
        back_cell_state=np.zeros((self.batch_size+1,self.hidden_size))
        back_hidden_state=np.zeros((self.batch_size+1,self.hidden_size))
        gra_cell=np.zeros((1,self.hidden_size))
        hidden=np.zeros((1,self.hidden_size))
        self.hx=np.zeros((self.batch_size,self.hidden_size+self.input_size))
        steps=1
        self.gradient_weights_y=np.zeros((self.hidden_size+1, self.output_size))
        self.gradient_weights_f=np.zeros((self.hidden_size+self.input_size+1,self.hidden_size))
        self.gradient_weights_i=np.zeros((self.hidden_size+self.input_size+1,self.hidden_size))
        self.gradient_weights_c_hat=np.zeros((self.hidden_size+self.input_size+1,self.hidden_size))
        self.gradient_weights_o=np.zeros((self.hidden_size+self.input_size+1,self.hidden_size))

        for b in reversed(range(self.batch_size)):
            gradient_y_fcl_h=self.fcl_y.backward(np.expand_dims(self.error_tensor[b],0))
            gradient_y_h = gradient_y_fcl_h + hidden

            gradient_h_o=np.tanh(self.cell_state[b+1])
            gradient_o_x =(self.ot[b]-self.ot[b]**2)

            gradient_h_ct =self.ot[b]*(1-np.tanh(self.cell_state[b+1])**2)
            gradient_h_ct_all=gradient_h_ct*gradient_y_h+gra_cell
            gra_cell=self.ft[b]*gradient_h_ct_all

            gradient_ct_sf=self.cell_state[b]     # C(t-1)
            gradient_sf_x =(self.ft[b]-self.ft[b]**2)

            gradient_ct_si=self.c_hat_t[b]
            gradient_si_x =(self.it[b]-self.it[b]**2)

            gradient_ct_c_hat=self.it[b]
            gradient_c_hat_x = (1 - self.c_hat_t[b] ** 2)

            error_o= gradient_y_h * gradient_h_o * gradient_o_x
            error_c_hat=gradient_h_ct_all*gradient_ct_c_hat * gradient_c_hat_x
            error_i=gradient_h_ct_all*gradient_ct_si * gradient_si_x
            error_f=gradient_h_ct_all*gradient_ct_sf * gradient_sf_x

            eo_hx=self.fcl_o.backward(error_o)
            ec_hat_hx=self.fcl_c_hat.backward(error_c_hat)
            ei_hx=self.fcl_i.backward(error_i)
            eo_fx=self.fcl_f.backward(error_f)
            self.hx[b]=eo_hx+ec_hat_hx+ei_hx+eo_fx
            hidden=self.hx[b, 0:self.hidden_size]

            self.fcl_y.input_tensor = np.expand_dims(np.hstack((self.hidden_state[b+1], 1)),0)

            self.fcl_f.input_tensor=np.expand_dims(np.hstack((self.hidden_state[b],self.input_tensor[b], 1)),0)
            self.fcl_i.input_tensor = np.expand_dims(np.hstack((self.hidden_state[b],self.input_tensor[b], 1)), 0)
            self.fcl_c_hat.input_tensor = np.expand_dims(np.hstack((self.hidden_state[b],self.input_tensor[b], 1)), 0)
            self.fcl_o.input_tensor = np.expand_dims(np.hstack((self.hidden_state[b],self.input_tensor[b], 1)), 0)

            steps+=1
            if steps <= self.bptt_length:
                self.weights_y = self.fcl_y.get_weights()
                self.weights_f = self.fcl_f.get_weights()
                self.weights_i=self.fcl_i.get_weights()
                self.weights_c_hat=self.fcl_c_hat.get_weights()
                self.weights_o=self.fcl_o.get_weights()
                self.get_gradient_weights()     # in bptt all gradient++

        if self.optimizer is not None:
            self.weights_y = self.optimizer.calculate_update(self.delta, self.weights_y, self.gradient_weights_y)
            self.weights_f = self.optimizer.calculate_update(self.delta, self.weights_f, self.gradient_weights_f)
            self.weights_i = self.optimizer.calculate_update(self.delta, self.weights_i, self.gradient_weights_i)
            self.weights_c_hat = self.optimizer.calculate_update(self.delta, self.weights_c_hat, self.gradient_weights_c_hat)
            self.weights_o = self.optimizer.calculate_update(self.delta, self.weights_o, self.gradient_weights_o)
            self.fcl_y.set_weights(self.weights_y)
            self.fcl_f.set_weights(self.weights_f)
            self.fcl_i.set_weights(self.weights_i)
            self.fcl_c_hat.set_weights(self.weights_c_hat)
            self.fcl_o.set_weights(self.weights_o)

        out_x=self.hx[:,self.hidden_size:self.hidden_size+self.input_size+1]

        return out_x

    def initialize(self,weights_initializer,bias_initializer):
        self.fcl_y.initialize(weights_initializer, bias_initializer)
        self.fcl_f.initialize(weights_initializer,bias_initializer)
        self.fcl_i.initialize(weights_initializer,bias_initializer)
        self.fcl_c_hat.initialize(weights_initializer, bias_initializer)
        self.fcl_o.initialize(weights_initializer, bias_initializer)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def get_weights(self):
        wf= self.fcl_f.get_weights()
        wi=self.fcl_i.get_weights()
        wc_hat=self.fcl_c_hat.get_weights()
        wo= self.fcl_o.get_weights()

        weights=np.hstack((wf,wi,wc_hat,wo))
        return weights

    def set_weights(self, weights):
        set_weights=weights
        self.fcl_f.set_weights(set_weights[:,0:self.hidden_size])
        self.fcl_i.set_weights(set_weights[:,self.hidden_size:self.hidden_size*2])
        self.fcl_c_hat.set_weights(set_weights[:,self.hidden_size*2:self.hidden_size*3])
        self.fcl_o.set_weights(set_weights[:,self.hidden_size*3:self.hidden_size*4])

    def get_gradient_weights(self):
       self.gradient_weights_y+=self.fcl_y.get_gradient_weights()

       self.gradient_weights_f+=self.fcl_f.get_gradient_weights()
       self.gradient_weights_i += self.fcl_i.get_gradient_weights()
       self.gradient_weights_c_hat += self.fcl_c_hat.get_gradient_weights()
       self.gradient_weights_o += self.fcl_o.get_gradient_weights()

       self.gradient_weights_hx = np.hstack((self.gradient_weights_f, self.gradient_weights_i,
                                             self.gradient_weights_c_hat,self.gradient_weights_o))
       print('gradient_weights_hx',self.gradient_weights_hx.shape)

       return self.gradient_weights_hx

    def sigmoid(self,input):

        return 1.0/(1.0+np.exp(-input))



