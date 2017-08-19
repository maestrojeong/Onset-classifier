from utils import *

class Deep_Neural_Network():
    def __init__(self, data_num, input_data, output_data, nclasses = 6, train_ratio = 0.8, epoch = 10, batch_size = 10, decay_lr = 10, decay_rate = 0.5):
        self.data_num = data_num	
        self.nclasses = nclasses
        self.input_data = input_data
        self.output_data = onehot(output_data, self.nclasses)
        self.nfeatures = input_data.shape[1]  
        self.train_ratio = train_ratio
        self.epoch = epoch
        self.batch_size = batch_size
        self.decay_lr = decay_lr
        self.decay_rate= decay_rate
        self.data_manager()
        self.sess = tf.Session()
        
    def data_manager(self):
        '''
            randomly part the data to train_data or test_dsat according to the train_ratio
        '''
        train_input_data = []
        train_output_data = []
        test_input_data = []
        test_output_data = []
        for i in range(self.data_num):
            if np.argmax(self.output_data[i]) == 4: # Exclude bass sound
                continue
            if np.random.rand()<self.train_ratio:
                train_input_data.append(self.input_data[i])
                train_output_data.append(self.output_data[i])
            else:
                test_input_data.append(self.input_data[i])
                test_output_data.append(self.output_data[i])
      
        train_input_data = np.array(train_input_data)
        train_output_data = np.array(train_output_data)
        test_input_data = np.array(test_input_data)
        test_output_data = np.array(test_output_data)
        
        self.train_data = {'input' : train_input_data, 'output' : train_output_data}
        self.test_data = {'input' : test_input_data, 'output' : test_output_data}
        print('Train data')
        print('input_data shape : {}'.format(self.train_data['input'].shape))
        print('output_data shape : {}'.format(self.train_data['output'].shape))
        print('Test data')
        print('input_data shape : {}'.format(self.test_data['input'].shape))
        print('output_data shape : {}'.format(self.test_data['output'].shape))
         
    def model(self):

        self.x = tf.placeholder(tf.float32, [None, self.nfeatures], name = 'input')
        self.y = tf.placeholder(tf.float32, [None, self.nclasses], name = 'output')
          
        h1_units = 25
        with tf.variable_scope("layer1"):
            h1 = linear(self.x, self.nfeatures, h1_units, 'tanh') 
        with tf.variable_scope("layer2"):
            y_hat = linear(h1, h1_units, self.nclasses)
        
        tf.add_to_collection("input", self.x)
        tf.add_to_collection("output", y_hat)

        correct = tf.equal(tf.argmax(y_hat, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        self.error = -tf.reduce_mean(self.y*tf.log(tf.clip_by_value(y_hat, clip_value_max = 1-1e-7, clip_value_min = 1e-7))
                            +(1-self.y)*tf.log(tf.clip_by_value(1-y_hat, clip_value_max = 1-1e-7, clip_value_min = 1e-7)))       
        
        self.learning_rate = tf.Variable(1e-1, trainable = False)
        self.update_lr = tf.assign(self.learning_rate, self.decay_rate*self.learning_rate)
        global_step = tf.Variable(0.0, trainable = False)
        self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(loss = self.error, global_step = global_step)
    
    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, '../../models_save/DNN/model')

    def run(self):
        self.sess.run(tf.global_variables_initializer())  

        print("Accuracy : {}".format(self.sess.run(self.accuracy, feed_dict = {self.x : self.test_data['input'], self.y : self.test_data['output']})))
        
        for i in range(self.epoch):
            if i%self.decay_lr == 0:
                self.sess.run(self.update_lr)
                print("Learning_rate : {}".format(self.sess.run(self.learning_rate)))
            temp = shuffle(self.train_data)    
            batch_number = int(len(temp['input'])/self.batch_size)
            for j in range(batch_number):
                self.sess.run(self.train, feed_dict = {self.x : temp['input'][j*self.batch_size:(j+1)*self.batch_size], 
                                        self.y : temp['output'][j*self.batch_size:(j+1)*self.batch_size]})
            
            train_cost = self.sess.run(self.error, feed_dict = {self.x : self.train_data['input'], self.y : self.train_data['output']}) 
            test_cost = self.sess.run(self.error, feed_dict = {self.x : self.test_data['input'], self.y : self.test_data['output']})
            print("Epoch({}/{}) train cost : {}, test cost : {}".format(i, self.epoch, train_cost, test_cost))
         
        print("Accuracy : {}".format(self.sess.run(self.accuracy, feed_dict = {self.x : self.test_data['input'], self.y : self.test_data['output']})))


if __name__ == '__main__':
    
    forward_diff = 0.03
    backward_diff = 0.07
    
    load_file = '../../dataset/dataset_-{}to{}.txt'.format(forward_diff, backward_diff) # Store file here
    save_file = '../../models_save/CONV/info.txt'

    result = pickle_load(load_file) 
    input_data = result['input']
    output_data = result['output']
    sound_type = result['sound_type']

    print(sound_type)
    print(input_data.shape)
    print(output_data.shape)

    DNN = Deep_Neural_Network(data_num = len(output_data),
                        input_data = input_data,
                        output_data = output_data,
                        nclasses = 6,
                        train_ratio = 0.8,
                        epoch = 20,
                        batch_size = 5,
                        decay_lr = 10,
                        decay_rate = 0.2)

    DNN.model()
    DNN.run()
    DNN.save()

    diff = {'forward' : forward_diff, 'backward' : backward_diff, 'sound_type' : sound_type}
    pickle_store(diff, save_file)
