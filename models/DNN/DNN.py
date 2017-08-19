from utils import *

class BasicConfig(object):
    def __init__(self):
        self.train_ratio = 0.8
        self.epoch = 20
        self.batch_size = 5
        self.decay_lr = 10
        self.decay_rate = 0.2

class DeepNeuralNetwork(BasicConfig):
    def __init__(self, input_data, output_data, exclude_label, nclasses, directory):
        BasicConfig.__init__(self)
        
        self.nclasses = nclasses
        self.input_data = input_data
        self.output_data = onehot(output_data, self.nclasses)
        self.exclude_label = exclude_label
        self.directory = directory

        self.data_num = len(output_data)	
        self.nfeatures = input_data.shape[1]  
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
            # Exclude bass and percussion with exclude_label
            if np.argmax(self.output_data[i]) in self.exclude_label:
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
        save_position = os.path.join(self.directory, 'model')
        saver = tf.train.Saver()
        saver.save(self.sess, save_position)

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
    islog= False
    comp_ratio = 1.0

    load_file = '../../dataset/f{}b{}log{}comp{}.pkl'.format(forward_diff, backward_diff, islog, comp_ratio) # Load the file
    save_dir = '../../models_save/DNN/f{}b{}log{}comp{}'.format(forward_diff, backward_diff, islog, comp_ratio) # Store file here
    save_pkl = os.path.join(save_dir, 'info.pkl')
    save_info = os.path.join(save_dir, 'info.txt') 

    create_dir(save_dir)
    result = pickle_load(load_file) 

    input_data = result['input']
    output_data = result['output']
    sound_type = result['sound_type']
    exclude_label = [0, 4]

    print(sound_type)
    print(input_data.shape)
    print(output_data.shape)

    for label in exclude_label:
        print("{} is excluded".format(sound_type[label]))

    DNN = DeepNeuralNetwork(
                        input_data = input_data,
                        output_data = output_data,
                        exclude_label = exclude_label,
                        nclasses = len(sound_type),
                        directory = save_dir
                        )
    DNN.model()
    DNN.run()
    DNN.save()
    
    spec = {'sound_type' : sound_type,
            'exclude_label' : exclude_label}

    txt_store(spec, save_info)
    pickle_store(spec, save_pkl)
