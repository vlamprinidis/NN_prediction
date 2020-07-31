# run_graph.sh before using this script
import wget
from tensorflow.core.framework import graph_pb2
from google.protobuf import text_format, json_format
import tensorflow as tf
import os
    
def _from_url(url):
    out = './mygraph.pbtxt'
    os.system('rm -f {}'.format(out))
    wget.download(url,out)

    gdef = graph_pb2.GraphDef()

    with open(out, 'r') as f:
        graph_str = f.read()

    mygraph_msg = text_format.Parse(graph_str, gdef)

    graph_dict = json_format.MessageToDict(mygraph_msg)
    
    os.system('rm {}'.format(out))
        
    return graph_dict

def tf_graph_dict(Model):
    logdir = '/home/ubuntu/gtflow'
    os.system('rm -rf {}'.format(logdir))
    os.makedirs(logdir)

    model = Model.model
    data = Model.tf_data.batch(32)
    
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, write_graph = True, profile_batch=3)

    model.fit(data, steps_per_epoch = 4, epochs = 1, callbacks=[tb_callback])
        
    return _from_url('http://localhost:6008/data/plugin/graphs/graph?run=.')

def torch_graph_dict(Model):
    out = '/home/ubuntu/gtorch'
    os.system('rm -rf {}'.format(out))
    os.makedirs(out)
    
    from torch.utils.tensorboard import SummaryWriter
    
    model, train_loader = Model.model, Model.train_loader

    writer = SummaryWriter(out)

    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    writer.add_graph(model, images)
    writer.close()
    
    return _from_url('http://localhost:6010/data/plugin/graphs/graph?run=.')
